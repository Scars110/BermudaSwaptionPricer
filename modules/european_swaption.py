# modules/european_swaption.py
import numpy as np
import pandas as pd
from .utils import Utils # Relative import
# No direct import of DiscountCurve class needed; an instance is passed

class EuropeanSwaption:
    def __init__(self, discount_curve_instance, is_payer=True, day_count_convention='ACT/360'):
        self.discount_curve = discount_curve_instance # Instance of your Python DiscountCurve
        self.is_payer = is_payer
        self.day_count_convention = day_count_convention
        if self.discount_curve.valuation_date is None:
            raise ValueError("DiscountCurve instance must have a valuation_date. Call build_curve() first.")
        self.valuation_date = self.discount_curve.valuation_date # pd.Timestamp

    def price_european_swaption(self, expiry_years, tenor_years, strike, model_type='HW1F', parameters=None):
        if parameters is None: raise ValueError("Model parameters must be provided.")

        expiry_date_pdts = self.valuation_date + pd.DateOffset(years=expiry_years)
        maturity_date_pdts = expiry_date_pdts + pd.DateOffset(years=tenor_years)

        if model_type == 'HW1F':
            if len(parameters) != 2: raise ValueError("HW1F requires 2 parameters: [a_x, sigma_x]")
            return self._price_hw1f(self.valuation_date, expiry_date_pdts, maturity_date_pdts, strike, parameters[0], parameters[1])
        elif model_type == 'HW2F':
            if len(parameters) != 5: raise ValueError("HW2F requires 5 parameters: [a_x, a_y, sigma_x, sigma_y, rho]")
            return self._price_hw2f(self.valuation_date, expiry_date_pdts, maturity_date_pdts, strike, *parameters)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}.")

    def _price_hw1f(self, val_date_pdts, exp_date_pdts, mat_date_pdts, K_strike, a_x, sigma_x, payment_frequency=1):
        T0_yf = Utils.year_fraction(val_date_pdts, exp_date_pdts, self.day_count_convention)
        
        # Underlying swap payment dates: from expiry to maturity of swap
        # Utils.generate_payment_dates returns [SwapEffectiveDate, PayDate1, ..., SwapMaturityDate]
        swap_payment_dates = Utils.generate_payment_dates(exp_date_pdts, mat_date_pdts, payment_frequency)

        P_val_exp = self.discount_curve.discount(exp_date_pdts) # P(val_date, expiry_date)

        # Cashflows for the fixed leg of the swap.
        # Utils.bond_cashflows expects payment_dates [T0, T1,...,TN] and returns cfs [0, C1,...,CN]
        fixed_leg_cfs = Utils.bond_cashflows(K_strike, swap_payment_dates, self.day_count_convention)

        PV_fixed_leg = 0.0
        # fixed_leg_cfs[j] is the cashflow at swap_payment_dates[j]
        # We sum PV of cashflows from swap_payment_dates[1] onwards
        for i in range(1, len(swap_payment_dates)):
            PV_fixed_leg += fixed_leg_cfs[i] * self.discount_curve.discount(swap_payment_dates[i])
        
        # Nested helper P(u -> T_i) for HW1F
        # u_yf and Ti_yf are year fractions from val_date_pdts
        def P_u_Ti_hw1(u_yf: float, Ti_yf: float) -> float:
            P0_Ti = self.discount_curve.curve.discount(Ti_yf) # QL curve discount by YF
            P0_u  = self.discount_curve.curve.discount(u_yf)
            if P0_u == 0: return 0.0
            H_val = Utils.H_function(a_x, u_yf, Ti_yf) # u is start, Ti is end for H_function
            exp_term = 0.0
            if a_x == 0: exp_term = -0.5 * (sigma_x**2) * u_yf * (H_val**2)
            else: exp_term = -(sigma_x**2)/(4.0*a_x) * (1.0 - np.exp(-2.0*a_x*u_yf)) * (H_val**2)
            return (P0_Ti / P0_u) * np.exp(exp_term)

        # Duration functions
        def DBx_func(u_yf: float) -> float:
            num, den = 0.0, 0.0
            for i in range(1, len(swap_payment_dates)): # Sum over actual cashflow payments
                cf_date = swap_payment_dates[i]
                cf_amount = fixed_leg_cfs[i]
                Ti_yf = Utils.year_fraction(val_date_pdts, cf_date, self.day_count_convention)
                num += cf_amount * P_u_Ti_hw1(u_yf, Ti_yf) * Utils.H_function(a_x, u_yf, Ti_yf)
                den += cf_amount * P_u_Ti_hw1(u_yf, Ti_yf)
            return num / den if den != 0.0 else 0.0

        DPx_func = lambda u_yf, T_target_yf: Utils.H_function(a_x, u_yf, T_target_yf)
        DBx_fwd_func = lambda u_yf: DBx_func(u_yf) - DPx_func(u_yf, T0_yf)
        var_integrand = lambda u_yf: sigma_x**2 * DBx_fwd_func(u_yf)**2

        integrated_var = Utils.integrate_variance(var_integrand, 0.0, T0_yf)
        sigma_B = np.sqrt(max(0, integrated_var))

        if sigma_B < 1e-9 or P_val_exp == 0: # Effectively zero vol or zero discount
            if self.is_payer: return max(0.0, P_val_exp - PV_fixed_leg) # Payer option to swap fixed for float
            else: return max(0.0, PV_fixed_leg - P_val_exp)             # Receiver option
        
        log_arg = PV_fixed_leg / P_val_exp
        if log_arg <= 0 : # Handle cases where PV_fixed_leg might be zero or negative (highly unlikely for typical rates)
            if self.is_payer: return max(0.0, P_val_exp - PV_fixed_leg)
            else: return max(0.0, PV_fixed_leg - P_val_exp)

        d1 = (np.log(log_arg) + 0.5 * sigma_B**2) / sigma_B
        d2 = d1 - sigma_B

        if self.is_payer: # Payer swaption: an option to PAY the fixed rate K.
                          # This is like an option to SELL the "annuity bond" B(K,Tn) for "reference bond" P(T0)
                          # Price = P(T0) * N(-d2) - B(K,Tn) * N(-d1)
            return P_val_exp * Utils.normal_cdf(-d2) - PV_fixed_leg * Utils.normal_cdf(-d1)
        else: # Receiver swaption: an option to RECEIVE the fixed rate K.
              # This is like an option to BUY the "annuity bond" B(K,Tn) for "reference bond" P(T0)
              # Price = B(K,Tn) * N(d1) - P(T0) * N(d2)
            return PV_fixed_leg * Utils.normal_cdf(d1) - P_val_exp * Utils.normal_cdf(d2)


    def _price_hw2f(self, val_date_pdts, exp_date_pdts, mat_date_pdts, K_strike, a_x, a_y, sigma_x, sigma_y, rho, payment_frequency=1):
        T0_yf = Utils.year_fraction(val_date_pdts, exp_date_pdts, self.day_count_convention)
        swap_payment_dates = Utils.generate_payment_dates(exp_date_pdts, mat_date_pdts, payment_frequency)
        P_val_exp = self.discount_curve.discount(exp_date_pdts)
        fixed_leg_cfs = Utils.bond_cashflows(K_strike, swap_payment_dates, self.day_count_convention)

        PV_fixed_leg = 0.0
        for i in range(1, len(swap_payment_dates)):
            PV_fixed_leg += fixed_leg_cfs[i] * self.discount_curve.discount(swap_payment_dates[i])

        def P_u_Ti_factor(u_yf: float, Ti_yf: float, factor_a: float, factor_sigma: float) -> float:
            P0_Ti = self.discount_curve.curve.discount(Ti_yf)
            P0_u  = self.discount_curve.curve.discount(u_yf)
            if P0_u == 0: return 0.0
            H_val = Utils.H_function(factor_a, u_yf, Ti_yf)
            exp_term = 0.0
            if factor_a == 0: exp_term = -0.5*(factor_sigma**2)*u_yf*(H_val**2)
            else: exp_term = -(factor_sigma**2)/(4.0*factor_a)*(1.0 - np.exp(-2.0*factor_a*u_yf))*(H_val**2)
            return (P0_Ti / P0_u) * np.exp(exp_term)

        def DB_factor_func(u_yf: float, factor_a: float, factor_sigma: float) -> float:
            num, den = 0.0, 0.0
            for i in range(1, len(swap_payment_dates)):
                cf_date = swap_payment_dates[i]
                cf_amount = fixed_leg_cfs[i]
                Ti_yf = Utils.year_fraction(val_date_pdts, cf_date, self.day_count_convention)
                # For DBx, use P_u_Ti with a_x, sigma_x. For DBy, use P_u_Ti with a_y, sigma_y.
                P_uTi_val = P_u_Ti_factor(u_yf, Ti_yf, factor_a, factor_sigma)
                H_factor_val = Utils.H_function(factor_a, u_yf, Ti_yf)
                num += cf_amount * P_uTi_val * H_factor_val
                den += cf_amount * P_uTi_val
            return num / den if den != 0.0 else 0.0
        
        DBx_func_val = lambda u_yf: DB_factor_func(u_yf, a_x, sigma_x) # Pass sigma_x for P_u_Ti_x
        DBy_func_val = lambda u_yf: DB_factor_func(u_yf, a_y, sigma_y) # Pass sigma_y for P_u_Ti_y

        DPx_func_val = lambda u_yf, T_target_yf: Utils.H_function(a_x, u_yf, T_target_yf)
        DPy_func_val = lambda u_yf, T_target_yf: Utils.H_function(a_y, u_yf, T_target_yf)
        
        DBx_fwd_func_val = lambda u_yf: DBx_func_val(u_yf) - DPx_func_val(u_yf, T0_yf)
        DBy_fwd_func_val = lambda u_yf: DBy_func_val(u_yf) - DPy_func_val(u_yf, T0_yf)

        def var_integrand_hw2f(u_yf: float) -> float:
            dbx_fwd = DBx_fwd_func_val(u_yf)
            dby_fwd = DBy_fwd_func_val(u_yf)
            return (sigma_x**2 * dbx_fwd**2 +
                    sigma_y**2 * dby_fwd**2 +
                    2.0 * rho * sigma_x * sigma_y * dbx_fwd * dby_fwd)

        integrated_var = Utils.integrate_variance(var_integrand_hw2f, 0.0, T0_yf)
        sigma_B = np.sqrt(max(0, integrated_var))

        if sigma_B < 1e-9 or P_val_exp == 0:
            if self.is_payer: return max(0.0, P_val_exp - PV_fixed_leg)
            else: return max(0.0, PV_fixed_leg - P_val_exp)

        log_arg = PV_fixed_leg / P_val_exp
        if log_arg <= 0:
            if self.is_payer: return max(0.0, P_val_exp - PV_fixed_leg)
            else: return max(0.0, PV_fixed_leg - P_val_exp)
            
        d1 = (np.log(log_arg) + 0.5 * sigma_B**2) / sigma_B
        d2 = d1 - sigma_B

        if self.is_payer:
            return P_val_exp * Utils.normal_cdf(-d2) - PV_fixed_leg * Utils.normal_cdf(-d1)
        else:
            return PV_fixed_leg * Utils.normal_cdf(d1) - P_val_exp * Utils.normal_cdf(d2)
