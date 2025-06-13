# modules/calibration_problem.py
import numpy as np
import QuantLib as ql
from .utils import Utils
import fast_pricer

class CalibrationProblem:
    """
    
    Encapsulates the data and cost function for the Hull-White model calibration.
    Dynamically calculate ATM strikes and market prices for a given valuation date and set of market volatilities.
    """
    def __init__(self, swaption_metadata, market_vols_series, discount_curve, model_type='HW1F', is_payer=True):
        self.swaption_metadata = swaption_metadata
        self.market_vols = market_vols_series
        self.dc = discount_curve
        self.val_date_ql = self.dc.curve.referenceDate()
        self.model_type = model_type
        self.is_payer = is_payer
        
        self.strikes = []
        self.market_prices = []
        self._calculate_inputs()

    def _calculate_inputs(self):
        """Calculates ATM strikes and market prices for the entire basket."""
        temp_prices = []
        for swaption in self.swaption_metadata:
            # Read expiry in months and tenor in years from the metadata
            expiry_months = swaption['expiry_months']
            tenor_years = swaption['tenor_years']
            
            normal_vol = self.market_vols.get(swaption['name'], 0.0)

            # Calculate expiry date using months
            expiry_date = self.val_date_ql + ql.Period(expiry_months, ql.Months)
            expiry_years_fraction = self.dc.curve.dayCounter().yearFraction(self.val_date_ql, expiry_date)
            
            # 1. Dynamically calculate the At-The-Money (ATM) strike
            atm_strike = Utils.calculate_forward_swap_rate(self.dc, expiry_date, tenor_years)
            self.strikes.append(atm_strike)
            
            # 2. Calculate the market price using this ATM strike
            discount_factor = self.dc.discount(expiry_date)
            market_price = Utils.bachelier_pricer(
                F=atm_strike,
                K=atm_strike,
                T=expiry_years_fraction,
                sigma_normal=normal_vol,
                df=discount_factor,
                is_payer=self.is_payer
            )
            temp_prices.append(market_price)
        
        self.market_prices = np.array(temp_prices)

    def cost_function(self, params):
        """Calculates the Sum of Squared Relative Errors (SSRE)."""
        model_prices = []
        for i, swaption in enumerate(self.swaption_metadata):
            # Arguments for the C++ pricer must match the new signature
            pricer_args = {
                'py_discount_curve': self.dc,
                'val_day': self.val_date_ql.dayOfMonth(), 
                'val_month': self.val_date_ql.month(), 
                'val_year': self.val_date_ql.year(),
                'expiry_months': swaption['expiry_months'], # Pass months
                'tenor_years': swaption['tenor_years'],     # Pass years
                'strike': self.strikes[i],
                'is_payer': self.is_payer
            }
            if self.model_type == 'HW1F':
                pricer_args.update({'a_x': params[0], 'sigma_x': params[1]})
                price = fast_pricer.price_hw1f(**pricer_args)
            elif self.model_type == 'HW2F':
                pricer_args.update({
                    'a_x': params[0], 'sigma_x': params[1],
                    'a_y': params[2], 'sigma_y': params[3], 'rho': params[4]
                })
                price = fast_pricer.price_hw2f(**pricer_args)
            model_prices.append(price)
        
        model_prices = np.array(model_prices)
        relative_errors = np.where(self.market_prices > 1e-9, (model_prices - self.market_prices) / self.market_prices, 0)
        return np.sum(relative_errors**2)
