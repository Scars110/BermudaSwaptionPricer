#include "european_swaption_pricer.hpp"
#include "utils.hpp"
#include <ql/time/daycounters/actual360.hpp>
#include <cmath>
#include <algorithm>

double price_hw1f_cpp(
    const QuantLib::Date& val_date, const QuantLib::Date& exp_date, const QuantLib::Date& mat_date,
    const HW1F_Params& params, double K_strike, bool is_payer,
    const boost::shared_ptr<QuantLib::YieldTermStructure>& discount_curve, int payment_frequency
) {
    double a_x = params.a;
    double sigma_x = params.sigma;
    QuantLib::Actual360 dc;
    double T0_yf = Utils::year_fraction(val_date, exp_date, dc);
    auto swap_payment_dates = Utils::generate_payment_dates(exp_date, mat_date, payment_frequency);
    auto fixed_leg_cfs = Utils::bond_cashflows(K_strike, swap_payment_dates, dc);

    double P_val_exp = discount_curve->discount(exp_date);

    double PV_fixed_leg = 0.0;
    for (size_t i = 1; i < swap_payment_dates.size(); ++i) {
        PV_fixed_leg += fixed_leg_cfs[i] * discount_curve->discount(swap_payment_dates[i]);
    }

    auto P_u_Ti_hw1 = [&](double u_yf, double Ti_yf) {
        double P0_Ti = discount_curve->discount(Ti_yf);
        double P0_u = discount_curve->discount(u_yf);
        if (P0_u < 1e-12) return 0.0;
        double H_val = Utils::H_function(a_x, u_yf, Ti_yf);
        double exp_term = (std::abs(a_x) < 1e-9)
            ? -0.5 * std::pow(sigma_x, 2) * u_yf * std::pow(H_val, 2)
            : -(std::pow(sigma_x, 2) / (4.0 * a_x)) * (1.0 - std::exp(-2.0 * a_x * u_yf)) * std::pow(H_val, 2);
        return (P0_Ti / P0_u) * std::exp(exp_term);
    };

    auto DBx_func = [&](double u_yf) {
        double num = 0.0, den = 0.0;
        for (size_t i = 1; i < swap_payment_dates.size(); ++i) {
            double Ti_yf = Utils::year_fraction(val_date, swap_payment_dates[i], dc);
            double p_uti = P_u_Ti_hw1(u_yf, Ti_yf);
            num += fixed_leg_cfs[i] * p_uti * Utils::H_function(a_x, u_yf, Ti_yf);
            den += fixed_leg_cfs[i] * p_uti;
        }
        return (den < 1e-12) ? 0.0 : num / den;
    };

    auto var_integrand = [&](double u_yf) {
        double dbx_fwd = DBx_func(u_yf) - Utils::H_function(a_x, u_yf, T0_yf);
        return std::pow(sigma_x, 2) * std::pow(dbx_fwd, 2);
    };

    double integrated_var = Utils::integrate_variance(var_integrand, 0.0, T0_yf);
    double sigma_B = std::sqrt(std::max(0.0, integrated_var));

    if (sigma_B < 1e-9 || P_val_exp < 1e-12) {
        double intrinsic = is_payer ? (P_val_exp - PV_fixed_leg) : (PV_fixed_leg - P_val_exp);
        return std::max(0.0, intrinsic);
    }

    double log_arg = PV_fixed_leg / P_val_exp;
     if (log_arg <= 0) {
        double intrinsic = is_payer ? (P_val_exp - PV_fixed_leg) : (PV_fixed_leg - P_val_exp);
        return std::max(0.0, intrinsic);
    }
    
    double d1 = (std::log(log_arg) + 0.5 * std::pow(sigma_B, 2)) / sigma_B;
    double d2 = d1 - sigma_B;

    if (is_payer) {
        return P_val_exp * Utils::normal_cdf(-d2) - PV_fixed_leg * Utils::normal_cdf(-d1);
    } else {
        return PV_fixed_leg * Utils::normal_cdf(d1) - P_val_exp * Utils::normal_cdf(d2);
    }
}
double price_hw2f_cpp(
    const QuantLib::Date& val_date, const QuantLib::Date& exp_date, const QuantLib::Date& mat_date,
    const HW2F_Params& params, double K_strike, bool is_payer,
    const boost::shared_ptr<QuantLib::YieldTermStructure>& discount_curve, int payment_frequency
) {
    double a_x = params.a_x;
    double sigma_x = params.sigma_x;
    double a_y = params.a_y;
    double sigma_y = params.sigma_y;
    double rho = params.rho;
    QuantLib::Actual360 dc;
    double T0_yf = Utils::year_fraction(val_date, exp_date, dc);
    auto swap_payment_dates = Utils::generate_payment_dates(exp_date, mat_date, payment_frequency);
    auto fixed_leg_cfs = Utils::bond_cashflows(K_strike, swap_payment_dates, dc);

    // Present value of the floating leg's notional at expiry
    double P_val_exp = discount_curve->discount(exp_date);

    // Present value of the fixed leg cash flows 
    double PV_fixed_leg = 0.0;
    for (size_t i = 1; i < swap_payment_dates.size(); ++i) {
        PV_fixed_leg += fixed_leg_cfs[i] * discount_curve->discount(swap_payment_dates[i]);
    }
    
    // This calculates the price of a zero-coupon bond at a future time 'u'.
    auto P_u_Ti_hw2 = [&](double u_yf, double Ti_yf) {
        double P0_Ti = discount_curve->discount(Ti_yf);
        double P0_u = discount_curve->discount(u_yf);
        if (P0_u < 1e-12) return 0.0;

        // Variance and convariance terms for x(u) and y(u)
        double var_x = (std::abs(a_x) < 1e-9) ? u_yf * sigma_x * sigma_x : (sigma_x * sigma_x) / (2.0 * a_x) * (1.0 - std::exp(-2.0 * a_x * u_yf));
        double var_y = (std::abs(a_y) < 1e-9) ? u_yf * sigma_y * sigma_y : (sigma_y * sigma_y) / (2.0 * a_y) * (1.0 - std::exp(-2.0 * a_y * u_yf));
        double cov_xy = (std::abs(a_x + a_y) < 1e-9) ? u_yf * rho * sigma_x * sigma_y : (rho * sigma_x * sigma_y) / (a_x + a_y) * (1.0 - std::exp(-(a_x + a_y) * u_yf));
        
        double H_x_val = Utils::H_function(a_x, u_yf, Ti_yf);
        double H_y_val = Utils::H_function(a_y, u_yf, Ti_yf);

        double total_variance = H_x_val * H_x_val * var_x + H_y_val * H_y_val * var_y + 2.0 * H_x_val * H_y_val * cov_xy;
        
        return (P0_Ti / P0_u) * std::exp(0.5 * total_variance);
    };

    // Stochastic durations
    auto DBx_func = [&](double u_yf) {
        double num = 0.0, den = 0.0;
        for (size_t i = 1; i < swap_payment_dates.size(); ++i) {
            double Ti_yf = Utils::year_fraction(val_date, swap_payment_dates[i], dc);
            double p_uti = P_u_Ti_hw2(u_yf, Ti_yf);
            num += fixed_leg_cfs[i] * p_uti * Utils::H_function(a_x, u_yf, Ti_yf);
            den += fixed_leg_cfs[i] * p_uti;
        }
        return (den < 1e-12) ? 0.0 : num / den;
    };
    auto DBy_func = [&](double u_yf) {
        double num = 0.0, den = 0.0;
        for (size_t i = 1; i < swap_payment_dates.size(); ++i) {
            double Ti_yf = Utils::year_fraction(val_date, swap_payment_dates[i], dc);
            double p_uti = P_u_Ti_hw2(u_yf, Ti_yf);
            num += fixed_leg_cfs[i] * p_uti * Utils::H_function(a_y, u_yf, Ti_yf);
            den += fixed_leg_cfs[i] * p_uti;
        }
        return (den < 1e-12) ? 0.0 : num / den;
    };
    
    // Integrand for the total variance calculation
    auto var_integrand = [&](double u_yf) {
        double dbx = DBx_func(u_yf);
        double dby = DBy_func(u_yf);
        
        // Stochastic duration of the numeraire P(t, T0)
        double Hx_T0 = Utils::H_function(a_x, u_yf, T0_yf);
        double Hy_T0 = Utils::H_function(a_y, u_yf, T0_yf);

        double dbx_fwd = dbx - Hx_T0;
        double dby_fwd = dby - Hy_T0;

        return std::pow(sigma_x * dbx_fwd, 2) + std::pow(sigma_y * dby_fwd, 2) + 2.0 * rho * sigma_x * sigma_y * dbx_fwd * dby_fwd;
    };

    double integrated_var = Utils::integrate_variance(var_integrand, 0.0, T0_yf);
    double sigma_B = std::sqrt(std::max(0.0, integrated_var));

    // If volatility is near zero, return the intrinsic value.
    if (sigma_B < 1e-9 || P_val_exp < 1e-12) {
        double intrinsic = is_payer ? (P_val_exp - PV_fixed_leg) : (PV_fixed_leg - P_val_exp);
        return std::max(0.0, intrinsic);
    }
    
    double log_arg = PV_fixed_leg / P_val_exp;
    if (log_arg <= 0) { // Should not happen with positive rates/forward
        double intrinsic = is_payer ? (P_val_exp - PV_fixed_leg) : (PV_fixed_leg - P_val_exp);
        return std::max(0.0, intrinsic);
    }

    // Black-Scholes type formula
    double d1 = (std::log(log_arg) + 0.5 * std::pow(sigma_B, 2)) / sigma_B;
    double d2 = d1 - sigma_B;

    if (is_payer) {
        //  V_float * N(-d2) - V_fixed * N(-d1)
        return P_val_exp * Utils::normal_cdf(-d2) - PV_fixed_leg * Utils::normal_cdf(-d1);
    } else {
        // V_fixed * N(d1) - V_float * N(d2)
        return PV_fixed_leg * Utils::normal_cdf(d1) - P_val_exp * Utils::normal_cdf(d2);
    }
}