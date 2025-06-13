#include "utils.hpp"
#include <ql/time/calendars/target.hpp>
#include <ql/math/distributions/normaldistribution.hpp>
#include <cmath>

namespace Utils {
    QuantLib::Time year_fraction(const QuantLib::Date& d1, const QuantLib::Date& d2, const QuantLib::DayCounter& dc) {
        return dc.yearFraction(d1, d2);
    }

    std::vector<QuantLib::Date> generate_payment_dates(const QuantLib::Date& start_date, const QuantLib::Date& end_date, int payment_frequency) {
        std::vector<QuantLib::Date> dates;
        dates.push_back(start_date);
        QuantLib::Calendar calendar = QuantLib::TARGET();
        QuantLib::Period freq(12 / payment_frequency, QuantLib::Months);
        QuantLib::Date current_date = start_date;
        while (current_date < end_date) {
            current_date = calendar.advance(current_date, freq);
            dates.push_back(std::min(current_date, end_date));
        }
        return dates;
    }

    std::vector<double> bond_cashflows(double strike, const std::vector<QuantLib::Date>& payment_dates, const QuantLib::DayCounter& dc, double notional) {
        if (payment_dates.size() < 2) return {0.0};
        std::vector<double> cashflows(payment_dates.size(), 0.0);
        for (size_t i = 1; i < payment_dates.size(); ++i) {
            double tau = year_fraction(payment_dates[i-1], payment_dates[i], dc);
            cashflows[i] = strike * tau * notional;
        }
        cashflows.back() += notional;
        return cashflows;
    }

    double H_function(double a, double t, double T) {
        if (std::abs(a) < 1e-9) return T - t;
        return (1.0 - std::exp(-a * (T - t))) / a;
    }

    double integrate_variance(const std::function<double(double)>& f, double a, double b, int n) {
        double h = (b - a) / n;
        double sum = 0.5 * (f(a) + f(b));
        for (int i = 1; i < n; ++i) sum += f(a + i * h);
        return sum * h;
    }

    double normal_cdf(double x) {
        QuantLib::CumulativeNormalDistribution cnd;
        return cnd(x);
    }
    double hw1f_zcb_price_analytic(
        double t, double T,
        const HW1F_Params& params,
        double r_t
    ) {
        double H_val = Utils::H_function(params.a, t, T);
        
        double variance_term = (std::abs(params.a) < 1e-9)
            ? 0.5 * params.sigma * params.sigma * t * H_val * H_val
            : (params.sigma * params.sigma) / (2.0 * params.a * params.a) * (1 - std::exp(-2.0 * params.a * t)) * H_val * H_val;
        
        return std::exp(-H_val * r_t + 0.5 * variance_term);
    }


    double hw2f_zcb_price_analytic(
        double t, double T,
        const HW2F_Params& params,
        double x_t, double y_t
    ) {
        // Variance terms for x(t) and y(t)
        double var_x = (std::abs(params.a_x) < 1e-9) ? t * params.sigma_x * params.sigma_x : (params.sigma_x * params.sigma_x) / (2.0 * params.a_x) * (1.0 - std::exp(-2.0 * params.a_x * t));
        double var_y = (std::abs(params.a_y) < 1e-9) ? t * params.sigma_y * params.sigma_y : (params.sigma_y * params.sigma_y) / (2.0 * params.a_y) * (1.0 - std::exp(-2.0 * params.a_y * t));
        
        // Covariance term for x(t) and y(t)
        double cov_xy = (std::abs(params.a_x + params.a_y) < 1e-9) ? t * params.rho * params.sigma_x * params.sigma_y
                        : (params.rho * params.sigma_x * params.sigma_y) / (params.a_x + params.a_y) * (1.0 - std::exp(-(params.a_x + params.a_y) * t));
        
        double H_x_val = Utils::H_function(params.a_x, t, T);
        double H_y_val = Utils::H_function(params.a_y, t, T);

        double total_variance = H_x_val * H_x_val * var_x + H_y_val * H_y_val * var_y + 2.0 * H_x_val * H_y_val * cov_xy;
        
        return std::exp(-H_x_val * x_t - H_y_val * y_t + 0.5 * total_variance);
    }
    
}