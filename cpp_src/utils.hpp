#ifndef UTILS_HPP
#define UTILS_HPP

#include <ql/time/date.hpp>
#include <ql/time/daycounter.hpp>
#include <vector>
#include <functional>
#include "european_swaption_pricer.hpp"

namespace Utils {
    QuantLib::Time year_fraction(const QuantLib::Date& d1, const QuantLib::Date& d2, const QuantLib::DayCounter& dc);
    
    std::vector<QuantLib::Date> generate_payment_dates(const QuantLib::Date& start_date, const QuantLib::Date& end_date, int payment_frequency);
    
    std::vector<double> bond_cashflows(double strike, const std::vector<QuantLib::Date>& payment_dates, const QuantLib::DayCounter& dc, double notional = 1.0);
    
    double H_function(double a, double t, double T);
    
    double integrate_variance(const std::function<double(double)>& f, double a, double b, int n = 100);
    
    double normal_cdf(double x);
    
    double hw2f_zcb_price_analytic(
        double t, double T,
        const HW2F_Params& params,
        double x_t, double y_t
    );
    double hw1f_zcb_price_analytic(
        double t, double T,
        const HW1F_Params& params,
        double r_t
    );

}

#endif