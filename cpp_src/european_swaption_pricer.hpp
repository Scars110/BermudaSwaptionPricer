#ifndef EUROPEAN_SWAPTION_PRICER_HPP
#define EUROPEAN_SWAPTION_PRICER_HPP

#include <ql/time/date.hpp>
#include <ql/termstructures/yieldtermstructure.hpp>
#include <boost/shared_ptr.hpp>

struct HW1F_Params {
    double a;
    double sigma;
};

struct HW2F_Params {
    double a_x;
    double sigma_x;
    double a_y;
    double sigma_y;
    double rho;
};

// Déclarations des fonctions de pricing
double price_hw1f_cpp(
    const QuantLib::Date& val_date, const QuantLib::Date& exp_date, const QuantLib::Date& mat_date,
    const HW1F_Params& params, double K_strike, bool is_payer,
    const boost::shared_ptr<QuantLib::YieldTermStructure>& discount_curve, int payment_frequency
);

double price_hw2f_cpp(
    const QuantLib::Date& val_date, const QuantLib::Date& exp_date, const QuantLib::Date& mat_date,
    const HW2F_Params& params, double K_strike, bool is_payer,
    const boost::shared_ptr<QuantLib::YieldTermStructure>& discount_curve, int payment_frequency
);
#endif