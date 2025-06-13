#ifndef LSM_PRICER_HPP
#define LSM_PRICER_HPP

#include <vector>
#include <dlib/matrix.h>
#include <ql/termstructures/yieldtermstructure.hpp>
#include <boost/shared_ptr.hpp>

struct HW1F_Params;
struct HW2F_Params;

// FONCTIONS DE SIMULATION 
dlib::matrix<double> simulate_hw1f_paths(
    const HW1F_Params& params, double initial_rate,
    const boost::shared_ptr<QuantLib::YieldTermStructure>& discount_curve,
    int num_paths, int num_steps, double T
);

std::vector<dlib::matrix<double>> simulate_hw2f_paths(
    const HW2F_Params& params, double initial_rate,
    const boost::shared_ptr<QuantLib::YieldTermStructure>& discount_curve,
    int num_paths, int num_steps, double T
);


// FONCTIONS DE CALCUL DE NPV DU SWAP SOUS-JACENT
double calculate_swap_npv(
    const HW1F_Params& params, const boost::shared_ptr<QuantLib::YieldTermStructure>& discount_curve,
    double r_t, double t, const QuantLib::Date& swap_start_date,
    int tenor_years, double K, int payment_frequency, bool is_payer
);

double calculate_swap_npv(
    const HW2F_Params& params, const boost::shared_ptr<QuantLib::YieldTermStructure>& discount_curve,
    double x_t, double y_t, double t, const QuantLib::Date& swap_start_date,
    int tenor_years, double K, int payment_frequency, bool is_payer
);


// FONCTIONS DE L'ALGORITHME LSM 

/**
 Exécute l'induction à rebours de l'algorithme LSM pour dériver les coefficients de régression. 
 return Une matrice où chaque ligne correspond à un pas de temps et contient les coefficients de la fonction de continuation.
 */
dlib::matrix<double> run_lsm_regression(
    const HW1F_Params& params, const boost::shared_ptr<QuantLib::YieldTermStructure>& discount_curve,
    int num_paths, int num_steps, const std::vector<double>& exercise_times,
    double K, int payment_frequency, bool is_payer
);

// Version surchargée pour HW2F
dlib::matrix<double> run_lsm_regression(
    const HW2F_Params& params, const boost::shared_ptr<QuantLib::YieldTermStructure>& discount_curve,
    int num_paths, int num_steps, const std::vector<double>& exercise_times,
    double K, int payment_frequency, bool is_payer
);

/**
 * Calcule le prix de la swaption bermudienne en utilisant une nouvelle simulation et la politique d'exercice optimale (coefficients de régression).
 */
double price_with_lsm_policy(
    const HW1F_Params& params, const dlib::matrix<double>& betas,
    const boost::shared_ptr<QuantLib::YieldTermStructure>& discount_curve,
    int num_paths, int num_steps, const std::vector<double>& exercise_times,
    double K, int payment_frequency, bool is_payer
);

// Version surchargée pour HW2F
double price_with_lsm_policy(
    const HW2F_Params& params, const dlib::matrix<double>& betas,
    const boost::shared_ptr<QuantLib::YieldTermStructure>& discount_curve,
    int num_paths, int num_steps, const std::vector<double>& exercise_times,
    double K, int payment_frequency, bool is_payer
);


#endif