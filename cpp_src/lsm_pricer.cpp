#include "lsm_pricer.hpp"
#include "utils.hpp"

// --- En-têtes requis pour les types et fonctions utilisés ---
#include <dlib/matrix.h>
#include <dlib/optimization.h>
#include <ql/time/period.hpp>
#include <ql/interestrate.hpp>
#include <algorithm> 
#include <random>
#include <ctime>
#include <iostream>

// Fonctions utilitaires pour créer les matrices de régression
dlib::matrix<double> create_design_matrix_hw1f(const std::vector<double>& state_values) {
    dlib::matrix<double> A(state_values.size(), 3);
    for (size_t i = 0; i < state_values.size(); ++i) {
        double r = state_values[i];
        A(i, 0) = 1.0; A(i, 1) = r; A(i, 2) = r * r;
    }
    return A;
}

dlib::matrix<double> create_design_matrix_hw2f(const std::vector<std::pair<double, double>>& state_values) {
    dlib::matrix<double> A(state_values.size(), 6);
    for (size_t i = 0; i < state_values.size(); ++i) {
        double x = state_values[i].first;
        double y = state_values[i].second;
        A(i, 0) = 1.0; A(i, 1) = x; A(i, 2) = y;
        A(i, 3) = x * x; A(i, 4) = y * y; A(i, 5) = x * y;
    }
    return A;
}

dlib::matrix<double> simulate_hw1f_paths(
    const HW1F_Params& params,
    double initial_rate,
    const boost::shared_ptr<QuantLib::YieldTermStructure>& discount_curve,
    int num_paths,
    int num_steps,
    double T
) {
    double dt = T / num_steps;
    double sqrt_dt = std::sqrt(dt);

    // Initialiser le générateur de nombres aléatoires
    std::mt19937 generator(static_cast<unsigned int>(std::time(nullptr)));
    std::normal_distribution<double> distribution(0.0, 1.0);

    // Initialiser la matrice de résultats avec dlib
    dlib::matrix<double> paths(num_paths, num_steps + 1);
    dlib::set_colm(paths, 0) = initial_rate; // Définir le taux initial pour tous les chemins

    for (int j = 0; j < num_steps; ++j) {
        double t = j * dt;
        // La dérive theta(t) est calibrée sur la courbe des taux initiale
        double fwd_rate = discount_curve->forwardRate(t, t + dt, QuantLib::Continuous).rate();
        double theta_t = fwd_rate + params.a * discount_curve->zeroRate(t, QuantLib::Continuous).rate();

        for (int i = 0; i < num_paths; ++i) {
            double dW = distribution(generator) * sqrt_dt;
            double prev_r = paths(i, j);
            // Discrétisation d'Euler: dr = (theta(t) - a*r(t))dt + sigma*dW
            paths(i, j + 1) = prev_r + (theta_t - params.a * prev_r) * dt + params.sigma * dW;
        }
    }
    return paths;
}
// À ajouter dans : cpp_src/lsm_pricer.cpp

std::vector<dlib::matrix<double>> simulate_hw2f_paths(
    const HW2F_Params& params,
    double initial_rate,
    const boost::shared_ptr<QuantLib::YieldTermStructure>& discount_curve,
    int num_paths,
    int num_steps,
    double T
) {
    double dt = T / num_steps;
    double sqrt_dt = std::sqrt(dt);

    std::mt19937 generator(static_cast<unsigned int>(std::time(nullptr) + 1)); // Seed différent
    std::normal_distribution<double> distribution(0.0, 1.0);

    dlib::matrix<double> paths_x(num_paths, num_steps + 1);
    dlib::matrix<double> paths_y(num_paths, num_steps + 1);

    // On suppose r(0) = x(0) et y(0) = 0
    dlib::set_colm(paths_x, 0) = initial_rate;
    dlib::set_colm(paths_y, 0) = 0.0;

    for (int j = 0; j < num_steps; ++j) {
        double t = j * dt;
        double fwd_rate = discount_curve->forwardRate(t, t + dt, QuantLib::Continuous).rate();
        double theta_t = fwd_rate + (params.a_x + params.a_y) * discount_curve->zeroRate(t, QuantLib::Continuous).rate();

        for (int i = 0; i < num_paths; ++i) {
            double dW1 = distribution(generator);
            double dW2 = distribution(generator);

            // Créer des mouvements browniens corrélés
            double dWx = dW1;
            double dWy = params.rho * dW1 + std::sqrt(1.0 - params.rho * params.rho) * dW2;

            double prev_x = paths_x(i, j);
            double prev_y = paths_y(i, j);

            paths_x(i, j + 1) = prev_x + (0.5 * theta_t - params.a_x * prev_x) * dt + params.sigma_x * dWx * sqrt_dt;
            paths_y(i, j + 1) = prev_y + (0.5 * theta_t - params.a_y * prev_y) * dt + params.sigma_y * dWy * sqrt_dt;
        }
    }
    return {paths_x, paths_y};
}


double calculate_swap_npv(
    const HW1F_Params& params, const boost::shared_ptr<QuantLib::YieldTermStructure>& discount_curve,
    double r_t, double t, const QuantLib::Date& swap_start_date, int tenor_years,
    double K, int payment_frequency, bool is_payer
) {
    QuantLib::DayCounter day_counter = discount_curve->dayCounter();
    QuantLib::Date ref_date = discount_curve->referenceDate();

    // --- NPV de la Jambe Fixe ---
    double npv_fixed = 0.0;
    auto payment_dates = Utils::generate_payment_dates(swap_start_date, swap_start_date + QuantLib::Period(tenor_years, QuantLib::Years), payment_frequency);
    
    for (size_t i = 0; i < payment_dates.size(); ++i) {
        if (payment_dates[i] <= swap_start_date) continue; // Ignorer les flux passés
        
        double tau = day_counter.yearFraction(payment_dates[i - 1], payment_dates[i]);
        double cash_flow = K * tau; // Notional de 1

        // Actualiser le flux jusqu'à l'instant t
        double T_i = day_counter.yearFraction(ref_date, payment_dates[i]);
        double p_market_T = discount_curve->discount(T_i);
        double p_market_t = discount_curve->discount(t);
        double p_analytic = Utils::hw1f_zcb_price_analytic(t, T_i, params, r_t);
        
        double discount_factor = p_analytic * (p_market_T / p_market_t);
        npv_fixed += cash_flow * discount_factor;
    }
    // Ajout du notional à la fin
    double T_end_fixed = day_counter.yearFraction(ref_date, payment_dates.back());
    double p_market_T_end_fixed = discount_curve->discount(T_end_fixed);
    double p_market_t_fixed = discount_curve->discount(t);
    double p_analytic_end_fixed = Utils::hw1f_zcb_price_analytic(t, T_end_fixed, params, r_t);
    npv_fixed += p_analytic_end_fixed * (p_market_T_end_fixed / p_market_t_fixed);


    // --- NPV de la Jambe Variable ---
    // Pour un swap standard, NPV_float(t) = P(t, T_start) - P(t, T_end)
    // Ici, T_start est l'instant t lui-même, donc P(t,t) = 1.
    // La valeur de la jambe flottante à l'instant t est simplement 1 (le notional)
    // moins la valeur d'une obligation zéro-coupon qui paie 1 à la fin du swap.
    double T_end_float = day_counter.yearFraction(ref_date, payment_dates.back());
    double p_market_T_end_float = discount_curve->discount(T_end_float);
    double p_market_t_float = discount_curve->discount(t);
    double p_analytic_end_float = Utils::hw1f_zcb_price_analytic(t, T_end_float, params, r_t);
    double p_t_T_end = p_analytic_end_float * (p_market_T_end_float / p_market_t_float);

    double npv_variable = 1.0 - p_t_T_end;

    return is_payer ? (npv_variable - npv_fixed) : (npv_fixed - npv_variable);
}

double calculate_swap_npv(
    const HW2F_Params& params, const boost::shared_ptr<QuantLib::YieldTermStructure>& discount_curve,
    double x_t, double y_t, double t, const QuantLib::Date& swap_start_date, int tenor_years,
    double K, int payment_frequency, bool is_payer
) {
    QuantLib::DayCounter day_counter = discount_curve->dayCounter();
    QuantLib::Date ref_date = discount_curve->referenceDate();

    // --- NPV de la Jambe Fixe ---
    double npv_fixed = 0.0;
    auto payment_dates = Utils::generate_payment_dates(swap_start_date, swap_start_date + QuantLib::Period(tenor_years, QuantLib::Years), payment_frequency);
    
    for (size_t i = 0; i < payment_dates.size(); ++i) {
        if (payment_dates[i] <= swap_start_date) continue;
        
        double tau = day_counter.yearFraction(payment_dates[i - 1], payment_dates[i]);
        double cash_flow = K * tau;

        double T_i = day_counter.yearFraction(ref_date, payment_dates[i]);
        double p_market_T = discount_curve->discount(T_i);
        double p_market_t = discount_curve->discount(t);
        double p_analytic = Utils::hw2f_zcb_price_analytic(t, T_i, params, x_t, y_t);
        
        double discount_factor = p_analytic * (p_market_T / p_market_t);
        npv_fixed += cash_flow * discount_factor;
    }
    double T_end_fixed = day_counter.yearFraction(ref_date, payment_dates.back());
    double p_market_T_end_fixed = discount_curve->discount(T_end_fixed);
    double p_market_t_fixed = discount_curve->discount(t);
    double p_analytic_end_fixed = Utils::hw2f_zcb_price_analytic(t, T_end_fixed, params, x_t, y_t);
    npv_fixed += p_analytic_end_fixed * (p_market_T_end_fixed / p_market_t_fixed);


    // --- NPV de la Jambe Variable ---
    double T_end_float = day_counter.yearFraction(ref_date, payment_dates.back());
    double p_market_T_end_float = discount_curve->discount(T_end_float);
    double p_market_t_float = discount_curve->discount(t);
    double p_analytic_end_float = Utils::hw2f_zcb_price_analytic(t, T_end_float, params, x_t, y_t);
    double p_t_T_end = p_analytic_end_float * (p_market_T_end_float / p_market_t_float);

    double npv_variable = 1.0 - p_t_T_end;

    return is_payer ? (npv_variable - npv_fixed) : (npv_fixed - npv_variable);
}

// #############################################################################
// #                          FONCTIONS DE RÉGRESSION LSM                        #
// #############################################################################

// --- RÉGRESSION LSM POUR HW1F ---
dlib::matrix<double> run_lsm_regression(
    const HW1F_Params& params, const boost::shared_ptr<QuantLib::YieldTermStructure>& discount_curve,
    int num_paths, int num_steps, const std::vector<double>& exercise_times,
    double K, int payment_frequency, bool is_payer
) {
    double T = exercise_times.back();
    double dt = T / num_steps;
    double initial_rate = discount_curve->zeroRate(0.0, QuantLib::Continuous).rate();
    auto paths = simulate_hw1f_paths(params, initial_rate, discount_curve, num_paths, num_steps, T);

    dlib::matrix<double> cash_flows(num_paths, num_steps + 1);
    cash_flows = 0.0;
    dlib::matrix<double> betas(exercise_times.size(), 3);
    betas = 0.0;
    
    std::vector<int> exercise_steps;
    for(double time : exercise_times) exercise_steps.push_back(static_cast<int>(std::round(time / dt)));
    QuantLib::Date ref_date = discount_curve->referenceDate();

    // Boucle à rebours sur les dates d'exercice
    for (int i = exercise_steps.size() - 1; i >= 0; --i) {
        int t_step = exercise_steps[i];
        double t = t_step * dt;
        QuantLib::Date current_date = ref_date + QuantLib::Period(static_cast<int>(t * 365.25), QuantLib::Days);

        dlib::matrix<double> exercise_values(num_paths, 1);
        std::vector<double> regression_x;
        std::vector<double> regression_b;

        for (long p = 0; p < num_paths; ++p) {
            exercise_values(p) = std::max(0.0, calculate_swap_npv(params, discount_curve, paths(p, t_step), t, current_date, 10, K, payment_frequency, is_payer));
            
            int next_t_step = (i == exercise_steps.size() - 1) ? t_step : exercise_steps[i + 1];
            double future_cf = (i == exercise_steps.size() - 1) ? exercise_values(p) : cash_flows(p, next_t_step);
            double discounted_future_cf = future_cf * std::exp(-paths(p, t_step) * dt * (next_t_step - t_step));

            if (exercise_values(p) > 1e-6) {
                regression_x.push_back(paths(p, t_step));
                regression_b.push_back(discounted_future_cf);
            }
        }
        
        if (regression_x.size() > 3) {
            dlib::matrix<double> A = create_design_matrix_hw1f(regression_x);
            dlib::matrix<double> b = dlib::mat(regression_b);
            dlib::matrix<double, 3, 1> beta_t = dlib::pinv(A) * b;
            dlib::set_rowm(betas, i) = dlib::trans(beta_t);
        }

        for (long p = 0; p < num_paths; ++p) {
             double r = paths(p, t_step);
             dlib::matrix<double, 1, 3> basis_funcs;
             basis_funcs = 1.0, r, r*r;
             double continuation_value = dlib::dot(dlib::rowm(betas, i), dlib::trans(basis_funcs));

             if (exercise_values(p) > continuation_value) {
                 cash_flows(p, t_step) = exercise_values(p);
             } else {
                 int next_t_step = (i == exercise_steps.size() - 1) ? t_step : exercise_steps[i + 1];
                 cash_flows(p, t_step) = cash_flows(p, next_t_step) * std::exp(-r * dt * (next_t_step - t_step));
             }
        }
    }
    return betas;
}

// --- RÉGRESSION LSM POUR HW2F ---
dlib::matrix<double> run_lsm_regression(
    const HW2F_Params& params, const boost::shared_ptr<QuantLib::YieldTermStructure>& discount_curve,
    int num_paths, int num_steps, const std::vector<double>& exercise_times,
    double K, int payment_frequency, bool is_payer
) {
    double T = exercise_times.back();
    double dt = T / num_steps;
    double initial_rate = discount_curve->zeroRate(0.0, QuantLib::Continuous).rate();
    auto path_vectors = simulate_hw2f_paths(params, initial_rate, discount_curve, num_paths, num_steps, T);
    auto& paths_x = path_vectors[0];
    auto& paths_y = path_vectors[1];

    dlib::matrix<double> cash_flows(num_paths, num_steps + 1);
    cash_flows = 0.0;
    dlib::matrix<double> betas(exercise_times.size(), 6);
    betas = 0.0;

    std::vector<int> exercise_steps;
    for(double time : exercise_times) exercise_steps.push_back(static_cast<int>(std::round(time / dt)));
    QuantLib::Date ref_date = discount_curve->referenceDate();

    for (int i = exercise_steps.size() - 1; i >= 0; --i) {
        int t_step = exercise_steps[i];
        double t = t_step * dt;
        QuantLib::Date current_date = ref_date + QuantLib::Period(static_cast<int>(t * 365.25), QuantLib::Days);

        dlib::matrix<double> exercise_values(num_paths, 1);
        std::vector<std::pair<double, double>> regression_xy;
        std::vector<double> regression_b;
        
        for (long p = 0; p < num_paths; ++p) {
            double x = paths_x(p, t_step);
            double y = paths_y(p, t_step);
            exercise_values(p) = std::max(0.0, calculate_swap_npv(params, discount_curve, x, y, t, current_date, 10, K, payment_frequency, is_payer));
            
            int next_t_step = (i == exercise_steps.size() - 1) ? t_step : exercise_steps[i + 1];
            double future_cf = (i == exercise_steps.size() - 1) ? exercise_values(p) : cash_flows(p, next_t_step);
            double discounted_future_cf = future_cf * std::exp(-(x + y) * dt * (next_t_step - t_step));
            
            if (exercise_values(p) > 1e-6) {
                regression_xy.push_back({x, y});
                regression_b.push_back(discounted_future_cf);
            }
        }
        
        if (regression_xy.size() > 6) {
            dlib::matrix<double> A = create_design_matrix_hw2f(regression_xy);
            dlib::matrix<double> b = dlib::mat(regression_b);
            // CORRECTION : Utilisation de pinv
            dlib::matrix<double, 6, 1> beta_t = dlib::pinv(A) * b;
            dlib::set_rowm(betas, i) = dlib::trans(beta_t);
        }

        for (long p = 0; p < num_paths; ++p) {
            double x = paths_x(p, t_step);
            double y = paths_y(p, t_step);
            dlib::matrix<double, 1, 6> basis_funcs;
            basis_funcs = 1.0, x, y, x*x, y*y, x*y;
            double continuation_value = dlib::dot(dlib::rowm(betas, i), dlib::trans(basis_funcs));

            if (exercise_values(p) > continuation_value) {
                cash_flows(p, t_step) = exercise_values(p);
            } else {
                int next_t_step = (i == exercise_steps.size() - 1) ? t_step : exercise_steps[i + 1];
                cash_flows(p, t_step) = cash_flows(p, next_t_step) * std::exp(-(x + y) * dt * (next_t_step - t_step));
            }
        }
    }
    return betas;
}
// #############################################################################
// #                       FONCTIONS DE PRICING AVEC LSM                       #
// #############################################################################

// --- PRICING AVEC LA POLITIQUE LSM POUR HW1F ---
double price_with_lsm_policy(
    const HW1F_Params& params, const dlib::matrix<double>& betas,
    const boost::shared_ptr<QuantLib::YieldTermStructure>& discount_curve,
    int num_paths, int num_steps, const std::vector<double>& exercise_times,
    double K, int payment_frequency, bool is_payer
) {
    double T = exercise_times.back();
    double dt = T / num_steps;
    double initial_rate = discount_curve->zeroRate(0.0, QuantLib::Continuous).rate();

    auto paths = simulate_hw1f_paths(params, initial_rate, discount_curve, num_paths, num_steps, T);
    dlib::matrix<double> cash_flows(num_paths, 1);
    QuantLib::Date ref_date = discount_curve->referenceDate();

    // Calculer le payoff à la dernière date d'exercice
    int last_step = static_cast<int>(std::round(T / dt));
    double last_t = last_step * dt;
    QuantLib::Date last_exercise_date = ref_date + QuantLib::Period(static_cast<int>(last_t * 365.25), QuantLib::Days);
    for (long p = 0; p < num_paths; ++p) {
        cash_flows(p) = std::max(0.0, calculate_swap_npv(params, discount_curve, paths(p, last_step), last_t, last_exercise_date, 10, K, 1, is_payer));
    }

    std::vector<int> exercise_steps;
    for(double time : exercise_times) exercise_steps.push_back(static_cast<int>(std::round(time / dt)));

    // Boucle à rebours pour appliquer la politique d'exercice
    for (int i = exercise_steps.size() - 2; i >= 0; --i) {
        int t_step = exercise_steps[i];
        double t = t_step * dt;
        QuantLib::Date current_date = ref_date + QuantLib::Period(static_cast<int>(t * 365.25), QuantLib::Days);

        // Actualiser les cash flows futurs
        for(long p = 0; p < num_paths; ++p) {
            cash_flows(p) *= std::exp(-paths(p, t_step) * dt * (exercise_steps[i+1]-t_step));
        }

        // Appliquer la politique d'exercice
        for (long p = 0; p < num_paths; ++p) {
            double exercise_value = std::max(0.0, calculate_swap_npv(params, discount_curve, paths(p, t_step), t, current_date, 10, K, 1, is_payer));
            
            dlib::matrix<double, 1, 3> basis_funcs;
            double r = paths(p, t_step);
            basis_funcs = 1.0, r, r*r;
            double continuation_value = dlib::dot(dlib::rowm(betas, i), dlib::trans(basis_funcs));

            if (exercise_value > continuation_value) {
                cash_flows(p) = exercise_value;
            }
        }
    }

    // Calculer le prix final : moyenne des payoffs actualisés à t=0
    double price = dlib::mean(cash_flows) * discount_curve->discount(exercise_times[0]);
    return price;
}

// --- PRICING AVEC LA POLITIQUE LSM POUR HW2F ---
double price_with_lsm_policy(
    const HW2F_Params& params, const dlib::matrix<double>& betas,
    const boost::shared_ptr<QuantLib::YieldTermStructure>& discount_curve,
    int num_paths, int num_steps, const std::vector<double>& exercise_times,
    double K, int payment_frequency, bool is_payer
) {
    double T = exercise_times.back();
    double dt = T / num_steps;
    double initial_rate = discount_curve->zeroRate(0.0, QuantLib::Continuous).rate();

    // 1. Simuler de nouveaux chemins pour x et y
    auto path_vectors = simulate_hw2f_paths(params, initial_rate, discount_curve, num_paths, num_steps, T);
    auto& paths_x = path_vectors[0];
    auto& paths_y = path_vectors[1];

    dlib::matrix<double> cash_flows(num_paths, 1);
    QuantLib::Date ref_date = discount_curve->referenceDate();

    // 2. Payoff à la dernière date d'exercice
    int last_step = static_cast<int>(std::round(T / dt));
    double last_t = last_step * dt;
    QuantLib::Date last_exercise_date = ref_date + QuantLib::Period(static_cast<int>(last_t * 365.25), QuantLib::Days);
    for (long p = 0; p < num_paths; ++p) {
        cash_flows(p) = std::max(0.0, calculate_swap_npv(params, discount_curve, paths_x(p, last_step), paths_y(p, last_step), last_t, last_exercise_date, 10, K, 1, is_payer));
    }

    std::vector<int> exercise_steps;
    for(double time : exercise_times) exercise_steps.push_back(static_cast<int>(std::round(time / dt)));

    // 3. Boucle à rebours
    for (int i = exercise_steps.size() - 2; i >= 0; --i) {
        int t_step = exercise_steps[i];
        double t = t_step * dt;
        QuantLib::Date current_date = ref_date + QuantLib::Period(static_cast<int>(t * 365.25), QuantLib::Days);

        for(long p = 0; p < num_paths; ++p) {
            double r = paths_x(p, t_step) + paths_y(p, t_step);
            cash_flows(p) *= std::exp(-r * dt * (exercise_steps[i+1]-t_step));
        }

        for (long p = 0; p < num_paths; ++p) {
            double x = paths_x(p, t_step);
            double y = paths_y(p, t_step);
            double exercise_value = std::max(0.0, calculate_swap_npv(params, discount_curve, x, y, t, current_date, 10, K, 1, is_payer));
            
            dlib::matrix<double, 1, 6> basis_funcs;
            basis_funcs = 1.0, x, y, x*x, y*y, x*y;
            double continuation_value = dlib::dot(dlib::rowm(betas, i), dlib::trans(basis_funcs));

            if (exercise_value > continuation_value) {
                cash_flows(p) = exercise_value;
            }
        }
    }
    
    // 4. Calculer le prix final
    double price = dlib::mean(cash_flows) * discount_curve->discount(exercise_times[0]);
    return price;
}