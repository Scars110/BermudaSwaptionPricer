#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <boost/shared_ptr.hpp>
#include <ql/time/date.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/period.hpp>
#include <ql/termstructures/yieldtermstructure.hpp>

#include "european_swaption_pricer.hpp"
#include "lsm_pricer.hpp"

namespace py = pybind11;
using namespace py::literals;

// Helper function to extract C++ pointer from Python QuantLib object 
struct SwigPyObject { PyObject_HEAD void *ptr; };
boost::shared_ptr<QuantLib::YieldTermStructure> get_ql_curve_ptr_from_py_obj(const py::object& py_curve) {
    PyObject* py_ptr_obj = PyObject_GetAttrString(py_curve.ptr(), "this");
    if (!py_ptr_obj) throw py::error_already_set();
    void* swig_void_ptr = reinterpret_cast<SwigPyObject*>(py_ptr_obj)->ptr;
    Py_DECREF(py_ptr_obj);
    if (!swig_void_ptr) { throw std::runtime_error("The SWIG object's internal pointer is null."); }
    auto* ql_ptr_ptr = static_cast<boost::shared_ptr<QuantLib::YieldTermStructure>*>(swig_void_ptr);
    if (!ql_ptr_ptr) { throw std::runtime_error("Failed to cast pointer from QuantLib Python object."); }
    return *ql_ptr_ptr;
}

// Définition du Module Python
PYBIND11_MODULE(fast_pricer, m) {
    m.doc() = "Moteur C++ pour la valorisation de swaptions européennes et bermudiennes (LSM)";

    // --- 1. Binding des structures de paramètres ---
    py::class_<HW1F_Params>(m, "HW1F_Params")
        .def(py::init<>())
        .def_readwrite("a", &HW1F_Params::a)
        .def_readwrite("sigma", &HW1F_Params::sigma);

    py::class_<HW2F_Params>(m, "HW2F_Params")
        .def(py::init<>())
        .def_readwrite("a_x", &HW2F_Params::a_x)
        .def_readwrite("sigma_x", &HW2F_Params::sigma_x)
        .def_readwrite("a_y", &HW2F_Params::a_y)
        .def_readwrite("sigma_y", &HW2F_Params::sigma_y)
        .def_readwrite("rho", &HW2F_Params::rho);

    // --- 2. Bindings pour les pricers de swaptions européennes ---
m.def("price_hw1f",
    [](const py::object& dc_obj, int vd, int vm, int vy, int em, int ty, double k, 
       double a, double s, // Accepte 'a' et 's' depuis Python
       bool p, int pf) {
        
        auto curve = get_ql_curve_ptr_from_py_obj(dc_obj.attr("curve"));
        HW1F_Params params{a, s}; 
        
        QuantLib::Date val_date(vd, static_cast<QuantLib::Month>(vm), vy);
        QuantLib::Date exp_date = QuantLib::TARGET().advance(val_date, QuantLib::Period(em, QuantLib::Months));
        QuantLib::Date mat_date = QuantLib::TARGET().advance(exp_date, QuantLib::Period(ty, QuantLib::Years));
        
        return price_hw1f_cpp(val_date, exp_date, mat_date, params, k, p, curve, pf);
    }, 
    "py_discount_curve"_a, "val_day"_a, "val_month"_a, "val_year"_a, 
    "expiry_months"_a, "tenor_years"_a, "strike"_a, 
    "a_x"_a, "sigma_x"_a, 
    "is_payer"_a, "payment_frequency"_a = 1
);


m.def("price_hw2f",
    [](const py::object& dc_obj, int vd, int vm, int vy, int em, int ty, double k,
       double ax, double sx, double ay, double sy, double r, bool p, int pf) {
       auto curve = get_ql_curve_ptr_from_py_obj(dc_obj.attr("curve"));
       HW2F_Params params{ax, sx, ay, sy, r};
        QuantLib::Date val_date(vd, static_cast<QuantLib::Month>(vm), vy);
        QuantLib::Date exp_date = QuantLib::TARGET().advance(val_date, QuantLib::Period(em, QuantLib::Months));
        QuantLib::Date mat_date = QuantLib::TARGET().advance(exp_date, QuantLib::Period(ty, QuantLib::Years));
        return price_hw2f_cpp(val_date, exp_date, mat_date, params, k, p, curve, pf);
    },
    "py_discount_curve"_a, "val_day"_a, "val_month"_a, "val_year"_a,
    "expiry_months"_a, "tenor_years"_a, "strike"_a,
    "a_x"_a, "sigma_x"_a, "a_y"_a, "sigma_y"_a, "rho"_a,
    "is_payer"_a, "payment_frequency"_a = 1
);

    // 3. Bindings pour les fonctions LSM 
    m.def("run_lsm_regression",
        [](const HW1F_Params& params, const py::object& dc_obj, int num_paths, int num_steps, const std::vector<double>& exercise_times, double K, int payment_frequency, bool is_payer) {
            auto curve = get_ql_curve_ptr_from_py_obj(dc_obj.attr("curve"));
            return run_lsm_regression(params, curve, num_paths, num_steps, exercise_times, K, payment_frequency, is_payer);
        }, "params"_a, "discount_curve_obj"_a, "num_paths"_a, "num_steps"_a, "exercise_times"_a, "K"_a, "payment_frequency"_a, "is_payer"_a
    );

    m.def("run_lsm_regression",
        [](const HW2F_Params& params, const py::object& dc_obj, int num_paths, int num_steps, const std::vector<double>& exercise_times, double K, int payment_frequency, bool is_payer) {
            auto curve = get_ql_curve_ptr_from_py_obj(dc_obj.attr("curve"));
            return run_lsm_regression(params, curve, num_paths, num_steps, exercise_times, K, payment_frequency, is_payer);
        }, "params"_a, "discount_curve_obj"_a, "num_paths"_a, "num_steps"_a, "exercise_times"_a, "K"_a, "payment_frequency"_a, "is_payer"_a
    );
    
    m.def("price_with_lsm_policy",
        [](const HW1F_Params& params, const dlib::matrix<double>& betas, const py::object& dc_obj, int num_paths, int num_steps, const std::vector<double>& exercise_times, double K, int payment_frequency, bool is_payer) {
            auto curve = get_ql_curve_ptr_from_py_obj(dc_obj.attr("curve"));
            return price_with_lsm_policy(params, betas, curve, num_paths, num_steps, exercise_times, K, payment_frequency, is_payer);
        }, "params"_a, "betas"_a, "discount_curve_obj"_a, "num_paths"_a, "num_steps"_a, "exercise_times"_a, "K"_a, "payment_frequency"_a, "is_payer"_a
    );
        
    m.def("price_with_lsm_policy",
        [](const HW2F_Params& params, const dlib::matrix<double>& betas, const py::object& dc_obj, int num_paths, int num_steps, const std::vector<double>& exercise_times, double K, int payment_frequency, bool is_payer) {
            auto curve = get_ql_curve_ptr_from_py_obj(dc_obj.attr("curve"));
            return price_with_lsm_policy(params, betas, curve, num_paths, num_steps, exercise_times, K, payment_frequency, is_payer);
        }, "params"_a, "betas"_a, "discount_curve_obj"_a, "num_paths"_a, "num_steps"_a, "exercise_times"_a, "K"_a, "payment_frequency"_a, "is_payer"_a
    );
}