cmake_minimum_required(VERSION 3.12)
project(fast_pricer_project)

list(APPEND CMAKE_PREFIX_PATH "C:/local/C++/libraries/QuantLib-install")
set(Python_EXECUTABLE "C:/Programs/AANACONDA/python.exe" CACHE PATH "Path to a specific python executable")


add_subdirectory(pybind11)

find_package(PythonLibs REQUIRED)
find_package(QuantLib REQUIRED)
find_package(dlib REQUIRED)     

# Créer le module Python 
pybind11_add_module(fast_pricer
    cpp_src/utils.cpp
    cpp_src/european_swaption_pricer.cpp
    cpp_src/lsm_pricer.cpp         
    cpp_src/bindings.cpp
)

# Lier les bibliothèques nécessaires 
target_link_libraries(fast_pricer PRIVATE
    QuantLib::QuantLib
    dlib::dlib                      
)

# Définir le standard C++
target_compile_features(fast_pricer PRIVATE cxx_std_20)