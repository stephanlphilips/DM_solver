#ifndef DM_SOLVER_CORE_H
#define DM_SOLVER_CORE_H

#include <armadillo> 
#include "python_c_interface.h"

extern "C" void calculate_evolution(Python_c_Interface* PythonDM);

#endif