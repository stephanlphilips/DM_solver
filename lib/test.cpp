#include "VonNeuman_core.h"

int main(){
	VonNeumannSolver my_solver(2);
	my_solver;

	arma::mat test(6, 2);
	test(0,0) = 10e-9;
	test(0,1) = 0;

	test(1,0) = 20e-9;
	test(1,1) = 1;
	
	test(2,0) = 30e-9;
	test(2,1) = 1;
	
	test(3,0) = 30e-9;
	test(3,1) = 0;
	
	test(4,0) = 50e-9;
	test(4,1) = 0;
	test(5,0) = 50e-9;
	test(5,1) = 2;

	arma::cube cut_off_data(2,3,1);
	// b = 0.05644846  0.11289692  0.05644846
	// a = 1.         -1.22465158  0.45044543
	cut_off_data(0,0,0) =  0.05644846;
	cut_off_data(0,1,0) =  0.11289692;
	cut_off_data(0,2,0) =  0.05644846;
	cut_off_data(1,0,0) =  1.;
	cut_off_data(1,1,0) = -1.22465158;
	cut_off_data(1,2,0) =  0.45044543;

	std::cout<< cut_off_data;
	arma::cx_mat H1(2,2, arma::fill::randu);
	arma::cx_mat psi0(2,2, arma::fill::randu);
	my_solver.add_H1_AWG(test, H1, cut_off_data);
	my_solver.calculate_evolution(psi0, 0.,100e-9,500);
}  