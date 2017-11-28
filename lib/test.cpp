#include "VonNeuman_core.h"

int main(){
	VonNeumannSolver my_solver(2);
	my_solver;

	arma::mat test(5, 2);
	test(0,0) = 1;
	test(0,1) = 0;

	test(1,0) = 2;
	test(1,1) = 1;
	
	test(2,0) = 3;
	test(2,1) = 1;
	
	test(3,0) = 3;
	test(3,1) = 0;
	
	test(4,0) = 5;
	test(4,1) = 2;

	std::cout<< test;
	// arma::cx_mat H1(2,2, arma::fill::randu);
	// arma::cx_mat psi0(2,2, arma::fill::randu);
	// my_solver.add_H1_AWG(test, H1);
	// my_solver.calculate_evolution(psi0, 0.,10,500);
}