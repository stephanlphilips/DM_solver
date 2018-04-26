arma::cx_vec matrix_dependent_parameter(arma::cx_cube *H_0, int i, int j, arma::cx_mat pertubation_function, double time_step){
	// genertates the matrix H1(i,j) = f_pert(int[H0(i,j)]dt/time_step)
	// f_pert = a0(x-x0)^0 +  a1(x-x1) + a2(x-x2)^2 + ...
	arma::cx_vec amplitude = arma::zeros<arma::cx_vec>(H_0->n_slices);
	for (int k = 0; k < H_0->n_slices; ++k){
		amplitude[k] = poly_function(H_0->at(i,j,k)/time_step, pertubation_function);
	}
	return amplitude;
}

void generate_parameter_dependent_matrices(arma::cx_cube *H0, arma::cx_mat matrix_elem, int i, int j, arma::cx_mat pertubation_function, double time_step){
	arma::cx_vec my_amplitudes = matrix_dependent_parameter(H0, i,j, pertubation_function, time_step);
 
	for (int i = 0; i < my_amplitudes.n_elem; ++i){
		H0->slice(i) += my_amplitudes[i]*matrix_elem;
	}
}

void generate_time_dependent_matrices(arma::cx_cube *H0, arma::Mat<int> loc, double frequency, double start, double stop, int steps){
	if (frequency == 0){
		std::cout << "Note: Time dependent matrix with frequency 0 detected. This integral will be neglected.\n";
		return;
	}
	double delta_t  = (stop-start)/steps;

	// devide by delta t, because now the exponential plays the role (consider constant amp*freq dep -> [intgral] only one freq dep.)
	arma::cx_vec my_amplitudes = integrate_cexp(start, stop, steps, frequency)/delta_t;
	arma::cx_vec my_amplitudes_conj = arma::conj(my_amplitudes);


	for (int i = 0; i < steps; ++i){
		for(int j =0; j < loc.n_rows; ++j){
			// No bounds check here. 
			H0->at(loc(j,0), loc(j,1),i) *= my_amplitudes(i);
			H0->at(loc(j,1), loc(j,0),i) *= my_amplitudes_conj(i);
		}
	}
}