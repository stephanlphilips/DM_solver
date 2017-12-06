arma::vec get_white_noise(double amplitude, int steps, double time_step){
	arma::vec white_noise(steps);

	// Make noise generator
	unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);

	double total_time = steps*time_step;
	std::normal_distribution<double> tmp(0, amplitude/std::sqrt(time_step));

	for (int i = 0; i < steps; ++i){
		white_noise[i] = tmp(generator);
	}

	return white_noise;
}

arma::vec get_1f_noise(double noise_strength, double alpha, int steps, double time_step){
	// Due to the discretisation of the fourrier transform, we want the noise to not we a frequency component of the simulation,
	// e.g. when we would take the fourrier components of a signal of 100 ns with steps of 1 ns
		// f1 = 1e7
		// f2 = 2e7
		// f... = 4.3e8
	// All these frequencies have in common that if you multiply them by 100ns, you will get an integer number, which means that the 
	// when the 100ns are passed, exactly 2 pi has passed. This is unwanted behavoir. 
	// This is resolved by making noise for a longer time (e.g. 10 times longer) with a random number of steps added on top
	
	// 
	// Note that the method is inefficient.
	// 
	
	// 1) make more steps
	int new_steps = 10*steps;
	// 2) add random amount of steps
	unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> tmp(0, steps);
	new_steps += std::abs(tmp(generator));
	std::cout << new_steps << "\n";
	// 1/f amplitude
	arma::vec freq_amp = fft_freq(new_steps, 1/time_step);
	// Make sure the dc component is zero. 
	freq_amp(0) = 1e300;

	// gaussian white noise generated from gaussian distribution
	arma::vec white_noise(new_steps);
	white_noise = get_white_noise(noise_strength, new_steps, time_step);
	
	// FFT noise
	arma::cx_vec fft_white_noise = arma::fft(white_noise);
	// Note % in armadillo, shur product!
	// note devide alpha by 2 since 1/f relation is related to the power spectrum and ~ V**2
	fft_white_noise = fft_white_noise%arma::pow(arma::abs(1/freq_amp),alpha/2.);
	arma::vec one_f_noise_long = arma::real(arma::ifft(fft_white_noise));

	arma::vec one_f_noise_cut(steps);
	for (int i = 0; i < steps; ++i){
		one_f_noise_cut(i) = one_f_noise_long(i);
	}


	return one_f_noise_cut;
}


arma::vec get_gaussian_noise(double T2, int steps){
	// Generator to get gaussian noise distribution
	std::normal_distribution<double> tmp(0, std::sqrt(2)/T2);

	unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);

	double my_noise = tmp(generator);
	arma::vec gaussian_noise(steps);

	return gaussian_noise.fill(my_noise);
}

class noise{
	// Class describing generalized noise model.
private:
	double T2;
	double noise_amp;
	double alpha;
	//if gaussian effective T2* (standard derivative) ; if exponential, this is the noise amplitude
	int type;
	// type 0 = gaussian noise 
	// type 1 = 1/f noise.
	
	arma::cx_mat noise_matrix;

	std::vector<std::pair<int,int>> param_of_interest;
	std::vector<arma::cx_mat> function_params;
	std::vector<arma::cx_mat> noise_matrix_depencies;
public:
	void init_gauss(arma::cx_mat inp_noise_matrix, double input_T2){
		T2 = input_T2;
		noise_matrix = inp_noise_matrix;
		type = 0;
	}

	void init_pink(arma::cx_mat inp_noise_matrix, double input_noise_amp, double input_alpha){
		noise_matrix = inp_noise_matrix;
		noise_amp = input_noise_amp;
		alpha = input_alpha;
		type = 1;
	}

	void init_white(arma::cx_mat inp_noise_matrix, double input_noise_amp){
		noise_matrix = inp_noise_matrix;
		noise_amp = input_noise_amp;
		type = 2;
	}

	void add_param_dep(std::pair<int,int> param_locations, arma::cx_mat function_dep_param){
		param_of_interest.push_back(param_locations);
		function_params.push_back(function_dep_param);
		noise_matrix_depencies.push_back(noise_matrix);
	}

	void add_param_matrix_dep(arma::cx_mat custom_input_matrix, std::pair<int,int> param_locations, arma::cx_mat function_dep_param){
		param_of_interest.push_back(param_locations);
		function_params.push_back(function_dep_param);
		noise_matrix_depencies.push_back(custom_input_matrix);
	}

	arma::cx_cube get_noise(arma::cx_cube* H0, int steps, double time_step){
		arma::cx_cube noise_matrices = arma::cx_cube(arma::zeros<arma::cube>(noise_matrix.n_rows ,noise_matrix.n_rows ,steps),arma::zeros<arma::cube>(noise_matrix.n_rows ,noise_matrix.n_rows ,steps));

		arma::vec noise_amplitudes(steps);
		if (type == 0)
			noise_amplitudes = get_gaussian_noise(T2, steps);
		if (type == 1)
			noise_amplitudes = get_1f_noise(noise_amp, alpha, steps, time_step);
		if (type == 2)
			noise_amplitudes = get_white_noise(noise_amp, steps, time_step);
		
		for (int i = 0; i < steps; ++i){
			noise_matrices.slice(i) = noise_matrix*noise_amplitudes(i);
		}
		
		for (int i = 0; i < function_params.size(); ++i){
			arma::cx_vec amplitudes = matrix_dependent_parameter(H0, param_of_interest[i].first, param_of_interest[i].second, function_params[i], time_step);
			for (int j = 0; j < noise_matrices.n_slices; ++j){
				noise_matrices.slice(j) += amplitudes(j)*noise_amplitudes(j)*noise_matrix_depencies[i];
			}
		}

		return noise_matrices;
	}
};

