class magnetic_noise{
private:
	double T2;
	arma::cx_mat noise_matrix;

	std::vector<std::pair<int,int>> param_of_interest;
	std::vector<arma::cx_mat> function_params;
	std::vector<arma::cx_mat> noise_matrix_depencies;
public:
	void init(arma::cx_mat inp_noise_matrix, double input_T2){
		T2 = input_T2;
		noise_matrix = inp_noise_matrix;
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

		std::normal_distribution<double> tmp(0, 1/T2/std::sqrt(2));

		unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);

		double my_noise = tmp(generator);
		noise_matrices.each_slice() += noise_matrix * my_noise;
		
		for (int i = 0; i < noise_matrix_depencies.size(); ++i)
		{
			arma::cx_vec amplitudes = matrix_dependent_parameter(H0, param_of_interest[i].first, param_of_interest[i].second, function_params[i] , time_step);
			#pragma omp parallel for 
			for (int j = 0; j < noise_matrices.n_slices; ++j){
				noise_matrices.slice(j) += my_noise*amplitudes(j)*noise_matrix_depencies[i];
			}
		}
		return noise_matrices;
	}
};


class noise{
private:
	double T2;
	double noise_amp;
	//if gaussian effective T2* (standard derivative) ; if exponential, this is the noise amplitude
	int type;
	// type 0 = gaussian noise 
	// type 1 = 1/f noise.
	
	arma::cx_mat noise_matrix;

	std::vector<std::pair<int,int>> param_of_interest;
	std::vector<arma::cx_mat> function_params;
	std::vector<arma::cx_mat> noise_matrix_depencies;
public:
	void init(arma::cx_mat inp_noise_matrix, double input_T2){
		// default is gaussian ..
		T2 = input_T2;
		noise_matrix = inp_noise_matrix;
		type = 0;
	}

	void init(arma::cx_mat inp_noise_matrix, double input_noise_amp, int noise_type){
		if (noise_type == 0){
			T2 = input_noise_amp;
		}
		if (noise_type == 1){
			noise_amp = input_noise_amp; 
		}
		noise_matrix = inp_noise_matrix;
		type = noise_type;
			
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

		std::normal_distribution<double> tmp(0, 1/T2/std::sqrt(2));

		unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);

		double my_noise = tmp(generator);
		noise_matrices.each_slice() += noise_matrix * my_noise;
		
		for (int i = 0; i < noise_matrix_depencies.size(); ++i)
		{
			arma::cx_vec amplitudes = matrix_dependent_parameter(H0, param_of_interest[i].first, param_of_interest[i].second, function_params[i] , time_step);
			#pragma omp parallel for 
			for (int j = 0; j < noise_matrices.n_slices; ++j){
				noise_matrices.slice(j) += my_noise*amplitudes(j)*noise_matrix_depencies[i];
			}
		}
		return noise_matrices;
	}
};

class one_f_noise{
private:
	arma::cx_mat matrix_element;
	double noise_strength;
	arma::Mat<int> param_of_interest;
	arma::cx_cube function_params;
	double alpha = 1;

public:
	void init(arma::cx_mat noise_matrix, double amp_noise){
		matrix_element = noise_matrix;
		noise_strength = amp_noise;
	}
	void init(arma::cx_mat noise_matrix, double amp_noise, double new_alpha){
		matrix_element = noise_matrix;
		noise_strength = amp_noise;
		alpha = new_alpha;
	}
	void init(arma::cx_mat noise_matrix, double amp_noise, arma::Mat<int> param_locations, arma::cx_cube function_dep_param){
		noise_strength = amp_noise;
		matrix_element = noise_matrix;
		param_of_interest = param_locations;
		function_params = function_dep_param;
	}
	void init(arma::cx_mat noise_matrix, double amp_noise, arma::Mat<int> param_locations, arma::cx_cube function_dep_param,double new_alpha){
		noise_strength = amp_noise;
		matrix_element = noise_matrix;
		param_of_interest = param_locations;
		function_params = function_dep_param;
		alpha = new_alpha;
	}

	arma::cx_cube get_noise(arma::cx_cube* H0, int steps, double time_step){
		// for every time a different noise matrix.
		arma::cx_cube my_noisy_matrices = arma::cx_cube(arma::zeros<arma::cube>(matrix_element.n_rows,matrix_element.n_rows,steps),arma::zeros<arma::cube>(matrix_element.n_rows,matrix_element.n_rows,steps));
		
		// 1/f amplitude
		arma::vec freq_amp = fft_freq(steps, 1/time_step);
		// Make sure the dc component is zero. #justabignumber
		freq_amp(0) = 1e308;

		// gaussian white noise generated from gaussian distribution
		arma::vec white_noise(steps);

		// Make noise generator
		unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);
	
		std::normal_distribution<double> tmp(0, noise_strength);
		for (int i = 0; i < steps; ++i){
			white_noise[i] = tmp(generator);
		}

		arma::cx_vec fft_white_noise = arma::fft(white_noise);
		// Note % in armadillo, shur product!
		// note devide alpha by 2 since 1/f relation is related to the power spectrum and ~ V**2
		fft_white_noise = fft_white_noise%arma::pow(arma::abs(1/freq_amp),alpha/2.);
		arma::vec one_f_noise = arma::real(arma::ifft(fft_white_noise));
		
		for (int i = 0; i < steps; ++i){
			my_noisy_matrices.slice(i) = matrix_element*one_f_noise[i];
		}

		for (int i = 0; i < function_params.n_slices; ++i){
			arma::cx_vec amplitudes = matrix_dependent_parameter(H0, param_of_interest(0,i), param_of_interest(1,i), function_params.slice(i), time_step);
			for (int j = 0; j < my_noisy_matrices.n_slices; ++j){
				my_noisy_matrices.slice(j) = amplitudes(j)*one_f_noise(j)*matrix_element;
			}
		}
		return my_noisy_matrices;
	}
};
