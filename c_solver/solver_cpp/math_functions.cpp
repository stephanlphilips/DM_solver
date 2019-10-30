#include "math_functions.h"


std::complex<double> poly_function(std::complex<double> x, arma::cx_mat pertubation_function){
	// a0(x-x0)^0 +  a1(x-x1) + a2(x-x2)^2 + ...
	// matrix as [[a0, x0], ... ,[an, xn]]
	std::complex<double> return_value = 0;
	for (int i = 0; i < pertubation_function.n_cols; ++i){
		return_value += pertubation_function(0,i)*std::pow((x-pertubation_function(1,i)),i);
	}
	return return_value;
}

arma::cx_mat custom_matrix_exp(arma::cx_mat input_matrix){
	// ok, but can be more efficient using the pade method.
	// uses now matrix scaling in combination with a taylor
	int accuracy = 10;
	
	const double norm_val = arma::norm(input_matrix, "inf");
    
    const double log2_val = (norm_val > 0.0) ? double(std::log2(norm_val)) : double(0);
    
    int exponent = int(0);  std::frexp(log2_val, &exponent);

    const int s = int( (std::max)(int(0), exponent + int(10)) );

    input_matrix = input_matrix/double(std::pow(double(2), double(s)));
    
    arma::mat tmp1(input_matrix.n_rows,input_matrix.n_rows);
    tmp1.eye();
    arma::mat tmp2(input_matrix.n_rows,input_matrix.n_rows);
    tmp2.zeros();
    arma::cx_mat tmp(tmp1,tmp2);
	arma::cx_mat output_matrix(tmp1,tmp2);

    double factorial_i = 1.0;

	for(int i = 1; i < accuracy; i++) {
		factorial_i = factorial_i * i;
		tmp *= input_matrix;
		output_matrix += tmp/factorial_i;
	}

	for(int i=0; i < s; ++i)  { output_matrix = output_matrix*output_matrix; }

	return output_matrix;
}

arma::cx_vec integrate_cexp(double start, double stop, int steps, double frequency){
		arma::vec times = arma::linspace<arma::vec>(start,stop,steps+1);
		arma::cx_vec integration_results = arma::zeros<arma::cx_vec>(steps);

		const std::complex<double> j(0, 1);

		for (int i = 0; i < steps; ++i)
		{
			integration_results[i] = 1./(j*frequency*M_PI*2.)*(
                        std::exp(j*frequency*2.*M_PI*(times[i +1]))
                        -std::exp(j*frequency*2.*M_PI*(times[i])));
		}

		return integration_results;
}

arma::vec fft_freq(int n, double f){
	// n = number of points
	// f = frequency
	// return frequency components of the fourrier transform. (inspired from numpy implementation)
	// my_freq = [0, 1, ...,   n/2-1,     -n/2, ..., -1] *f/n   if n is even
	// my_freq = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] *f/n   if n is odd

	// determine number of elements in the left side of the sequence.
	int N = int(n/2) + n%2 -1;

	// Genreate left and right side of the sequence
	arma::vec my_freq_1 = arma::linspace<arma::vec>(0, N, N + 1);
	arma::vec my_freq_2 = arma::linspace<arma::vec>(- int(n/2) , -1, n-N-1);
	
	// Put the data together and make frequency absolute.
	arma::vec my_freq;
	my_freq = arma::join_cols<arma::mat>(my_freq_1,my_freq_2);
	my_freq /= n/f;
	
	return my_freq;
}
