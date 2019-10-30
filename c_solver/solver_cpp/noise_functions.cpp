#include "noise_functions.h"

arma::vec get_white_noise(int steps){
	arma::vec white_noise(steps);

	// Make noise generator
	unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);

	std::normal_distribution<double> tmp(0, 1);

	for (int i = 0; i < steps; ++i){
		white_noise[i] = tmp(generator);
	}

	return white_noise;
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

arma::vec get_noise_from_spectral_density(arma::vec* noise_spectrum, double noise_power, int n_samples){
	/*
	based on https://github.com/felixpatzelt/colorednoise
	*/

	// get real and imaginary noise (imaginary part needed to mess up to phase)
	arma::vec S_real = *noise_spectrum * get_white_noise(noise_spectrum->n_elem);
    arma::vec S_imag = *noise_spectrum * get_white_noise(noise_spectrum->n_elem);
    
    arma::cx_vec S_omega_w_noise = arma::cx_vec(S_real,S_imag);
    if (n_samples % 2 ==0)
    	S_omega_w_noise[0] = std::real(S_omega_w_noise[0]);
    
    arma::cx_vec S_omega_w_noise_full = arma::cx_vec(arma::zeros<arma::vec>(n_samples),arma::zeros<arma::vec>(n_samples));
    S_omega_w_noise_full.subvec(0, noise_spectrum->n_elem - 1 + n_samples%2) =  arma::reverse(S_omega_w_noise.subvec(1-n_samples%2, noise_spectrum->n_elem));
    S_omega_w_noise_full.subvec(noise_spectrum->n_elem - 1 + n_samples%2, n_samples) =  arma::conj(S_omega_w_noise.subvec(0, noise_spectrum->n_elem-1));

    arma::vec noise_values = arma::real(arma::ifft(S_omega_w_noise_full));

    return noise_values/arma::stddev(noise_values)*noise_power;
}

int main(int argc, char const *argv[])
{
	/* code */
	get_gaussian_noise(1000, 1000);
	return 0;
}