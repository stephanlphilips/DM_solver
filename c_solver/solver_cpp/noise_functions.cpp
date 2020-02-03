#include "noise_functions.h"


double get_gaussian_noise(double STD_static){
	// Generator to get gaussian noise distribution
	std::normal_distribution<double> tmp(0, STD_static);

	unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);

	double my_noise = tmp(generator);

	return my_noise;
}

arma::vec get_noise_from_spectral_density(arma::vec* STD_omega, int n_samples){
	/*
	based on https://github.com/felixpatzelt/colorednoise
	
	Args:
		STD_omega : standard deviations for the frequency components (ifft is fastest for a size of 2**n)
		n_samples : number of samples needed for the simulation

	*/

    const std::complex<double> i(0.0,1.0);    

	// #method 1 (does not work as well for some reason?)
	// arma::cx_vec epsilon_omega = *STD_omega % arma::randn(STD_omega->n_elem) %
	// 			arma::exp(i*arma::randu(STD_omega->n_elem)*2*M_PI) * 0.5;

	// arma::cx_vec episilon_f_FFT = arma::cx_vec(arma::zeros<arma::vec>(STD_omega->n_elem*2),arma::zeros<arma::vec>(STD_omega->n_elem*2));

	// episilon_f_FFT.subvec(1, STD_omega->n_elem-1) =  epsilon_omega.subvec(0, STD_omega->n_elem-2);
	// episilon_f_FFT.subvec(STD_omega->n_elem, STD_omega->n_elem*2-1) =  arma::reverse(arma::conj(epsilon_omega));
	// episilon_f_FFT(STD_omega->n_elem) = 0;

	// arma::vec noise_values = arma::real(arma::ifft(episilon_f_FFT));

	// method 2 (should be correct.)
	arma::arma_rng::set_seed_random();
	arma::vec S_omega_real = *STD_omega % arma::randn(STD_omega->n_elem);// * 0.5;
	arma::vec S_omega_imag = *STD_omega % arma::randn(STD_omega->n_elem);// * 0.5;
	
//	std::cout<< "(l,l-1,h-1,h) freq amplitude element C++: " << STD_omega->at(0)<< ", " << STD_omega->at(1) << ", " << STD_omega->at(STD_omega->n_elem-2) << ", " << STD_omega->at(STD_omega->n_elem-1) << "\n";
//	std::cout<< "number of elements in amplitude C++: " << STD_omega->n_elem << "\n";
	arma::cx_vec epsilon_omega = S_omega_real + i * S_omega_imag;
	arma::cx_vec episilon_f_FFT = arma::cx_vec(arma::zeros<arma::vec>(STD_omega->n_elem*2),arma::zeros<arma::vec>(STD_omega->n_elem*2));

	episilon_f_FFT.subvec(1, STD_omega->n_elem-1) =  epsilon_omega.subvec(0, STD_omega->n_elem-2);
	episilon_f_FFT.subvec(STD_omega->n_elem, STD_omega->n_elem*2-1) =  arma::reverse(arma::conj(epsilon_omega));
	episilon_f_FFT(STD_omega->n_elem) = 0;
//	std::cout<< "zero frequency component of noise array C++: " << episilon_f_FFT.at(0) << "\n";
//	arma::vec noise_values = arma::real(arma::ifft(episilon_f_FFT))/std::sqrt(2)*std::sqrt(STD_omega->n_elem);
	arma::vec noise_values = arma::real(arma::ifft(episilon_f_FFT))/std::sqrt(2)*std::sqrt(STD_omega->n_elem*2);
	//std::cout<< "standard deviation of noise array C++: " << arma::stddev(noise_values) << "\n";
//	std::cout<< "length array C++: " << noise_values.n_elem << "\n";


    return noise_values.subvec(0, n_samples-1);

}

arma::vec py_get_noise_from_spectral_density(arma::vec STD_omega, int n_samples){
	return get_noise_from_spectral_density(&STD_omega, n_samples);
}

// int main(int argc, char const *argv[])
// {
// 	arma::vec STD_omega = arma::ones<arma::vec>(50);
// 	// std::cout << STD_omega;
// 	int n_samples = 5;
// 	std::cout << get_noise_from_spectral_density(&STD_omega, n_samples);
// 	return 0;
// }