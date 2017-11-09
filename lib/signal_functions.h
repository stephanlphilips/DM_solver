class phase_microwave_RWA
{
double amp;
double phase;
double frequency;
double t_start;
double t_stop;
std::string modulation = "block";
double sigma;
double mu;

double get_amplitude(double time){
		if (! modulation.compare("block")){
			return 1;
		}
		if (! modulation.compare("gauss")){
			return gaussian(time);
		}
		return 0;
		}
	

	double gaussian(double time){
		return std::exp(-(time - mu)*(time - mu)/(2*sigma*sigma));
	}
	
public:

	void init(double amplitude, double phi, double freq, double t__start, double t__stop){
		amp = amplitude;
		phase = phi;
		frequency = freq;
		t_start = t__start;
		t_stop = t__stop;
	}
	
	void add_gauss_amp_mod(double sigma_gauss){
		// adds modulation, sigma, descibes the broadness of the pulse, 
		// the gaussian is: e**(x - x0)**2/(sqrt(2)*sigma)**2,
		// where x0 is the the middele of the pulse.
		modulation = "gauss";
		sigma = sigma_gauss;
		mu = (t_stop -t_start)/2;
	}

	arma::cx_vec integrate(double start, double stop, int steps){
		// simple function that performs integration
		arma::vec times = arma::linspace<arma::vec>(start,stop,steps+1);
		arma::cx_vec integration_results = arma::zeros<arma::cx_vec>(steps);

		arma::uword start_index = arma::abs(times - t_start).index_min();
		arma::uword stop_index = arma::abs(times - t_stop).index_min();

		const std::complex<double> j(0, 1);
		const double delta_t = (stop-start)/steps;

		if(frequency == 0.){
			#pragma omp parallel for 
			for (int i = start_index; i < stop_index; ++i){
				integration_results[i] = delta_t*amp*(get_amplitude(times[i]) + get_amplitude(times[i+1]))/2*std::exp(j*phase);
			}
		}
		else{
			#pragma omp parallel for 
			for (int i = start_index; i < stop_index; ++i){
				integration_results[i] = amp*(get_amplitude(times[i]) + get_amplitude(times[i+1]))/2*std::exp(j*phase)/(j*frequency*M_PI*2.)*(
	                        std::exp(j*frequency*2.*M_PI*(times[i +1]))
	                        -std::exp(j*frequency*2.*M_PI*(times[i])));
			}
		}
		return integration_results;
	}
};

class AWG_pulse
{
	double params[4];
	static double f (double x, void* params) {
		double fd_1 = (std::exp((-x + ((double*)params)[1])/((double*)params)[3])+1);
		double fd_2 = (std::exp((x -  ((double*)params)[2])/((double*)params)[3])+1);
		double f = 1/fd_1/fd_2;
		return f;
	}
	// static double f_riemann(double x, double* params){
	// 	double fd_1 = (std::exp((-x + ((double*)params)[1])/((double*)params)[3])+1);
	// 	double fd_2 = (std::exp((x -  ((double*)params)[2])/((double*)params)[3])+1);
	// 	double f = 1/fd_1/fd_2;
	// 	return f;
	// }
public:
	void init(double amplitude, double skew_, double t__start, double t__stop){
		params[0] = amplitude;
		params[1] = t__start;
		params[2] = t__stop;
		params[3] = skew_;
	}
	
	arma::cx_vec integrate(double start, double stop, int steps){
		arma::vec times = arma::linspace<arma::vec>(start,stop,steps+1);
		arma::cx_vec integration_results = arma::zeros<arma::cx_vec>(steps);


		// double result, error;

		// gsl_integration_workspace *workspace  = gsl_integration_workspace_alloc (100);
		// gsl_function F;
		// F.function = &f;
		// F.params = params;

		// #pragma omp parallel for 
		// for (int i = 0; i < times.size()-1; ++i){
		// 	gsl_integration_qags (&F, times[i], times[i+1], 1e-8, 1e-8, 100, workspace, &result, &error); 
		// 	integration_results[i] = result*params[0];
		// }

		// return integration_results;

		// Sometimes convergence problems, but since the function is so simple, we can also just do it the riemann way.
		double delta_t = times[1] - times[0];
		times += delta_t/2;
		for (int i = 0; i < times.size() -1; ++i){
			integration_results[i] = f(times[i], params)*delta_t*params[0];
		}
		return integration_results;
	}
};

