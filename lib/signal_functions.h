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
			for (int i = start_index; i < stop_index; ++i){
				integration_results[i] = delta_t*amp*(get_amplitude(times[i]) + get_amplitude(times[i+1]))/2*std::exp(j*phase);
			}
		}
		else{
			for (int i = start_index; i < stop_index; ++i){
				integration_results[i] = amp*(get_amplitude(times[i]) + get_amplitude(times[i+1]))/2*std::exp(j*phase)/(j*frequency*M_PI*2.)*(
	                        std::exp(j*frequency*2.*M_PI*(times[i +1]))
	                        -std::exp(j*frequency*2.*M_PI*(times[i])));
			}
		}
		return integration_results;
	}
};

class AWG_pulse_old
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
		times -= delta_t/2;
		for (int i = 0; i < times.size() -1; ++i){
			integration_results[i] = f(times[i], params)*delta_t*params[0];
		}
		return integration_results;
	}
};

class AWG_pulse{
	/* 
		Class to be used to construct linear pulses for AWG's.
		Contains filter functions to simulate the effect of the limited bandwith of AWG's 
		The bandwith is incorporated by a FIR/IIR filter.
	*/
private:
	// 2xn array that contains timestamps and amplitudes.
	arma::mat amp_data;
	// matrix containing wich elemements the pulse should effect.
	arma::cx_mat matrix_element;
	// fliter coeffiecient for fir filter (optional)
	arma::mat filter_coeff;

	// Precalc of function
	arma::vec pulse_data;

	arma::vec construct_init_pulse(arma::vec* times,int steps){
		// Simple function that construct a pulse from a vector containing the times of the pulse.

		arma::vec init_pulse(steps);
		init_pulse.fill(0);
		if (amp_data.n_elem == 0){
			return init_pulse;
		}
		// make the init pulse.
		arma::uvec my_loc = arma::find(*times < amp_data(0,0));

		init_pulse.elem(my_loc).fill(amp_data(0,1));

		double end_voltage;


		for (int i = 0; i < amp_data.n_rows -1; ++i){

			my_loc = arma::find(*times >= amp_data(i,0) and *times < amp_data(i+1,0));
			if (my_loc.n_elem ==0)
				continue;

			end_voltage = amp_data(i,1) + (amp_data(i+1,1) - amp_data(i,1))*
					((*times)(my_loc[my_loc.n_elem-1]) - amp_data(i,0))/(amp_data(i+1,0)- amp_data(i,0));
			arma::vec start_stop_values = arma::linspace<arma::vec>(amp_data(i,1),end_voltage,my_loc.n_elem);
			init_pulse.elem(my_loc) = start_stop_values;

		}

		my_loc = arma::find(*times >= amp_data(amp_data.n_rows-1,0));
		init_pulse.elem(my_loc).fill(amp_data(amp_data.n_rows-1,1));

		return init_pulse;
	}


	arma::vec apply_IRR_filter(arma::vec b, arma::vec a, arma::vec * ideal_signal){
		// Apply filter function -- simple implementation (can prob be faster (see scipy approach), but mathematically we are correct :) )
		// b are the feedforward filter coefficients
		// a are the feedback filter coefficients

		int len_a = a.n_elem;
		int len_b = b.n_elem;

		int maxlen;
		maxlen = (len_a > len_b) ? len_b : len_a ;

		a = arma::flipud(a);
		b = arma::flipud(b);

		arma::vec signal_in(ideal_signal->n_elem + maxlen);
		signal_in.head(maxlen).fill(ideal_signal-> at(0));
		signal_in.tail(ideal_signal->n_elem) = *ideal_signal;

		arma::vec signal_out(ideal_signal->n_elem + maxlen);
		signal_out.fill(ideal_signal-> at(0));

		for (int i = maxlen; i < signal_out.n_elem; ++i){
			signal_out(i) = 1/arma::sum(a.tail(1))*(
				sum(b % signal_in.subvec(i-len_b+1, i)) -
				sum(a.head(len_a-1)% signal_out.subvec(i-len_a+1 , i-1))
				);
		}

		return signal_out.tail(ideal_signal->n_elem);
	}
public:
	void init(arma::mat amplitude_data){
		amp_data = amplitude_data;
	}
	void init(arma::mat amplitude_data, arma::mat filter_coefficients){
		amp_data = amplitude_data;
		filter_coeff = filter_coefficients;
	}
	void init(arma::mat amplitude_data, arma::cx_mat input_matrix){
		amp_data = amplitude_data;
		matrix_element = input_matrix;
	}
	void init(arma::mat amplitude_data, arma::cx_mat input_matrix, arma::mat filter_coefficients){
		amp_data = amplitude_data;
		matrix_element = input_matrix;
		filter_coeff = filter_coefficients;
	}

	arma::vec generate_pulse(double start_time, double end_time, int steps){
		double delta_t =  (end_time-start_time)/((double)steps);
		// get time steps where to calculate the pulse. 
		arma::vec times = arma::linspace<arma::vec>(start_time,end_time,steps+1) + delta_t/2;
		// remove last step
		times.shed_row(times.n_rows - 1);

		arma::vec amplitudes_pulse = construct_init_pulse(&times,steps);
		
		// Loop though all the filters given as input.
		for (int i = 0; i < filter_coeff.n_rows; ++i){
			int type = filter_coeff(i,0);
			int N = filter_coeff(i,1); //order of the filter;
			double cut_off = filter_coeff(i,2);
			double rate = 1/delta_t; // sample rate;
			// TODO: this should be done in a cleaner way...
			if (type == 0 and N < 10){
				Dsp::SimpleFilter <Dsp::Butterworth::LowPass <10> > f;
				f.setup (N, rate, cut_off);
				// Note that this library cascased the filter.
				for (int j = 0; j < f.getNumStages(); ++j)
				{
					arma::vec a;
					arma::vec b;

					a << f[j].getA0() << f[j].getA1()<< f[j].getA2();
					b << f[j].getB0() << f[j].getB1()<< f[j].getB2();

					amplitudes_pulse = apply_IRR_filter(b,a, &amplitudes_pulse);
				}
			} else if (type == 1 and N < 10){
				Dsp::SimpleFilter <Dsp::Bessel::LowPass <10> > f;
				f.setup (N, rate, cut_off);
				// Note that this library cascased the filter.
				for (int j = 0; j < f.getNumStages(); ++j)
				{
					arma::vec a;
					arma::vec b;

					a << f[j].getA0() << f[j].getA1()<< f[j].getA2();
					b << f[j].getB0() << f[j].getB1()<< f[j].getB2();

					amplitudes_pulse = apply_IRR_filter(b,a, &amplitudes_pulse);
				}
			} else { 
				std::cout << "Undefined type of filter or too high order (> 10). Skipping this filter.";
				continue;
			}
		}
		return amplitudes_pulse;

	}

	void integrate(arma::cx_cube* H0, double start_time, double end_time, int steps){
		double delta_t =  (end_time-start_time)/((double)steps);
		
		// Generate amplitudes for all steps
		// Note that the amplitude at time 0 is infact the amplitude between t[0] and t[1] (your itegral goes from a to b, so you need to take the middle element)
		arma::vec amplitudes_pulse = generate_pulse(start_time, end_time, steps);

		// Just simple numerical integration.
		for (int i = 0; i < steps; ++i){
			H0->slice(i) += matrix_element*amplitudes_pulse(i)*delta_t;
		}
	}
};
