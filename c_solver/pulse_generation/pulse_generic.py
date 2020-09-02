import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy import signal

from c_solver.pulse_generation.baseband_pulses import pulse_data_blocks, base_pulse_element, function_data, filter_data
from c_solver.pulse_generation.ac_pulses import MW_data_single, envelope_generator
from c_solver.pulse_generation.utility import get_effective_point_number
from c_solver.pulse_generation.keysight_filter.keysight_filter import keysight_anti_ringing_filtered_output

class pulse():
    """docstring for baseband_pulse"""
    def __init__(self):
        self.block_data = pulse_data_blocks()
        self.function_data = list()
        self.MW_data = list()
        self.filter_data = list()
    
    def add_filter(self, filter_type, f_cut=None):
        '''
        add a filter to the simulation.
        
        filter_type (str) : type of used filter
        f_cut : cutoff frequency. Used for 'firwin' and 'butter' types. Default = None

        Expected strings for the filters are:
            - 'firwin' : FIR filter design using the window method
            - 'butter' : Butterworth digital and analog filter design
            - 'keysight_ar' : Keysight anti ringing filter
                returns adds filter to input signal
        '''
        if filter_type == 'keysight_ar':
            pass
        else:
            if f_cut is None:
                raise Exception('Input a cutoff frequency for this filter type')
        
        self.filter_data.append(filter_data(f_cut, filter_type))
    
    
    def add_function(self, start, stop, function):
        '''
        add a pulse shape that is defined in a function.
        
        start (double) : start time at which the function has to be inserted
        stop (double) : stop time that the function has the be inserted.
        function (def) : python function

        Expected format of the function is:
            def my_function(input_array):
                # args:
                #   input_array (np.ndarray) : array that ranges from 0 to 1 (where 0 is the start of the pulse and 1 is the end)
                # returns:
                #   pulse_amplitudes (np.ndarray) : returns pulse amplitudes

                returns pulse_amplitudes
        '''
        
        
        self.function_data.append(function_data(start, stop, function))

    def add_block(self,start,stop, amplitude):
        '''
        add a block pulse on top of the existing pulse.
        '''

        pulse = base_pulse_element(start,stop, amplitude, amplitude)
        self.block_data.add_pulse(pulse)
    
    
    def add_offset(self, amplitude):
        '''
        add a block pulse on top of the existing pulse.
        '''

        pulse = base_pulse_element(0,-1, amplitude, amplitude)
        self.block_data.add_pulse(pulse)
    
    
    def add_ramp(self, start, stop, amplitude, keep_amplitude=False):
        '''
        Makes a linear ramp
        Args:
            start (double) : starting time of the ramp
            stop (double) : stop time of the ramp
            amplitude : total hight of the ramp, starting from the base point
            keep_amplitude : when pulse is done, keep reached amplitude for time infinity
        '''
        
        if keep_amplitude == True:
            pulse = base_pulse_element(start,stop, 0, amplitude)
            self.block_data.add_pulse(pulse)
            pulse = base_pulse_element(stop,-1,amplitude, amplitude)
            self.block_data.add_pulse(pulse)
        else:
            pulse = base_pulse_element(start,stop, 0, amplitude)
            self.block_data.add_pulse(pulse)

        

    def add_ramp_tan(self, start, stop, amplitude, keep_amplitude=False):
        '''
        Makes a linear ramp
        Args:
            start (double) : starting time of the ramp
            stop (double) : stop time of the ramp
            amplitude : total hight of the ramp, starting from the base point
            keep_amplitude : when pulse is done, keep reached amplitude for time infinity
        '''
        
        if keep_amplitude == True:
            pulse = base_pulse_element(start,-1, 0, amplitude)
        else:
            pulse = base_pulse_element(start,stop, 0, amplitude)

        self.block_data.add_pulse(pulse)

    def add_ramp_ss(self, start, stop, start_amplitude, stop_amplitude, keep_amplitude=False):
        '''
        Makes a linear ramp (with start and stop amplitude)
        Args:
            start (double) : starting time of the ramp
            stop (double) : stop time of the ramp
            amplitude : total hight of the ramp, starting from the base point
            keep_amplitude : when pulse is done, keep reached amplitude for time infinity
        '''
        if keep_amplitude == True:
            pulse = base_pulse_element(start,-1, start_amplitude, stop_amplitude)
        else:
            pulse = base_pulse_element(start,stop, start_amplitude, stop_amplitude)

        self.block_data.add_pulse(pulse)

    def add_MW_pulse(self, t0, t1, amp, freq, phase = 0., AM = None, PM = None, is_RWA = False):
        '''
        Make a sine pulse (generic constructor)

        Args:
            t0(float) : start time in ns
            t1(float) : stop tiume in ns
            amp (float) : amplitude of the pulse.
            freq(float) : frequency
            phase (float) : phase of the microwave.
            AM ('str/tuple/function') : function describing an amplitude modulation (see examples in c_solver.pulse_generation.ac_pulses)
            PM ('str/tuple/function') : function describing an phase modulation (see examples in c_solver.pulse_generation.ac_pulses)
        '''
        MW_data_temp = MW_data_single(t0, t1, amp, freq, phase, envelope_generator(AM, PM),is_RWA)
        self.MW_data.append(MW_data_temp)    

    def __add__(self, other):
        new_pulse = pulse()
        new_pulse.block_data += other.block_data
        new_pulse.block_data += self.block_data

        new_pulse.function_data += copy.deepcopy(other.function_data)
        new_pulse.function_data += copy.deepcopy(self.function_data)

        new_pulse.MW_data += copy.deepcopy(other.MW_data)
        new_pulse.MW_data += copy.deepcopy(self.MW_data)
        
        new_pulse.filter_data += copy.deepcopy(other.filter_data)
        new_pulse.filter_data += copy.deepcopy(self.filter_data)

        return new_pulse

    def get_pulse_raw(self, endtime, sample_rate):
        '''
        get data formated in numpy style of the pulse.

        Args:
            start (double) : start time of the pulse
            stop (double) : stop time of the pulse
            sample_rate (double) : sample rate in Hz

        Returns:
            sequence (np.ndarray) : 1D arrray with the amplitude values of the pulse
        '''
        sequence = self.block_data.render(endtime, sample_rate)
#        print("length  pulse at line 167")
#        print(len(sequence))
#        print("block data render")
#        print(sequence[0:2])
        time_step = 1./sample_rate

        for f_data in self.function_data:

            start_idx = int(np.ceil(f_data.start/time_step*1e-9))
            stop_idx = int(f_data.stop/time_step*1e-9)

            #effective_start_time = start_idx*time_step
            #effective_stop_time = stop_idx*time_step

            #norm_start  = (f_data.start - effective_start_time)/(f_data.stop-f_data.start)
            #norm_stop = 1 - (f_data.stop - effective_stop_time)/(f_data.stop-f_data.start)
            normalized_time_seq = np.linspace(0, 1, stop_idx-start_idx)
            #normalized_time_seq = np.linspace(0, 1, norm_stop-norm_start)
            sequence[start_idx:stop_idx] += f_data.function(normalized_time_seq)
            
#            print("length of function pulse")
#            print(len(sequence))
        
         #add fitler functions to the seqeunce
        for f_data in self.filter_data:
            
            # if f_data.f_FIR:
            #     b = signal.firwin(21, f_data.f_cut_freq, pass_zero = 'lowpass',fs=sample_rate)
            #     a = [1.0]
            # else:
            #     b, a = signal.butter(3,f_data.f_cut_freq, btype = 'lowpass',fs=sample_rate)
            # initial_voltage = sequence[0]
            # sequence += np.full(sequence.shape,-1.0*initial_voltage)
            # sequence = signal.lfilter(b,a,sequence)
            # sequence += np.full(sequence.shape,1.0*initial_voltage)
            
            if f_data.filter_type == 'keysight_ar':
                sequence = keysight_anti_ringing_filtered_output(sequence, sample_rate)
            else:
                if f_data.filter_type == 'firwin':
                    b = signal.firwin(21, f_data.f_cut_freq, pass_zero = 'lowpass',fs=sample_rate)
                    a = [1.0]
                elif f_data.filter_type == 'butter':
                    b, a = signal.butter(3,f_data.f_cut_freq, btype = 'lowpass',fs=sample_rate)
                else:
                    raise Exception('Fitler type not implemented yet')
                initial_voltage = sequence[0]
                sequence += np.full(sequence.shape,-1.0*initial_voltage)
                sequence = signal.lfilter(b,a,sequence)
                sequence += np.full(sequence.shape,1.0*initial_voltage)
#        
            
#        print("number of microwave data")
#        print(len(self.MW_data))
#        print("before MW pulse")
#        print(sequence[0:2])
#        print("length  pulse at line 208")
#        print(len(sequence))
        # render MW pulses.
        for MW_data_single_object in self.MW_data:
            # start stop time of MW pulse

            start_pulse = MW_data_single_object.start*1e-9
            stop_pulse = MW_data_single_object.stop *1e-9
            
            # max amp, freq and phase.
            amp  =  MW_data_single_object.amplitude
            freq =  MW_data_single_object.frequency
            phase = MW_data_single_object.start_phase
#            print("amp, freq, phase of MW shape")
#            print([amp,freq,phase])
            
            
            # evelope data of the pulse
            if MW_data_single_object.envelope is None:
                MW_data_single_object.envelope = envelope_generator()

            amp_envelope = MW_data_single_object.envelope.get_AM_envelope((stop_pulse - start_pulse), sample_rate)
            phase_envelope = MW_data_single_object.envelope.get_PM_envelope((stop_pulse - start_pulse), sample_rate)
#            print("time conflict MW shape")
#            print([(stop_pulse - start_pulse)*sample_rate, int((stop_pulse - start_pulse)* sample_rate)])
#            amp_envelope = np.zeros([int(sample_rate*1e-9)],dtype = np.double)+1.
#            phase_envelope = np.zeros([int(sample_rate*1e-9)],dtype = np.double)
            
            for f_data in self.filter_data:
            
                if f_data.f_FIR:
                    b = signal.firwin(21, f_data.f_cut_freq, pass_zero = 'lowpass',fs=sample_rate)
                    a = [1.0]
                else:
                    b, a = signal.butter(2, f_data.f_cut_freq, btype = 'lowpass',fs=sample_rate)
                amp_envelope = signal.lfilter(b,a,amp_envelope)
                phase_envelope = signal.lfilter(b,a,phase_envelope)
            

            #self.baseband_pulse_data[-1,0] convert to point numbers
            n_pt = len(amp_envelope)
            phase_envelope = phase_envelope[0:n_pt]
            
            
            start_pt = get_effective_point_number(start_pulse, time_step)
            stop_pt = start_pt + n_pt
            
            if MW_data_single_object.is_RWA == True:
#                print("RWA modus enabled")
                sequence[start_pt:(stop_pt)] += 1./2.*amp*amp_envelope*np.exp(-1j*(np.linspace(start_pulse, stop_pulse, n_pt,dtype=complex)*np.complex(freq)*2.*np.pi  + phase + phase_envelope) )
            else:
            # add up the sin pulse.
#                print("RWA modus disabled")
                sequence[start_pt:(stop_pt)] += amp*amp_envelope*np.cos(np.linspace(start_pulse, stop_pulse, n_pt,dtype=complex)*freq*2.*np.pi + phase + phase_envelope )
                
        return sequence

    def get_pulse(self,endtime, sample_rate):
        '''
        get data formated in numpy style of the pulse + times

        Args:
            start (double) : start time of the pulse
            stop (double) : stop time of the pulse
            sample_rate (double) : sample rate in Hz

        Returns:
            sequence (np.ndarray) : 1D arrray with the amplitude values of the pulse
            times (np.ndarray) : time value corresponding to the sequence value.
        '''

        sequence = self.get_pulse_raw(endtime, sample_rate)
        time_step = 1/sample_rate
        times = np.linspace(0, len(sequence)*time_step*1e9, len(sequence))

        return times, sequence

    def plot_pulse(pulse , endtime, smpl_rate = 1e11, f_function = None, scaling = 1.):
        if f_function == None:
            def function(arg):
                return arg
        t, v  =pulse.get_pulse(endtime, smpl_rate)
        data = scaling*f_function(np.real(v))
        plt.plot(t, data ,'b')
#        plt.plot(t,np.imag(v),'r')
        plt.xlabel('time (ns)')
        plt.ylabel('amplitude (a.u.)')
        plt.show()

if __name__ == '__main__':
    
    p = pulse()
    delay=1
    def enevlope_fun(time):
        return 10.*(np.arctan(3)+np.arctan(6*(time-delay/2)/delay))/(2*np.arctan(3))
    def enevlope_fun_ss(time):
        return 10.*(np.arctan(3)+np.arctan(6*(1-time-delay/2)/delay))/(2*np.arctan(3))
            #self.pulseDummy.add_ramp(t_start,(t_start+t_ramp),self.exchange_dc)
    p.add_block(0,20,10)
    p.add_function(20,40,enevlope_fun_ss)
    p.add_block(40,60,10)
    p.add_ramp(70,72,10,keep_amplitude=True)
    p.add_filter(400*1e6,False)
    
    #p.add_block(10,50,10)
    #p.add_ramp(10,15,10)
    #p.add_MW_pulse(200,300,10,1e9)
    t, v  =p.get_pulse(150, 1e11)
    plt.plot(t,v)
    print(v[0])
    plt.xlabel('time (ns)')
    plt.ylabel('amplitude (a.u.)')
    plt.show()