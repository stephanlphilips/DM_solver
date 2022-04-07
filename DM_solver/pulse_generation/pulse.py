from DM_solver.pulse_generation.pulse_types import baseband_pulse, function_pulse, MW_pulse
from DM_solver.pulse_generation.envelopes import envelope_generator

import matplotlib.pyplot as plt
import numpy as np

class pulse():
    def __init__(self):
        self.pulse_data = pulse_data()
        self.__phase_shift = 0
    
    def add_constant(self, amp):
        '''
        Args:
            amp (double) : add constant amplitude amp for the whole pulse
        '''
        self.add_block(0, -1, amp)

    def add_block(self,t_start, t_stop, amp):
        '''
        Args:
            t_start (double) : start time [s]
            t_stop (double) : stop time [s]
            start_amplitude (float)
        '''
        pulse = baseband_pulse(t_start, t_stop, amp, amp)
        self.pulse_data.add_pulse(pulse)

    def add_ramp(self, t_start, t_stop, start_amp, stop_amp):
        '''
        Args:
            t_start (double) : start time [s]
            t_stop (double) : stop time [s]
            start_amp (float) : start amplitude
            stop_amp (float) : stop amplitude
        '''
        pulse = baseband_pulse(t_start,t_stop, start_amp, stop_amp)
        self.pulse_data.add_pulse(pulse)

    def add_function(self, t_start, t_stop, function):
        '''  
        Args:      
            t_start (double) : start time at which the function has to be inserted
            t_stop (double) : stop time that the function has the be inserted.
            function (def) : python function with an agrument that takes an array with values between 0 and 1.
        '''
        self.pulse_data.add_pulse(function_pulse(t_start, t_stop, function))

    def add_MW_pulse(self, t_start, t_stop, amp, freq, phase = 0., AM = None, PM = None, is_RWA = False):
        '''
        Args:
            t_start(float) : start time in ns
            t_stop(float) : stop tiume in ns
            amp (float) : amplitude of the pulse.
            freq(float) : frequency
            phase (float) : phase of the microwave.
            AM ('str/tuple/function') : function describing an amplitude modulation (see examples in DM_solver.pulse_generation.envelopes)
            PM ('str/tuple/function') : function describing an phase modulation (see examples in DM_solver.pulse_generation.envelopes)
            is_RWA (bool) : render function as e^(iwt) instead of a normal sinus
        '''
        MW_data_temp = MW_pulse(t_start, t_stop, amp, freq, phase + self.__phase_shift, envelope_generator(AM, PM), is_RWA)
        self.pulse_data.add_pulse(MW_data_temp)    

    def add_phase_shift(self, phase):
        '''
        Args:
            phase (float) : phase shift to be added to the MWves [rad]
        '''
        self.__phase_shift += phase

    def add_filter(self, filter_function):
        '''
        Args:
            filter_function (function) : function that filters the signal
        '''  
        self.pulse_data.add_filter(filter_function)

    def render(self, t_tot, sample_rate):
        '''
        Args:
            t_tot (float) : total time to be rendered [s]
            sample_rate (float) : sample rate [Hz]
        '''
        return self.pulse_data.render(t_tot, sample_rate)

    def plot_pulse(self , t_tot=None, sample_rate = 1e11, f_function = None, scaling = 1.):
        if t_tot == None:
            t_tot = self.pulse_data.total_time
        if f_function == None:
            f_function = lambda x : x
        
        amp = self.render(t_tot, sample_rate)
        amp = scaling*f_function(np.real(amp))

        plt.plot(np.linspace(0, amp.size/sample_rate, amp.size+1)[:-1 ], amp,'b')
        plt.xlabel('time (s)')
        plt.ylabel('amplitude (a.u.)')
        plt.show()
        
class pulse_data():
    def __init__(self):
        self.pulse_objects = []
        self.filters       = []

    def add_pulse(self, pulse):
        self.pulse_objects.append(pulse)

    def add_filter(self, filter_function):
        self.filters.append(filter_function)

    def __add__(self, other):
        self.pulse_objects += other.pulse_objects

    def render(self, t_tot, sample_rate):
        npt = round(t_tot*sample_rate)
        data = np.zeros(npt)

        for pulse in self.pulse_objects:
            if isinstance(pulse, (baseband_pulse, function_pulse)):
                data = pulse.render(data, 1/sample_rate)

        for f in self.filters:
            data = f(data, sample_rate)

        for pulse in self.pulse_objects:
            if isinstance(pulse, MW_pulse):
                pulse.filters = self.filters
                data = pulse.render(data, 1/sample_rate)
        
        return data
    
    @property
    def total_time(self):
        t_pulse = 0
        for pulse in self.pulse_objects:
            if t_pulse< pulse.stop:
                t_pulse = pulse.stop
        return t_pulse

if __name__ == '__main__':
    p = pulse()

    p.add_block(3e-9, 8e-9, 1)
    p.add_ramp( 8e-9, 15e-9, 1, 0)
    p.add_MW_pulse(20e-9, 30e-9, 1, 1e9, AM='blackman')

    SineShape = lambda x:(1-np.cos(x*np.pi*2))/2
    p.add_function(31e-9, 39e-9, SineShape)

    from DM_solver.pulse_generation.filters import keysight_anti_ringing_filtered_output
    p.add_filter(keysight_anti_ringing_filtered_output)


    p.plot_pulse(50e-9)

