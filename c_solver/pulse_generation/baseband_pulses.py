from dataclasses import dataclass
from c_solver.pulse_generation.utility import get_effective_point_number

import numpy as np
import copy

@dataclass
class function_data:
    start : np.double
    stop : np.double
    function : any

@dataclass
class base_pulse_element:
    start : float
    stop : float
    v_start : float
    v_stop : float

    index_start : int = 0
    index_stop : int = 0


class pulse_data_blocks():
    def __init__(self):
        self.localdata = [base_pulse_element(0,1e-9,0,0)]
        self.re_render = True
        self.voltage_data = np.array([0])
        self._total_time = 0
    
    def add_pulse(self, pulse):
        self.localdata.append(pulse)
        self.re_render = True
        if self._total_time < pulse.stop:
            self._total_time = pulse.stop

    def __add__(self, other):
        pulse_data = copy.copy(self)
        if isinstance(other, pulse_data_blocks):
            pulse_data.localdata += copy.copy(other.localdata)
        else:
            raise ValueError("invalid data type for count up provided.")

        pulse_data.re_render = True

        return pulse_data

    def __local_render(self):
        time_steps = []

        t_step = 1e-9
        for i in self.localdata:
            time_steps.append(i.start)
            time_steps.append(i.start+t_step)
            if i.stop == -1:
                time_steps.append(self.total_time-t_step)
                time_steps.append(self.total_time)
            else:
                time_steps.append(i.stop-t_step)
                time_steps.append(i.stop)

        time_steps_np, index_inverse = np.unique(np.array(time_steps), return_inverse=True)

        for i in range(int(len(index_inverse)/4)):
            self.localdata[i].index_start = index_inverse[i*4+1]
            self.localdata[i].index_stop = index_inverse[i*4+2]


        voltage_data = np.zeros([len(time_steps_np)])


        for i in self.localdata:
            delta_v = i.v_stop-i.v_start
            min_time = time_steps_np[i.index_start]
            max_time = time_steps_np[i.index_stop]
            rescaler = delta_v/(max_time-min_time)

            for j in range(i.index_start, i.index_stop+1):
                voltage_data[j] += i.v_start + (time_steps_np[j] - min_time)*rescaler


        # cleaning up the array (remove 1e-10 spacings between data points):
        new_data_time = []
        new_data_voltage = []


        new_data_time.append(time_steps_np[0])
        new_data_voltage.append(voltage_data[0])

        i = 1
        while( i < len(time_steps_np)-1):
            if time_steps_np[i+1] - time_steps_np[i] < t_step*2 and time_steps_np[i] - time_steps_np[i-1] < t_step*2:
                i+=1

            new_data_time.append(time_steps_np[i])
            new_data_voltage.append(voltage_data[i])
            i+=1
        if i < len(time_steps_np):
            new_data_time.append(time_steps_np[i]) 
            new_data_voltage.append(voltage_data[i])


        return new_data_time, new_data_voltage

    def render(self, endtime, sample_rate):
        '''
        make a full rendering of the waveform at a predetermined sample rate.
        '''

        # express in Gs/s
        sample_rate = sample_rate*1e-9
        sample_time_step = 1/sample_rate
        
        self._total_time = endtime
        t_tot = endtime

        # get number of points that need to be rendered
        t_tot_pt = get_effective_point_number(t_tot, sample_time_step) + 1

        my_sequence = np.zeros([int(t_tot_pt)])
        
        # start rendering pulse data
        time_data, voltage_data = self.pulse_data
        baseband_pulse = np.empty([len(time_data), 2])

        baseband_pulse[:,0] = time_data
        baseband_pulse[:,1] = voltage_data


        for i in range(0,len(baseband_pulse)-1):
            t0_pt = get_effective_point_number(baseband_pulse[i,0], sample_time_step)
            t1_pt = get_effective_point_number(baseband_pulse[i+1,0], sample_time_step) + 1
            t0 = t0_pt*sample_time_step
            t1 = t1_pt*sample_time_step
            if t0 > t_tot:
                continue
            elif t1 > t_tot + sample_time_step:
                if baseband_pulse[i,1] == baseband_pulse[i+1,1]:
                    my_sequence[t0_pt: t_tot_pt] = baseband_pulse[i,1]
                else:
                    val = py_calc_value_point_in_between(baseband_pulse[i,:], baseband_pulse[i+1,:], t_tot)
                    my_sequence[t0_pt: t_tot_pt] = np.linspace(
                        baseband_pulse[i,1], 
                        val, t_tot_pt-t0_pt)
            else:
                if baseband_pulse[i,1] == baseband_pulse[i+1,1]:
                    my_sequence[t0_pt: t1_pt] = baseband_pulse[i,1]
                else:
                    my_sequence[t0_pt: t1_pt] = np.linspace(baseband_pulse[i,1], baseband_pulse[i+1,1], t1_pt-t0_pt)
        # top off the sequence -- default behavior, extend the last value
        if len(baseband_pulse) > 1:
            pt = get_effective_point_number(baseband_pulse[i+1,0], sample_time_step)
            my_sequence[pt:] = baseband_pulse[i+1,1]

        return my_sequence

    @property
    def pulse_data(self):
        if self.re_render == True:
            self.time_data, self.voltage_data = self.__local_render()
        return (self.time_data, self.voltage_data)

    @property
    def total_time(self):
        return self._total_time