from DM_solver.pulse_generation.envelopes import envelope_generator
from dataclasses import dataclass, field

import numpy as np

@dataclass
class baseband_pulse:
    start : float
    stop : float
    v_start : float
    v_stop : float

    def render(self, array, delta_t):
        start_idx = round(self.start/delta_t)
        stop_idx  = round(self.stop/delta_t)
        
        if self.stop < 0:
            stop_idx = array.size
        
        if start_idx > array.size:
            return array

        npt = stop_idx - start_idx
        delta_V = self.v_stop - self.v_start
        
        if npt > 1e8:
            raise ValueError(f"Pulse length very long {stop_idx - start_idx}, check if timings/sample rate are realistic")
        
        temp = np.linspace(self.v_start + delta_V/npt, self.v_stop - delta_V/npt, npt)

        if stop_idx > array.size:
            stop_idx = array.size
        
        array[start_idx:stop_idx] += temp[:stop_idx-start_idx]
        
        return array

@dataclass
class function_pulse:
    start : float
    stop : float
    function : any

    def render(self, array, delta_t):
        start_idx = round(self.start/delta_t)
        stop_idx  = round(self.stop/delta_t)
        
        if self.stop < 0:
            stop_idx = array.size

        if start_idx > array.size:
            return array

        npt = stop_idx - start_idx        
        if npt > 1e8:
            raise ValueError(f"Pulse length very long {stop_idx - start_idx}, check if timings/sample rate are realistic")
                
        if stop_idx > array.size:
            stop_idx = array.size

        array[start_idx:stop_idx] += self.function(np.linspace(0, 1, npt))[:stop_idx-start_idx]
        
        return array

@dataclass
class MW_pulse:
    start : float
    stop : float
    amplitude : float
    frequency : float
    start_phase : float = 0
    envelope : envelope_generator = envelope_generator()
    filters : list = field(default_factory=lambda: [])
    is_RWA : bool = False
    
    def render(self, array, delta_t):
        start_idx = round(self.start/delta_t)
        stop_idx  = round(self.stop/delta_t)
        
        if self.stop < 0:
            stop_idx = array.size
        
        if start_idx > array.size:
            return array

        npt = stop_idx - start_idx

        if stop_idx > array.size:
            stop_idx = array.size

        if npt > 1e8:
            raise ValueError(f"Pulse length very long {stop_idx - start_idx}, check if timings/sample rate are realistic")

        
        amp_envelope_full= np.zeros(array.shape)
        amp_envelope_full[start_idx:stop_idx] = self.envelope.get_AM_envelope((npt))[:stop_idx-start_idx]

        phase_envelope_full= np.zeros(array.shape)
        phase_envelope_full[start_idx:stop_idx] = self.envelope.get_PM_envelope((npt))[:stop_idx-start_idx]

        time_full= np.zeros(array.shape)
        time_full[start_idx:stop_idx] = np.linspace(start_idx*delta_t, stop_idx*delta_t, stop_idx-start_idx+1)[:-1]

        for f in self.filters:
            amp_envelope_full = f(amp_envelope_full, 1/delta_t)
        
        if self.is_RWA == True:
            temp = (self.amplitude*amp_envelope_full*
                        np.exp(-1j*(2.*np.pi*time_full*self.frequency + self.start_phase + phase_envelope_full)))/2
        else:
            temp = (self.amplitude*amp_envelope_full*
                        np.cos(2.*np.pi*self.frequency*time_full + self.start_phase + phase_envelope_full))
        
        return array + temp