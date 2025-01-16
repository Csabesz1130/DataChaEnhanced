import numpy as np
from scipy import signal
from scipy.integrate import simps
from src.utils.logger import app_logger

class ActionPotentialProcessor:
    def __init__(self, data, time_data, params=None):
        """
        A processor for multi-step patch-clamp signals. 
        The multi-step approach 'lays to zero' each voltage step.
        
        Args:
            data (array-like): Current data in pA.
            time_data (array-like): Time data in seconds.
            params (dict): e.g. {
                'n_cycles': 2,
                't0': 20,      # ms
                't1': 100,     # ms
                't2': 100,     # ms
                't3': 1000,    # ms
                'V0': -80,     # mV
                'V1': -100,    # mV
                'V2': 10,      # mV
                'cell_area_cm2': 1e-4
            }
        """
        self.data = np.array(data)           # pA
        self.time_data = np.array(time_data) # seconds

        self.params = params or {
            'n_cycles': 2,
            't0': 20,   # ms
            't1': 100,  # ms
            't2': 100,  # ms
            't3': 1000, # ms
            'V0': -80,  # mV
            'V1': -100, # mV
            'V2': 10,   # mV
            'cell_area_cm2': 1e-4
        }

        self.processed_data = None
        self.baseline = None
        
        # Storage for cycles and time windows
        self.cycles = []
        self.cycle_times = []
        self.cycle_indices = []
        
        app_logger.debug(f"Parameters validated: {self.params}")

    def process_signal(self):
        """
        1) baseline_correction_initial
        2) multi_segment_normalization
        3) find_cycles
        4) optional additional normalization
        5) calculate_integral
        """
        try:
            self.baseline_correction_initial()
            self.multi_segment_normalization()
            self.find_cycles()
            # self.normalize_signal()  # Optional further step
            results = self.calculate_integral()

            if not results:
                return None, None, {
                    'integral_value': 'No analysis performed',
                    'capacitance_uF_cm2': 'No analysis performed',
                    'cycle_indices': []
                }
            return self.processed_data, self.time_data, results
        
        except Exception as e:
            app_logger.error(f"Error in signal processing: {str(e)}")
            return None, None, {
                'integral_value': f"Error: {str(e)}",
                'capacitance_uF_cm2': 'Error',
                'cycle_indices': []
            }

    def baseline_correction_initial(self):
        """
        Step A: Remove an initial offset using the first t0 ms or up to 1000 samples,
        whichever is smaller. We do a median to reduce outlier influence.
        """
        sampling_rate = 1.0 / np.mean(np.diff(self.time_data))
        t0_samps = int(self.params['t0'] * sampling_rate / 1000)
        baseline_win = min(t0_samps, 1000, len(self.data))

        baseline_slice = self.data[:baseline_win]
        self.baseline = np.median(baseline_slice)
        self.processed_data = self.data - self.baseline

        app_logger.info(f"Initial baseline correction: removed {self.baseline:.2f} pA")

    def multi_segment_normalization(self):
        """
        Enhanced segment-wise normalization with adaptive window sizes and robust error handling
        """
        try:
            # Calculate sampling rate
            sampling_rate = 1.0 / np.mean(np.diff(self.time_data))
            
            # Convert time parameters to sample counts
            t0_samps = int(self.params['t0'] * sampling_rate / 1000)
            t1_samps = int(self.params['t1'] * sampling_rate / 1000)
            t2_samps = int(self.params['t2'] * sampling_rate / 1000)
            t3_samps = int(self.params.get('t3', 0) * sampling_rate / 1000)
            
            def remove_spikes(data, threshold_std=3.0, window_size=51):
                """Remove spikes while preserving baseline"""
                if len(data) < window_size:
                    window_size = max(5, len(data) // 2)
                    if window_size % 2 == 0:
                        window_size += 1
                        
                half_window = window_size // 2
                rolling_med = np.zeros_like(data)
                rolling_std = np.zeros_like(data)
                
                for i in range(len(data)):
                    start_idx = max(0, i - half_window)
                    end_idx = min(len(data), i + half_window + 1)
                    window_data = data[start_idx:end_idx]
                    rolling_med[i] = np.median(window_data)
                    rolling_std[i] = np.std(window_data)
                
                # Identify spikes
                spike_mask = np.abs(data - rolling_med) > threshold_std * rolling_std
                
                # Replace spikes with interpolated values
                cleaned_data = data.copy()
                if np.any(spike_mask):
                    non_spike_indices = np.where(~spike_mask)[0]
                    spike_indices = np.where(spike_mask)[0]
                    
                    if len(non_spike_indices) > 0:  # Only interpolate if we have non-spike points
                        cleaned_data[spike_indices] = np.interp(
                            spike_indices, 
                            non_spike_indices, 
                            data[non_spike_indices]
                        )
                
                return cleaned_data
            
            def find_stable_baseline(data, min_window=50):
                """Find stable baseline using adaptive window sizes"""
                # Adjust window size based on data length
                data_len = len(data)
                if data_len < min_window:
                    return np.median(data), np.std(data)
                    
                # Try different window sizes
                window_sizes = [
                    min(data_len, size) 
                    for size in [data_len//10, data_len//5, data_len//2]
                    if size >= min_window
                ]
                
                if not window_sizes:  # If no valid window sizes, use single window
                    window_sizes = [data_len]
                
                best_mad = float('inf')
                best_median = 0
                
                for window_size in window_sizes:
                    for start in range(0, data_len - window_size + 1, window_size//2):
                        window_data = data[start:start + window_size]
                        median = np.median(window_data)
                        mad = np.median(np.abs(window_data - median))
                        
                        if mad < best_mad:
                            best_mad = mad
                            best_median = median
                
                return best_median, best_mad
            
            # Initial baseline correction
            initial_baseline = np.median(self.data[:min(1000, t0_samps)])
            self.processed_data = self.data - initial_baseline
            app_logger.info(f"Initial baseline correction: removed {initial_baseline:.2f} pA")
            
            # Process each segment
            current_idx = 0
            for duration, name in [(t0_samps, "t0"), (t1_samps, "t1"), 
                                (t2_samps, "t2"), (t3_samps, "t3")]:
                if duration > 0:
                    end_idx = min(current_idx + duration, len(self.processed_data))
                    segment_data = self.processed_data[current_idx:end_idx]
                    
                    if len(segment_data) < 5:  # Skip very short segments
                        current_idx = end_idx
                        continue
                    
                    # Remove spikes first
                    cleaned_data = remove_spikes(segment_data)
                    
                    # Find stable baseline
                    baseline, mad = find_stable_baseline(cleaned_data)
                    
                    # Determine if segment needs correction
                    is_stable = mad < 50  # pA
                    should_correct = abs(baseline) > 20 or name in ['t1', 't2', 't3']
                    
                    if should_correct and is_stable:
                        # Apply correction
                        self.processed_data[current_idx:end_idx] -= baseline
                        app_logger.debug(
                            f"Segment {name}: Corrected baseline by {baseline:.2f} pA "
                            f"(MAD: {mad:.2f})"
                        )
                    else:
                        app_logger.debug(
                            f"Segment {name}: No correction needed "
                            f"(baseline: {baseline:.2f} pA, MAD: {mad:.2f})"
                        )
                    
                    # Re-clean after baseline correction
                    self.processed_data[current_idx:end_idx] = remove_spikes(
                        self.processed_data[current_idx:end_idx]
                    )
                    
                    current_idx = end_idx
            
            # Final alignment check
            if len(self.processed_data) > 1000:
                final_region = self.processed_data[-1000:]
                final_baseline = np.median(final_region)
                if abs(final_baseline) > 10:  # Stricter final threshold
                    self.processed_data -= final_baseline
                    app_logger.info(f"Applied final baseline correction of {final_baseline:.2f} pA")
            
            app_logger.info("Advanced multi-segment normalization completed successfully")
            
        except Exception as e:
            app_logger.error(f"Error in multi-segment normalization: {str(e)}")
            raise

    def find_cycles(self):
        """
        Step C: Identify negative peaks using find_peaks on -signal,
        ignoring big noise by using std-based prominence.
        """
        sampling_rate = 1.0 / np.mean(np.diff(self.time_data))
        t1_samples = int(self.params['t1'] * sampling_rate / 1000)

        neg_peaks, _ = signal.find_peaks(-self.processed_data,
                                         prominence=np.std(self.processed_data),
                                         distance=t1_samples)

        for i, peak in enumerate(neg_peaks[: self.params['n_cycles']]):
            start = max(0, peak - t1_samples//2)
            end = min(len(self.processed_data), peak + t1_samples*2)
            
            cycle = self.processed_data[start:end]
            cycle_time = self.time_data[start:end] - self.time_data[start]
            
            self.cycles.append(cycle)
            self.cycle_times.append(cycle_time)
            self.cycle_indices.append((start, end))
            app_logger.debug(f"Cycle {i+1} found at index {peak}")

    def normalize_signal(self):
        """
        Step D (optional): Additional baseline or scale 
        after multi_segment_normalization if needed.
        """
        if not self.cycles:
            app_logger.info("No cycles for final normalization, skipping.")
            return

        # Example: subtract mean of each cycle's first 50 points
        offset_vals = []
        for cyc in self.cycles:
            if len(cyc) > 50:
                offset_vals.append(np.mean(cyc[:50]))
        if offset_vals:
            final_offset = np.mean(offset_vals)
            self.processed_data -= final_offset
            for i, (start, end) in enumerate(self.cycle_indices):
                self.cycles[i] = self.processed_data[start:end]
            app_logger.info(f"Final offset subtracted: {final_offset:.2f} pA")

    def calculate_integral(self):
        """
        Step E: Integrate the first cycle above threshold => total charge => 
        compute capacitance in µF/cm² using (V1 - V0).
        """
        try:
            if not self.cycles:
                return {
                    'integral_value': 'No cycles found',
                    'capacitance_uF_cm2': '0.0000 µF/cm²',
                    'cycle_indices': []
                }
            cycle = self.cycles[0]
            cycle_time = self.cycle_times[0]

            current_in_A = cycle * 1e-12
            time_in_s = cycle_time
            voltage_diff_in_V = (self.params['V1'] - self.params['V0']) * 1e-3

            peak_curr = np.max(np.abs(current_in_A))
            thr = 0.1 * peak_curr
            mask = np.abs(current_in_A) > thr

            if not np.any(mask):
                return {
                    'integral_value': '0.0000 C',
                    'capacitance_uF_cm2': '0.0000 µF/cm²',
                    'cycle_indices': self.cycle_indices
                }

            charge_C = np.trapz(current_in_A[mask], time_in_s[mask])
            total_cap_F = abs(charge_C / voltage_diff_in_V)
            total_cap_uF = total_cap_F * 1e6

            area = self.params.get('cell_area_cm2', 1e-4)
            cap_uF_cm2 = total_cap_uF / area

            results = {
                'integral_value': f"{abs(charge_C):.6e} C",
                'capacitance_uF_cm2': f"{cap_uF_cm2:.4f} µF/cm²",
                'cycle_indices': self.cycle_indices,
                'raw_values': {
                    'charge_C': charge_C,
                    'capacitance_F': total_cap_F,
                    'area_cm2': area
                }
            }

            app_logger.info(f"Integrated charge: {charge_C:.2e} C, "
                            f"Capacitance: {cap_uF_cm2:.4f} µF/cm²")
            return results

        except Exception as e:
            app_logger.error(f"Error calculating integral: {str(e)}")
            return {
                'integral_value': 'Error in calculation',
                'capacitance_uF_cm2': 'Error',
                'cycle_indices': self.cycle_indices
            }
