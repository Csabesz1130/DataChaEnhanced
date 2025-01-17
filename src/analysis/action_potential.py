import numpy as np
from scipy import signal
from src.utils.logger import app_logger

class ActionPotentialProcessor:
    def __init__(self, data, time_data, params=None):
        """
        Multi-step patch clamp data processor.
        
        data in pA, time_data in seconds.
        Typical params keys:
          'n_cycles', 't0', 't1', 't2', 't3',
          'V0', 'V1', 'V2',
          'cell_area_cm2'.
        """
        self.data = np.array(data)
        self.time_data = np.array(time_data)
        self.params = params or {
            'n_cycles': 2,
            't0': 20,
            't1': 100, 
            't2': 100,
            't3': 1000,
            'V0': -80,
            'V1': -100,
            'V2': 10,
            'cell_area_cm2': 1e-4
        }
        
        self.processed_data = None
        self.baseline = None
        self.cycles = []
        self.cycle_times = []
        self.cycle_indices = []
        self.orange_curve = None
        self.orange_curve_times = None
        
        app_logger.debug(f"Parameters validated: {self.params}")

    def process_signal(self):
        """
        The main pipeline:
        1. baseline_correction_initial
        2. advanced_baseline_normalization
        3. apply 50-point averaging
        4. generate_orange_curve
        5. find_cycles
        6. calculate_integral
        """
        try:
            self.baseline_correction_initial()
            self.advanced_baseline_normalization()
            
            # Apply 50-point averaging to processed data
            self.processed_data = self.apply_50point_average(self.processed_data)
            
            # Generate orange curve (decimated view)
            self.generate_orange_curve()
            
            self.find_cycles()
            results = self.calculate_integral()
            
            if not results:
                return None, None, None, {
                    'integral_value': 'No analysis performed',
                    'capacitance_uF_cm2': 'No analysis performed',
                    'cycle_indices': []
                }
            return self.processed_data, self.orange_curve, self.orange_curve_times, results
            
        except Exception as e:
            app_logger.error(f"Error in process_signal: {str(e)}")
            return None, None, None, {
                'integral_value': f"Error: {str(e)}",
                'capacitance_uF_cm2': 'Error',
                'cycle_indices': []
            }

    def apply_50point_average(self, data):
        """Apply stronger 50-point averaging similar to orange curve."""
        try:
            window_size = 50
            averaged_data = np.zeros_like(data)
            
            # Process data in non-overlapping windows
            for i in range(0, len(data) - window_size + 1, window_size):
                # Calculate average for current window
                window_avg = np.mean(data[i:i + window_size])
                # Apply this average to all points in the window
                averaged_data[i:i + window_size] = window_avg
            
            # Handle remaining points at the end
            if len(data) % window_size != 0:
                remaining_start = len(data) - (len(data) % window_size)
                remaining_avg = np.mean(data[remaining_start:])
                averaged_data[remaining_start:] = remaining_avg
            
            return averaged_data
            
        except Exception as e:
            app_logger.error(f"Error applying 50-point average: {str(e)}")
            raise

    def baseline_correction_initial(self):
        """Step 1: Initial baseline correction excluding outlier points."""
        sampling_rate = 1.0 / np.mean(np.diff(self.time_data))
        t0_samps = int(self.params['t0'] * sampling_rate / 1000)
        baseline_window = min(t0_samps, 1000, len(self.data))
        
        if baseline_window < 1:
            self.baseline = np.median(self.data)
        else:
            # Get initial window of data
            initial_data = self.data[:baseline_window]
            
            # Calculate median and std excluding outliers
            med = np.median(initial_data)
            std = np.std(initial_data)
            
            # Create mask for non-outlier points (within 3 std)
            mask = np.abs(initial_data - med) < 3 * std
            
            if np.any(mask):
                self.baseline = np.median(initial_data[mask])
            else:
                self.baseline = med
        
        # Subtract baseline
        self.processed_data = self.data - self.baseline
        app_logger.info(f"Initial baseline correction: subtracted {self.baseline:.2f} pA")

    def advanced_baseline_normalization(self):
        """
        Advanced normalization to align segments based on slope changes.
        Uses old-style for loops for better control and visualization.
        """
        try:
            sampling_rate = 1.0 / np.mean(np.diff(self.time_data))
            threshold_slope = 3 * np.std(np.diff(self.processed_data))
            
            # Start segment analysis
            segment_start = 0
            segments = []
            
            # Find segments using for loop
            for i in range(1, len(self.processed_data)):
                slope = abs(self.processed_data[i] - self.processed_data[i-1])
                if slope > threshold_slope:
                    # End current segment
                    if i - segment_start > 50:  # Minimum segment length
                        segments.append((segment_start, i))
                    segment_start = i + 1
            
            # Add final segment if exists
            if segment_start < len(self.processed_data) - 50:
                segments.append((segment_start, len(self.processed_data)))
            
            # Process each segment
            for start, end in segments:
                # Calculate baseline from last 50 points of segment
                baseline_points = self.processed_data[max(start, end-50):end]
                if len(baseline_points) > 0:
                    baseline = np.mean(baseline_points)
                    # Subtract baseline from segment
                    self.processed_data[start:end] -= baseline
            
            app_logger.info(f"Advanced normalization completed with {len(segments)} segments")
            
        except Exception as e:
            app_logger.error(f"Error in advanced_baseline_normalization: {str(e)}")
            raise

    def generate_orange_curve(self):
        """
        Generate decimated orange curve by taking one average point per 50 points.
        """
        try:
            orange_points = []
            orange_times = []
            window_size = 50
            
            for i in range(0, len(self.processed_data), window_size):
                # Get current window
                window = self.processed_data[i:min(i + window_size, len(self.processed_data))]
                time_window = self.time_data[i:min(i + window_size, len(self.time_data))]
                
                if len(window) > 0:
                    # Calculate average for window
                    avg_point = np.mean(window)
                    avg_time = np.mean(time_window)
                    
                    orange_points.append(avg_point)
                    orange_times.append(avg_time)
            
            self.orange_curve = np.array(orange_points)
            self.orange_curve_times = np.array(orange_times)
            
            app_logger.info(f"Orange curve generated with {len(orange_points)} points")
            
        except Exception as e:
            app_logger.error(f"Error generating orange curve: {str(e)}")
            raise

    def find_cycles(self):
        """Identify negative peaks using find_peaks on -self.processed_data."""
        sampling_rate = 1.0 / np.mean(np.diff(self.time_data))
        t1_samps = int(self.params['t1'] * sampling_rate / 1000)
        
        neg_peaks, _ = signal.find_peaks(-self.processed_data,
                                       prominence=np.std(self.processed_data),
                                       distance=t1_samps)
        
        for i, peak in enumerate(neg_peaks[:self.params['n_cycles']]):
            start = max(0, peak - t1_samps//2)
            end = min(len(self.processed_data), peak + t1_samps*2)
            cycle = self.processed_data[start:end]
            cycle_time = self.time_data[start:end] - self.time_data[start]
            
            self.cycles.append(cycle)
            self.cycle_times.append(cycle_time)
            self.cycle_indices.append((start, end))
            app_logger.debug(f"Cycle {i+1} found at index {peak}")

    def calculate_integral(self):
        """Calculate integral and capacitance from the first cycle."""
        try:
            if not self.cycles:
                return {
                    'integral_value': 'No cycles found',
                    'capacitance_uF_cm2': '0.0000 µF/cm²',
                    'cycle_indices': []
                }
            
            cycle = self.cycles[0]
            cycle_time = self.cycle_times[0]
            
            # Convert units
            current_in_A = cycle * 1e-12  # pA to A
            time_in_s = cycle_time  # already in seconds
            voltage_diff_in_V = (self.params['V1'] - self.params['V0']) * 1e-3  # mV to V
            
            # Find threshold for integration
            peak_curr = np.max(np.abs(current_in_A))
            thr = 0.1 * peak_curr
            mask = np.abs(current_in_A) > thr
            
            if not np.any(mask):
                return {
                    'integral_value': '0.0000 C',
                    'capacitance_uF_cm2': '0.0000 µF/cm²',
                    'cycle_indices': self.cycle_indices
                }
            
            # Calculate charge and capacitance
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