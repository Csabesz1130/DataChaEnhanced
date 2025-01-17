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

        app_logger.debug(f"Parameters validated: {self.params}")

    def process_signal(self):
        """
        The main pipeline:
        1. baseline_correction_initial
        2. advanced_baseline_normalization
        3. find_cycles
        4. calculate_integral
        """
        try:
            self.baseline_correction_initial()
            self.advanced_baseline_normalization()
            self.find_cycles()
            results = self.calculate_integral()

            if not results:
                return None, None, {
                    'integral_value': 'No analysis performed',
                    'capacitance_uF_cm2': 'No analysis performed',
                    'cycle_indices': []
                }
            return self.processed_data, self.time_data, results

        except Exception as e:
            app_logger.error(f"Error in process_signal: {str(e)}")
            return None, None, {
                'integral_value': f"Error: {str(e)}",
                'capacitance_uF_cm2': 'Error',
                'cycle_indices': []
            }

    def baseline_correction_initial(self):
        """
        Step 1: Subtract the median of the first t0 ms (or up to 1000 points).
        """
        sampling_rate = 1.0 / np.mean(np.diff(self.time_data))
        t0_samps = int(self.params['t0'] * sampling_rate / 1000)
        baseline_window = min(t0_samps, 1000, len(self.data))

        if baseline_window < 1:
            self.baseline = np.median(self.data)
        else:
            self.baseline = np.median(self.data[:baseline_window])

        # Subtract baseline
        self.processed_data = self.data - self.baseline
        app_logger.info(f"Initial baseline correction: subtracted {self.baseline:.2f} pA")

    def advanced_baseline_normalization(self):
        """
        Advanced normalization to align segments to zero based on abrupt slope changes.
        """
        try:
            sampling_rate = 1.0 / np.mean(np.diff(self.time_data))
            segment_start = 0
            threshold_slope = 3 * np.std(np.diff(self.processed_data))

            while segment_start < len(self.processed_data):
                segment_end = segment_start

                # Detect abrupt slope changes
                while segment_end < len(self.processed_data) - 1:
                    slope = abs(self.processed_data[segment_end + 1] - self.processed_data[segment_end])
                    if slope > threshold_slope:
                        break
                    segment_end += 1

                # Calculate average for last 50 points in the segment
                avg_window_start = max(segment_start, segment_end - 50)
                avg_window_end = segment_end
                segment_mean = np.mean(self.processed_data[avg_window_start:avg_window_end])

                # Subtract the segment mean to align to zero
                self.processed_data[segment_start:segment_end] -= segment_mean

                segment_start = segment_end + 1

            # Final alignment to zero
            global_mean = np.mean(self.processed_data)
            self.processed_data -= global_mean

            app_logger.info("Advanced baseline normalization applied with segment alignment.")

        except Exception as e:
            app_logger.error(f"Error in advanced_baseline_normalization: {str(e)}")
            raise

    def find_cycles(self):
        """
        Step 3: Identify negative peaks using find_peaks on -self.processed_data.
        """
        sampling_rate = 1.0 / np.mean(np.diff(self.time_data))
        # use t1 to set a typical cycle distance
        t1_samps = int(self.params['t1'] * sampling_rate / 1000)

        neg_peaks, _ = signal.find_peaks(-self.processed_data,
                                         prominence=np.std(self.processed_data),
                                         distance=t1_samps)

        for i, peak in enumerate(neg_peaks[: self.params['n_cycles']]):
            start = max(0, peak - t1_samps // 2)
            end = min(len(self.processed_data), peak + t1_samps * 2)
            cycle = self.processed_data[start:end]
            cycle_time = self.time_data[start:end] - self.time_data[start]

            self.cycles.append(cycle)
            self.cycle_times.append(cycle_time)
            self.cycle_indices.append((start, end))
            app_logger.debug(f"Cycle {i+1} found at index {peak}")

    def calculate_integral(self):
        """
        Step 4: Integrate the first cycle above a 10% threshold => total charge => 
        compute capacitance => return results.
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

            # find threshold
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
