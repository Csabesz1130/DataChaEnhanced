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
          2. multi_segment_normalization
          3. find_cycles
          4. calculate_integral
        """
        try:
            self.baseline_correction_initial()
            self.multi_segment_normalization()
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

        self.processed_data = self.data - self.baseline
        app_logger.info(f"Initial baseline correction: subtracted {self.baseline:.2f} pA")

    def multi_segment_normalization(self):
        """
        Advanced multi-step normalization with auto-detected plateau:
        - For each segment i>0:
            1) Identify a stable plateau region by scanning rolling std/derivative.
            2) Fit a 2-point line from plateau[0:20] -> plateau[-20:] offsets.
            3) Subtract that line from the entire segment.
        - If detection fails, fallback to a simpler skip-then-measure median approach.
        """
        try:
            sampling_rate = 1.0 / np.mean(np.diff(self.time_data))
            t0_samps = int(self.params['t0'] * sampling_rate / 1000)
            t1_samps = int(self.params['t1'] * sampling_rate / 1000)
            t2_samps = int(self.params['t2'] * sampling_rate / 1000)
            t3_samps = int(self.params.get('t3', 0) * sampling_rate / 1000)

            seg_lengths = [t0_samps, t1_samps, t2_samps, t3_samps]
            segments = []
            idx_start = 0
            
            # Build segment list
            for length in seg_lengths:
                if length <= 0:
                    continue
                idx_end = min(idx_start + length, len(self.processed_data))
                if idx_end <= idx_start:
                    break
                segments.append((idx_start, idx_end))
                idx_start = idx_end

            # Segment0 pinned by baseline_correction_initial()
            for seg_i in range(1, len(segments)):
                s_start, s_end = segments[seg_i]
                seg_len = s_end - s_start
                if seg_len < 50:
                    continue

                # We'll attempt to detect a stable plateau in the segment
                seg_data = self.processed_data[s_start:s_end].copy()

                # 1) Rolling derivative or std detection
                #    We'll compute a rolling standard deviation in windows of ~50 points
                #    and pick the region with the smallest std.
                window_size = max(30, seg_len // 10)  # e.g. 30 or 1/10th of segment
                rolling_std = []
                for i in range(seg_len):
                    start_i = max(0, i - window_size // 2)
                    end_i   = min(seg_len, i + window_size // 2)
                    sub = seg_data[start_i:end_i]
                    rolling_std.append(np.std(sub))
                rolling_std = np.array(rolling_std)

                # find the index of minimal std
                best_idx = np.argmin(rolling_std)
                # define a plateau region around best_idx
                plateau_half = window_size // 2
                plat_start = max(0, best_idx - plateau_half)
                plat_end   = min(seg_len, best_idx + plateau_half)
                
                # Ensure we have at least 40 points or so
                if (plat_end - plat_start) < 40:
                    # fallback simpler approach
                    plateau_offset = np.median(seg_data[10:50])
                    self.processed_data[s_start:s_end] -= plateau_offset
                    continue

                # 2) Two‐point linear baseline from the plateau
                #    We'll measure the first ~20 points and last ~20 points in that plateau region.
                sub_plat = seg_data[plat_start:plat_end]
                if (plat_end - plat_start) < 50:
                    # if short, just do a single median
                    plateau_offset = np.median(sub_plat)
                    self.processed_data[s_start:s_end] -= plateau_offset
                    continue

                offset_start = np.median(sub_plat[:20])
                offset_end   = np.median(sub_plat[-20:])
                plateau_len  = (plat_end - plat_start)

                slope = (offset_end - offset_start) / max(1, plateau_len - 1)

                # 3) Subtract line from entire segment
                for i in range(seg_len):
                    # map i -> local plateau index
                    # let's define i_plat = i - plat_start, clamped
                    i_plat = min(max(i - plat_start, 0), plateau_len - 1)
                    local_offset = offset_start + slope * i_plat
                    self.processed_data[s_start + i] -= local_offset

            app_logger.info("Advanced multi-segment normalization with auto plateau detection done.")

        except Exception as e:
            app_logger.error(f"Error in multi_segment_normalization: {str(e)}")
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
