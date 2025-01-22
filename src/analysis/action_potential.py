import numpy as np
from scipy import signal
from src.utils.logger import app_logger
from scipy.signal import savgol_filter

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

    def process_signal(self, use_alternative_method=False):
        """
        Process the signal and store all results.
        Added parameter use_alternative_method to choose integration method.

        Returns:
            A tuple of 8 items:
            0) self.processed_data
            1) self.orange_curve
            2) self.orange_curve_times
            3) self.normalized_curve
            4) self.normalized_curve_times
            5) self.average_curve
            6) self.average_curve_times
            7) results (dict)
        """
        try:
            # 1. Baseline correction
            self.baseline_correction_initial()

            # 2. Advanced baseline normalization
            self.advanced_baseline_normalization()

            # 3. Generate orange curve (50-point average)
            self.generate_orange_curve()

            # 4. Calculate normalized curve
            self.normalized_curve, self.normalized_curve_times = self.calculate_normalized_curve()

            # 5. Calculate average of the four segments in the normalized curve (flexible approach)
            self.average_curve, self.average_curve_times = self.calculate_segment_average()

            # 6. Identify cycles
            self.find_cycles()

            # 7. Perform integration (standard or alternative)
            if use_alternative_method:
                results = self.calculate_alternative_integral()
            else:
                results = self.calculate_integral()

            # 8. Return all relevant data
            return (
                self.processed_data,          # 0
                self.orange_curve,            # 1
                self.orange_curve_times,      # 2
                self.normalized_curve,        # 3
                self.normalized_curve_times,  # 4
                self.average_curve,           # 5
                self.average_curve_times,     # 6
                results                       # 7
            )

        except Exception as e:
            app_logger.error(f"Error in process_signal: {str(e)}")
            return (
                None, None, None, None,
                None, None, None,
                {
                    'integral_value': f"Error: {str(e)}",
                    'capacitance_uF_cm2': 'Error',
                    'cycle_indices': []
                }
            )

    def calculate_segment_average(self):
        """
        Calculate an average curve from the normalized curve by splitting it into
        four roughly equal slices. Each slice might be shorter than 200 points if
        total length isn't a multiple of 800. We then truncate slices to the same
        minimal length and do a point-by-point average.
        """
        try:
            if self.normalized_curve is None:
                return None, None

            total_len = len(self.normalized_curve)
            if total_len < 4 * 50:
                # e.g. if we want at least ~50+ points per slice
                app_logger.error(
                    f"Normalized curve only has {total_len} points; need >=200 total for 4 slices."
                )
                return None, None

            # Split into 4 slices
            slice_len = total_len // 4
            seg1 = self.normalized_curve[0 : slice_len]
            seg2 = self.normalized_curve[slice_len : 2*slice_len]
            seg3 = self.normalized_curve[2*slice_len : 3*slice_len]
            seg4 = self.normalized_curve[3*slice_len : 4*slice_len]

            # Corresponding times
            times1 = self.normalized_curve_times[0 : slice_len]
            times2 = self.normalized_curve_times[slice_len : 2*slice_len]
            times3 = self.normalized_curve_times[2*slice_len : 3*slice_len]
            times4 = self.normalized_curve_times[3*slice_len : 4*slice_len]

            # Make them all the same minimal length
            min_len = min(len(seg1), len(seg2), len(seg3), len(seg4))
            seg1 = seg1[:min_len]
            seg2 = seg2[:min_len]
            seg3 = seg3[:min_len]
            seg4 = seg4[:min_len]

            times1 = times1[:min_len]  # we'll use times1 as reference for final average

            all_segments = np.vstack([seg1, seg2, seg3, seg4])
            average_curve = np.mean(all_segments, axis=0)

            self.average_curve = average_curve
            self.average_curve_times = times1

            app_logger.info("Calculated flexible average curve from 4 slices")
            app_logger.debug(
                f"Average curve range: [{np.min(average_curve):.2f}, {np.max(average_curve):.2f}], "
                f"length={len(average_curve)}"
            )
            return average_curve, times1

        except Exception as e:
            app_logger.error(f"Error calculating segment average: {str(e)}")
            return None, None
        
    def apply_enhanced_smoothing(self, data, window_size=7, passes=2):
        if len(data) < window_size:
            return data
        smoothed = data.copy()
        for _ in range(passes):
            smoothed = savgol_filter(smoothed, window_size, polyorder=2)
        return smoothed
        
    def apply_average_to_peaks(self):
        """
        Add average_curve to high hyperpolarization and subtract from high depolarization,
        with special handling for depolarization endpoint.
        """
        try:
            if self.average_curve is None:
                self.average_curve, _ = self.calculate_segment_average()
                if self.average_curve is None:
                    return None, None, None, None

            # Segment indices
            depol_start = 835
            depol_end   = 1034
            hyperpol_start = 1035
            hyperpol_end   = 1234

            if len(self.orange_curve) < hyperpol_end:
                app_logger.error(f"Orange curve too short ({len(self.orange_curve)} points)")
                return None, None, None, None

            # Extract segments
            depol_data = self.orange_curve[depol_start:depol_end].copy()
            depol_times = self.orange_curve_times[depol_start:depol_end]
            hyperpol_data = self.orange_curve[hyperpol_start:hyperpol_end].copy()
            hyperpol_times = self.orange_curve_times[hyperpol_start:hyperpol_end]

            # Resample average curve if needed
            segment_length = min(len(depol_data), len(hyperpol_data))
            if len(self.average_curve) != segment_length:
                original_points = np.linspace(0, 1, len(self.average_curve))
                new_points = np.linspace(0, 1, segment_length)
                avg_curve = np.interp(new_points, original_points, self.average_curve)
            else:
                avg_curve = self.average_curve.copy()

            # Apply modifications
            voltage_diff = abs(self.params['V2'] - self.params['V0'])
            scaled_curve = avg_curve * voltage_diff * 0.2

            hyperpol_modified = hyperpol_data + scaled_curve
            depol_modified = depol_data - scaled_curve

            # Handle endpoints differently for each segment
            blend_points = 25
            
            # For depolarization: gradually reduce effect near end
            for i in range(blend_points):
                weight = ((blend_points - i) / blend_points) ** 2  # Quadratic falloff
                idx = -(i + 1)
                # Blend between modified and original data
                depol_modified[idx] = (weight * depol_modified[idx] + 
                                    (1 - weight) * depol_data[idx])

            # For hyperpolarization: direct return to original data
            hyperpol_modified[-blend_points:] = hyperpol_data[-blend_points:]

            # Store results
            self.modified_hyperpol = hyperpol_modified
            self.modified_hyperpol_times = hyperpol_times
            self.modified_depol = depol_modified
            self.modified_depol_times = depol_times

            return (
                self.modified_hyperpol,
                self.modified_hyperpol_times,
                self.modified_depol,
                self.modified_depol_times
            )

        except Exception as e:
            app_logger.error(f"Error applying average to peaks: {str(e)}")
            return None, None, None, None

    def baseline_correction_initial(self):
        """Initial baseline correction excluding outlier points."""
        sampling_rate = 1.0 / np.mean(np.diff(self.time_data))
        t0_samps = int(self.params['t0'] * sampling_rate / 1000)
        baseline_window = min(t0_samps, 1000, len(self.data))
        
        if baseline_window < 1:
            self.baseline = np.median(self.data)
        else:
            initial_data = self.data[:baseline_window]
            med = np.median(initial_data)
            std = np.std(initial_data)
            mask = np.abs(initial_data - med) < 3 * std
            if np.any(mask):
                self.baseline = np.median(initial_data[mask])
            else:
                self.baseline = med
        
        self.processed_data = self.data - self.baseline
        app_logger.info(f"Initial baseline correction: subtracted {self.baseline:.2f} pA")

    def advanced_baseline_normalization(self):
        """Align segments, remove offsets for hyper/depolarization."""
        try:
            sampling_rate = 1.0 / np.mean(np.diff(self.time_data))
            threshold_slope = 3 * np.std(np.diff(self.processed_data))
            segment_start = 0
            segments = []
            
            for i in range(1, len(self.processed_data)):
                slope = abs(self.processed_data[i] - self.processed_data[i-1])
                if slope > threshold_slope:
                    if i + 1 < len(self.processed_data):
                        next_slope = abs(self.processed_data[i+1] - self.processed_data[i])
                        if next_slope > threshold_slope and (i - segment_start > 50):
                            segments.append((segment_start, i))
                            segment_start = i + 1
            
            if segment_start < len(self.processed_data) - 50:
                segments.append((segment_start, len(self.processed_data)))
            
            baseline_window = 50
            for start, end in segments:
                pre_region = self.processed_data[start : min(start + baseline_window, end)]
                post_region = self.processed_data[max(start, end - baseline_window) : end]
                if len(pre_region) > 0 and len(post_region) > 0:
                    pre_baseline = np.median(pre_region)
                    post_baseline = np.median(post_region)
                    if np.std(pre_region) < np.std(post_region):
                        baseline = pre_baseline
                    else:
                        baseline = post_baseline
                    self.processed_data[start:end] -= baseline
                    if start > 0:
                        blend_points = min(10, start)
                        weights = np.linspace(0, 1, blend_points)
                        self.processed_data[start-blend_points:start] = (
                            weights * self.processed_data[start]
                            + (1 - weights) * self.processed_data[start - blend_points]
                        )
            
            app_logger.info(f"Advanced normalization completed with {len(segments)} segments")
            
        except Exception as e:
            app_logger.error(f"Error in advanced_baseline_normalization: {str(e)}")
            raise

    def generate_orange_curve(self):
        """Generate decimated curve by taking one average point per 50 points."""
        try:
            orange_points = []
            orange_times = []
            window_size = 50
            for i in range(0, len(self.processed_data), window_size):
                window = self.processed_data[i : i + window_size]
                time_window = self.time_data[i : i + window_size]
                if len(window) > 0:
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

    def calculate_normalized_curve(self):
        """Build normalized curves for hyper/depolarization segments."""
        try:
            if self.orange_curve is None:
                return None, None

            segments = [
                {"start": 35, "end": 234, "is_hyperpol": True},
                {"start": 235, "end": 434, "is_hyperpol": False},
                {"start": 435, "end": 634, "is_hyperpol": True},
                {"start": 635, "end": 834, "is_hyperpol": False}
            ]
            
            normalized_points = []
            normalized_times = []
            
            for seg in segments:
                start_idx = seg["start"]
                end_idx = seg["end"]
                if end_idx >= len(self.orange_curve):
                    app_logger.warning(f"Segment {start_idx}-{end_idx} out of range.")
                    continue
                selected_points = self.orange_curve[start_idx:end_idx]
                selected_times = self.orange_curve_times[start_idx:end_idx]
                voltage_step = -20.0 if seg["is_hyperpol"] else 20.0
                segment_norm = selected_points / voltage_step
                normalized_points.extend(segment_norm)
                normalized_times.extend(selected_times)
                app_logger.debug(f"Processed segment {start_idx+1}-{end_idx}: "
                                 f"{'hyperpolarization' if seg['is_hyperpol'] else 'depolarization'}")
                app_logger.debug(f"Voltage step: {voltage_step} mV")
                app_logger.debug(f"Current range: [{np.min(selected_points):.2f}, {np.max(selected_points):.2f}] pA")
                app_logger.debug(f"Conductance range: [{np.min(segment_norm):.2f}, {np.max(segment_norm):.2f}] nS")
            
            self.normalized_curve = np.array(normalized_points)
            self.normalized_curve_times = np.array(normalized_times)
            app_logger.info("Conductance values calculated for all segments")
            return self.normalized_curve, self.normalized_curve_times
        
        except Exception as e:
            app_logger.error(f"Error calculating normalized curve: {str(e)}")
            return None, None

    def find_cycles(self):
        """Identify consistent hyperpolarization cycles."""
        try:
            sampling_rate = 1.0 / np.mean(np.diff(self.time_data))
            t1_samps = int(self.params['t1'] * sampling_rate / 1000)
            neg_peaks, _ = signal.find_peaks(
                -self.processed_data,
                prominence=np.std(self.processed_data)*1.5,
                distance=t1_samps,
                width=(t1_samps//4, t1_samps)
            )
            
            min_depth = np.max(np.abs(self.processed_data)) * 0.4
            valid_peaks = []
            for peak in neg_peaks:
                if abs(self.processed_data[peak]) > min_depth:
                    valid_peaks.append(peak)
            
            window_before = t1_samps // 2
            window_after = int(t1_samps * 1.5)
            
            self.cycles = []
            self.cycle_times = []
            self.cycle_indices = []
            
            for i, peak in enumerate(valid_peaks[:self.params['n_cycles']]):
                start = max(0, peak - window_before)
                end = min(len(self.processed_data), peak + window_after)
                if end - start == window_before + window_after:
                    cycle = self.processed_data[start:end]
                    cycle_time = self.time_data[start:end] - self.time_data[start]
                    self.cycles.append(cycle)
                    self.cycle_times.append(cycle_time)
                    self.cycle_indices.append((start, end))
                    app_logger.debug(f"Cycle {i+1} found at index {peak}")

            if not self.cycles:
                app_logger.warning("No valid cycles found")
                return
            if len(self.cycles) >= 2:
                correlations = []
                reference_cycle = self.cycles[0]
                for cyc in self.cycles[1:]:
                    corr = np.corrcoef(reference_cycle, cyc)[0, 1]
                    correlations.append(corr)
                    app_logger.debug(f"Cycle correlation: {corr:.3f}")
                if any(c < 0.8 for c in correlations):
                    app_logger.warning("Detected cycles show significant differences")
        except Exception as e:
            app_logger.error(f"Error in cycle detection: {str(e)}")
            raise

    def calculate_alternative_integral(self):
        """Calculate integral using the entire normalized_curve average approach."""
        try:
            if self.normalized_curve is None:
                self.normalized_curve, self.normalized_curve_times = self.calculate_normalized_curve()
            if self.normalized_curve is None:
                return {
                    'integral_value': 'No normalized curve available',
                    'capacitance_uF_cm2': '0.0000 µF/cm²'
                }

            avg_norm = np.mean(self.normalized_curve)
            dt = np.mean(np.diff(self.normalized_curve_times))
            integral = np.trapz(self.normalized_curve, dx=dt)
            voltage_diff = abs(self.params['V2'] - self.params['V0'])
            total_integral = integral * voltage_diff

            total_cap_F = abs(total_integral * 1e-12)
            total_cap_uF = total_cap_F * 1e6
            area = self.params.get('cell_area_cm2', 1e-4)
            cap_uF_cm2 = total_cap_uF / area
            
            results = {
                'integral_value': f"{abs(total_integral):.6e} C",
                'capacitance_uF_cm2': f"{cap_uF_cm2:.4f} µF/cm²",
                'cycle_indices': self.cycle_indices,
                'raw_values': {
                    'normalized_integral': integral,
                    'voltage_diff': voltage_diff,
                    'capacitance_F': total_cap_F,
                    'area_cm2': area
                }
            }
            app_logger.info(f"Alternative integration method - "
                            f"Capacitance: {cap_uF_cm2:.4f} µF/cm²")
            return results
        except Exception as e:
            app_logger.error(f"Error in alternative integration: {str(e)}")
            return {
                'integral_value': f"Error: {str(e)}",
                'capacitance_uF_cm2': 'Error',
                'cycle_indices': []
            }

    def calculate_integral(self):
        """Calculate integral and capacitance from the first cycle (standard method)."""
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
            voltage_diff_in_V = (self.params['V1'] - self.params['V0'])

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
