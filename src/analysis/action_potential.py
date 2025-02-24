import numpy as np
from scipy import signal
from src.utils.logger import app_logger
import csv
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
            'V2': -20,
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

    def print_curve_points(self):
        """Print point ranges for each curve type after analysis"""
        try:
            n = self.params.get('normalization_points', {}).get('seg1', (35, None))[0]
            
            # Orange curve (50-point average)
            if self.orange_curve is not None:
                total_points = len(self.orange_curve)
                app_logger.info(f"\nORANGE CURVE (50-point average):")
                app_logger.info(f"Total points: {total_points}")
                app_logger.info(f"Point range: {1} - {total_points}")
                
            # Dark blue (Voltage-Normalized)
            if self.normalized_curve is not None:
                total_norm = len(self.normalized_curve)
                segments = [
                    {"range": f"{n} - {n+200}", "type": "hyperpol"},
                    {"range": f"{n+200} - {n+400}", "type": "depol"},
                    {"range": f"{n+400} - {n+600}", "type": "hyperpol"},
                    {"range": f"{n+600} - {n+800}", "type": "depol"}
                ]
                app_logger.info(f"\nDARK BLUE CURVE (Voltage-Normalized):")
                app_logger.info(f"Total points: {total_norm}")
                for i, seg in enumerate(segments, 1):
                    app_logger.info(f"Segment {i}: {seg['range']} ({seg['type']})")
                    
            # Magenta (Averaged Normalized)
            if self.average_curve is not None:
                total_avg = len(self.average_curve)
                app_logger.info(f"\nMAGENTA CURVE (Averaged Normalized):")
                app_logger.info(f"Total points: {total_avg}")
                app_logger.info(f"Point range: {n} - {n+total_avg}")
                
            # Purple (Modified Peaks)
            if hasattr(self, 'modified_hyperpol') and hasattr(self, 'modified_depol'):
                hyperpol_len = len(self.modified_hyperpol) if self.modified_hyperpol is not None else 0
                depol_len = len(self.modified_depol) if self.modified_depol is not None else 0
                app_logger.info(f"\nPURPLE CURVE (Modified Peaks):")
                app_logger.info(f"Hyperpolarization points: {hyperpol_len}")
                app_logger.info(f"Depolarization points: {depol_len}")
                if hasattr(self, 'modified_hyperpol_times') and self.modified_hyperpol_times is not None:
                    t_start = self.modified_hyperpol_times[0]
                    t_end = self.modified_hyperpol_times[-1]
                    app_logger.info(f"Hyperpol time range: {t_start*1000:.3f} - {t_end*1000:.3f} ms")
                if hasattr(self, 'modified_depol_times') and self.modified_depol_times is not None:
                    t_start = self.modified_depol_times[0]
                    t_end = self.modified_depol_times[-1]
                    app_logger.info(f"Depol time range: {t_start*1000:.3f} - {t_end*1000:.3f} ms")
                    
        except Exception as e:
            app_logger.error(f"Error printing curve points: {str(e)}")

    def print_curve_points_relationship(self):
        """Print point relationships including segment ranges."""
        try:
            n = self.get_segment_start()  # Get consistent start point

            if self.average_curve is not None:
                avg_start_ms = self.average_curve_times[0] * 1000
                avg_end_ms = self.average_curve_times[-1] * 1000
                app_logger.info("\nMAGENTA CURVE (Average):")
                app_logger.info(f"Points: {len(self.average_curve)}")
                app_logger.info(f"Time range: {avg_start_ms:.2f} - {avg_end_ms:.2f} ms")

            if hasattr(self, 'modified_hyperpol'):
                app_logger.info("\nPURPLE CURVE - Hyperpolarization:")
                # Original orange segment
                app_logger.info(f"Orange source points: {n+1000} - {n+1200}")
                
                if self.modified_hyperpol is not None:
                    hyp_start_ms = self.modified_hyperpol_times[0] * 1000
                    hyp_end_ms = self.modified_hyperpol_times[-1] * 1000
                    app_logger.info(f"Points: {len(self.modified_hyperpol)}")
                    app_logger.info(f"Time range: {hyp_start_ms:.2f} - {hyp_end_ms:.2f} ms")

            if hasattr(self, 'modified_depol'):
                app_logger.info("\nPURPLE CURVE - Depolarization:")
                # Original orange segment
                app_logger.info(f"Orange source points: {n+800} - {n+1000}")
                
                if self.modified_depol is not None:
                    dep_start_ms = self.modified_depol_times[0] * 1000
                    dep_end_ms = self.modified_depol_times[-1] * 1000
                    app_logger.info(f"Points: {len(self.modified_depol)}")
                    app_logger.info(f"Time range: {dep_start_ms:.2f} - {dep_end_ms:.2f} ms")
                    
            app_logger.info("\nPoint Relationships:")
            app_logger.info(f"Dark blue curve: {n} - {n+800}")
            app_logger.info(f"Magenta curve: {len(self.average_curve)} pts")
            app_logger.info(f"Purple curve source segments:")
            app_logger.info(f"  Depol: {n+800} - {n+1000}")
            app_logger.info(f"  Hyperpol: {n+1000} - {n+1200}")
                        
        except Exception as e:
            app_logger.error(f"Error printing curve relationships: {str(e)}")

    @staticmethod
    def parse_voltage_from_filename(filepath):
        """
        Parse V2 voltage value from filename following patterns like:
        20130507_0006_10mVdepol, 20130406_0023_-50mVdepol
        or
        20130509_0007_-50depol, 20130911_0036_-50depol
        
        Args:
            filepath (str): Full path to ATF file
            
        Returns:
            float: Voltage value in mV or None if not found
        """
        try:
            import os
            import re
            
            # Extract filename without path and extension
            filename = os.path.splitext(os.path.basename(filepath))[0]
            
            # Try first pattern: -50mV, 10mV, -10mV, 50mV followed by "depol"
            pattern1 = r'(-?\d+)mV(?:depol|$)'
            match = re.search(pattern1, filename)
            
            if match:
                voltage_str = match.group(1)
                voltage = float(voltage_str)
                app_logger.debug(f"Found voltage in filename {filename}: {voltage} mV")
                return voltage
                
            # Try second pattern: -50depol, 50depol
            pattern2 = r'(-?\d+)depol'
            match = re.search(pattern2, filename)
            
            if match:
                voltage_str = match.group(1)
                voltage = float(voltage_str)
                app_logger.debug(f"Found voltage in filename {filename}: {voltage} mV")
                return voltage
                
            app_logger.debug(f"No voltage value found in filename: {filename}")
            return None
            
        except Exception as e:
            app_logger.error(f"Error parsing voltage from filename: {str(e)}")
            return None

    def validate_curve_points(self):
        """
        Validate point relationships between curves.
        This logs actual slice indices for purple segments
        instead of computing them from lengths.
        """
        n = self.params.get('normalization_points', {}).get('seg1', (35, None))[0]
        
        # If your code changed the default from 2200 -> 1800, just keep it consistent
        expected_points = {
            'orange': 2200,       # or 1800, whichever you truly expect
            'normalized': 800,    # 4 segments x 200 points
            'average': 200,       # average segment
            'modified': 200       # each modified segment
        }

        expected_ranges = {
            'hyperpol': (n + 1000, n + 1200),
            'depol': (n + 800, n + 1000)
        }
        
        app_logger.info("\nPoint Range Validation:")
        app_logger.info(f"Starting point (n): {n}")

        # 1) Orange count
        if self.orange_curve is not None:
            actual_orange = len(self.orange_curve)
            app_logger.info(
                f"Orange points: {actual_orange} (expected: {expected_points['orange']})"
            )

        # 2) Normalized
        if self.normalized_curve is not None:
            actual_norm = len(self.normalized_curve)
            app_logger.info(
                f"Normalized points: {actual_norm} (expected: {expected_points['normalized']})"
            )

        # 3) Average
        if self.average_curve is not None:
            actual_avg = len(self.average_curve)
            app_logger.info(
                f"Average points: {actual_avg} (expected: {expected_points['average']})"
            )

        # 4) Purple slices
        if hasattr(self, 'modified_hyperpol') and self.modified_hyperpol is not None:
            # We rely on the stored indices from apply_average_to_peaks
            if hasattr(self, '_hyperpol_slice'):
                actual_slice = self._hyperpol_slice
                app_logger.info(
                    f"Hyperpol range: {actual_slice} (expected: {expected_ranges['hyperpol']})"
                )
            else:
                app_logger.info("No hyperpol slice info stored.")
            
        if hasattr(self, 'modified_depol') and self.modified_depol is not None:
            # Same for depol
            if hasattr(self, '_depol_slice'):
                actual_slice = self._depol_slice
                app_logger.info(
                    f"Depol range: {actual_slice} (expected: {expected_ranges['depol']})"
                )
            else:
                app_logger.info("No depol slice info stored.")

    def process_signal(self, use_alternative_method=False):
        """Process signal and generate all curves."""
        try:
            # Initial processing steps
            self.baseline_correction_initial()
            self.advanced_baseline_normalization()
            self.generate_orange_curve()
            self.normalized_curve, self.normalized_curve_times = self.calculate_normalized_curve()
            self.average_curve, self.average_curve_times = self.calculate_segment_average()
            
            # Generate purple curves
            (self.modified_hyperpol, 
            self.modified_hyperpol_times,
            self.modified_depol,
            self.modified_depol_times) = self.apply_average_to_peaks()

            # Find cycles and calculate integrals
            self.find_cycles()
            results = (self.calculate_alternative_integral() 
                    if use_alternative_method 
                    else self.calculate_integral())

            # Print diagnostics
            self.validate_curve_points()
            self.print_curve_points()
            self.print_curve_points_relationship()

            return (
                self.processed_data,
                self.orange_curve,
                self.orange_curve_times, 
                self.normalized_curve,
                self.normalized_curve_times,
                self.average_curve,
                self.average_curve_times,
                results
            )

        except Exception as e:
            app_logger.error(f"Error in process_signal: {str(e)}")
            return (None,) * 7 + ({"error": str(e)},)

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
        """Apply average to peaks with smooth segment transitions."""
        try:
            if self.average_curve is None:
                self.average_curve, _ = self.calculate_segment_average()
                if self.average_curve is None:
                    return None, None, None, None
                
            # Default starting point
            n = 35

            # Use custom starting point if provided
            if 'normalization_points' in self.params:
                norm_points = self.params['normalization_points']
                n = norm_points['seg1'][0]

            # Fixed segment indices
            depol_start = n + 800
            depol_end = n + 1000
            hyperpol_start = n + 1000
            hyperpol_end = n + 1200

            if len(self.orange_curve) < hyperpol_end:
                app_logger.error(f"Orange curve too short ({len(self.orange_curve)} points)")
                return None, None, None, None

            # Extract segments with same length as average curve
            depol_data = self.orange_curve[depol_start:depol_end].copy()
            depol_times = self.orange_curve_times[depol_start:depol_end]
            hyperpol_data = self.orange_curve[hyperpol_start:hyperpol_end].copy()
            hyperpol_times = self.orange_curve_times[hyperpol_start:hyperpol_end]

            # Make average curve match segment length
            if len(self.average_curve) < len(depol_data):
                pad_length = len(depol_data) - len(self.average_curve)
                scaled_curve = np.pad(self.average_curve, (0, pad_length), mode='edge')
            else:
                scaled_curve = self.average_curve[:len(depol_data)]

            # Apply voltage scaling
            voltage_diff = abs(self.params['V2'] - self.params['V0'])
            scaled_curve = scaled_curve * voltage_diff

            # Apply modifications
            hyperpol_modified = hyperpol_data + scaled_curve
            depol_modified = depol_data - scaled_curve

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

    def get_segment_start(self):
        """Get consistent segment start point."""
        if 'normalization_points' in self.params:
            return self.params['normalization_points']['seg1'][0]
        return 35  # Default start point

    def get_segment_bounds(self):
        """Get segment boundaries consistently."""
        n = self.get_segment_start()
        bounds = {
            'normalized': [
                (n, n + 200),
                (n + 200, n + 400),
                (n + 400, n + 600),
                (n + 600, n + 800)
            ],
            'peaks': {
                'depol': (n + 800, n + 1000),
                'hyperpol': (n + 1000, n + 1200)
            }
        }
        return bounds

    def advanced_baseline_normalization(self):
        """Align segments, remove offsets for hyper/depolarization."""
        try:
            sampling_rate = 1.0 / np.mean(np.diff(self.time_data))
            { app_logger.info(f"Sampling rate: {sampling_rate}") }
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
        """Build normalized curves using starting point from user input."""
        try:
            if self.orange_curve is None:
                return None, None
                
            # Default starting point
            n = 35

            # Use custom starting point if provided
            if 'normalization_points' in self.params:
                norm_points = self.params['normalization_points']
                n = norm_points['seg1'][0]

            # Create segments based on starting point
            segments = [
                {"start": n,      "end": n + 200, "is_hyperpol": True},
                {"start": n+200,  "end": n + 400, "is_hyperpol": False},
                {"start": n+400,  "end": n + 600, "is_hyperpol": True},
                {"start": n+600,  "end": n + 800, "is_hyperpol": False}
            ]
            
            normalized_points = []
            normalized_times = []
            
            for seg in segments:
                start_idx = seg["start"]
                end_idx = seg["end"]
                
                if end_idx >= len(self.orange_curve):
                    app_logger.warning(f"Segment {start_idx+1}-{end_idx+1} out of range.")
                    continue
                    
                selected_points = self.orange_curve[start_idx:end_idx]
                selected_times = self.orange_curve_times[start_idx:end_idx]
                voltage_step = -20.0 if seg["is_hyperpol"] else 20.0
                segment_norm = selected_points / voltage_step
                normalized_points.extend(segment_norm)
                normalized_times.extend(selected_times)
                
                app_logger.debug(f"Processed segment {start_idx+1}-{end_idx+1}")
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
            t1_samps = int(self.params['t1'] * sampling_rate)
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

    def calculate_integral(self):
        """
        Integrates the first detected cycle (self.cycles[0]) with baseline correction.
        The relationship pA×ms = pC and pC / mV = nF is used.
        Typically, the professors expect 1–3 nF values in these measurements.
        """
        try:
            # If no cycles have been found, there's nothing to integrate
            if not self.cycles:
                return {'error': 'No cycles found'}

            # 1) Get the data (pA) and time (seconds) from the first cycle
            cycle_data = self.cycles[0]
            cycle_time = self.cycle_times[0]

            # 2) Convert time from seconds to milliseconds for trapezoidal integration
            current_pA = cycle_data
            time_ms = cycle_time

            # 3) Voltage difference in mV (e.g. V1=-100 mV, V0=-80 mV => 20 mV)
            #delta_V_mV = abs(self.params['V1'] - self.params['V0'])

            # 4) Baseline correction (for instance, the median of the first 20 points)
            #baseline = np.median(current_pA[:20])
            #current_corrected = current_pA - baseline

            # 5) Trapezoidal integration: 1 pA × 1 ms = 1 pC
            charge_pC = 0

            # 6) Capacitance: 1 pC / 1 mV = 1 nF
            capacitance_nF = 0

            # 7) (Optional) Normalize to cell area, if 'cell_area_cm2' is provided
            area_cm2 = self.params.get('cell_area_cm2', 1e-4)
            specific_capacitance_nF_cm2 = capacitance_nF / area_cm2

            # 8) Validation against 1–3 nF range
            validation_result = 'OK' if 1 <= capacitance_nF <= 3 else 'OUT_OF_RANGE'
            if validation_result == 'OUT_OF_RANGE':
                app_logger.warning(
                    f"Capacitance {capacitance_nF:.2f} nF is outside the expected 1–3 nF range."
                )

            return {
                'charge_pC': f"{charge_pC:.2f} pC",
                'capacitance_nF': f"{capacitance_nF:.2f} nF",
                'specific_capacitance': f"{specific_capacitance_nF_cm2:.2f} nF/cm²",
                'validation': validation_result
            }

        except Exception as e:
            app_logger.error(f"Error in calculate_integral(): {str(e)}")
            return {'error': str(e)}


    def calculate_alternative_integral(self):
        """
        Example of an alternative integration approach, also using pA–ms for 
        the integral and mV for the voltage difference, yielding nF in the end.
        This differs from the 'classic' method in that it may, for instance, 
        integrate the normalized curve or another segment.
        """
        try:
            # Make sure the normalized curve exists if we want to use it
            #tehát itt az average curve-t kellene használni
            if not hasattr(self, 'average_curve') or self.average_curve is None:
                self.average_curve, self.average_curve_times = self.calculate_segment_average()

            # If there's still no normalized curve, we cannot proceed
            if self.average_curve is None or self.average_curve_times is None:
                return {
                    'integral_value': 'No normalized curve available',
                    'capacitance_nF': 'N/A'
                }

            # Assume the normalized_curve is also in pA. If it's actually nS or something else, adjust accordingly.
            current_pA = self.average_curve
            time_ms = self.average_curve_times

            # Optional baseline correction (example: median of the first 10 points)
            #baseline_alt = np.median(current_pA[:10])
            #current_corrected = current_pA 
            # Voltage difference in mV, e.g. between V2 and V0
            delta_V_mV = abs(self.params['V2'] - self.params['V0'])

            # Integration in pA–ms => pC
            charge_pC = np.trapz(current_pA, x=time_ms)

            # 1 pC / 1 mV = 1 nF
            capacitance_nF = charge_pC

            return {
                'method': 'alternative',
                'integral_value': f"{charge_pC:.2f} pC,",
                'capacitance_nF': f"{capacitance_nF:.2f} nF"
            }

        except Exception as e:
            app_logger.error(f"Error in calculate_alternative_integral(): {str(e)}")
            return {
                'integral_value': f"Error: {str(e)}",
                'capacitance_nF': 'Error'
            }
        
    def calculate_purple_integrals(self):
        """
        Integrates the 'modified' hyperpolarization (self.modified_hyperpol) and
        depolarization (self.modified_depol) segments separately, returning dimensioned
        results in pC.
        """
        try:
            if (not hasattr(self, 'modified_hyperpol') or self.modified_hyperpol is None
                or not hasattr(self, 'modified_hyperpol_times') or self.modified_hyperpol_times is None
                or not hasattr(self, 'modified_depol') or self.modified_depol is None
                or not hasattr(self, 'modified_depol_times') or self.modified_depol_times is None):
                return {
                    'error': 'No modified purple data available (hyperpol/depol missing)'
                }

            # Assume modified_*_times are already in ms
            hyperpol_current_pA = self.modified_hyperpol
            hyperpol_time_ms = self.modified_hyperpol_times
            depol_current_pA = self.modified_depol
            depol_time_ms = self.modified_depol_times

            # Integrate pA × ms => pC
            hyperpol_charge_pC = np.trapz(hyperpol_current_pA, x=hyperpol_time_ms)
            depol_charge_pC = np.trapz(depol_current_pA, x=depol_time_ms)

            # Format them
            # For example, if you just want a single integral_value describing both:
            purple_integral_str = (f"Hyperpol: {hyperpol_charge_pC:.2f} pC, "
                                f"Depol: {depol_charge_pC:.2f} pC")

            return {
                # Let’s name it differently so it doesn’t overwrite the main integral_value
                'purple_integral_value': purple_integral_str,
                'hyperpol_area': f"{hyperpol_charge_pC:.2f} pC",
                'depol_area': f"{depol_charge_pC:.2f} pC"
            }

        except Exception as e:
            app_logger.error(f"Error in calculate_purple_integrals(): {str(e)}")
            return {
                'purple_integral_value': f"Error: {str(e)}"
            }

    # Add to src/analysis/action_potential.py

def export_all_curves(self, results_dict, filename):
    """Export the current data to a CSV file with filename, V2 voltage, and organized sections"""
    try:
        import csv
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            
            # Write V2 voltage
            writer.writerow(["V2 (mV):", f"{self.params.get('V2', 'N/A')}"])
            writer.writerow([])
            
            # Write overall integral at the top
            writer.writerow(["Overall Integral (pC):", results_dict.get('integral_value', 'N/A')])
            writer.writerow([])  # blank line

            # ============ 1) Purple Hyperpol Section ============
            writer.writerow(["PURPLE HYPERPOL CURVE"])
            writer.writerow(["Hyperpol Integral from the app:", results_dict.get('hyperpol_area', 'N/A')])
            writer.writerow(["Index", "Hyperpol_pA", "Hyperpol_time_ms"])

            if hasattr(self, 'modified_hyperpol') and hasattr(self, 'modified_hyperpol_times'):
                for i in range(len(self.modified_hyperpol)):
                    writer.writerow([
                        i + 1,
                        f"{self.modified_hyperpol[i]:.7f}",
                        f"{self.modified_hyperpol_times[i]*1000:.7f}"
                    ])
            writer.writerow([])  # blank line separator

            # ============ 2) Purple Depol Section ===============
            writer.writerow(["PURPLE DEPOL CURVE"])
            writer.writerow(["Depol Integral from the app:", results_dict.get('depol_area', 'N/A')])
            writer.writerow(["Index", "Depol_pA", "Depol_time_ms"])

            if hasattr(self, 'modified_depol') and hasattr(self, 'modified_depol_times'):
                for i in range(len(self.modified_depol)):
                    writer.writerow([
                        i + 1,
                        f"{self.modified_depol[i]:.7f}",
                        f"{self.modified_depol_times[i]*1000:.7f}"
                    ])
            writer.writerow([])

            # ============ 3) Purple Integral Summary =============
            writer.writerow(["Purple Integral Summary:", results_dict.get('purple_integral_value', 'N/A')])
            writer.writerow([])

        app_logger.info(f"Exported purple curves to {filename}")
        return True

    except Exception as e:
        app_logger.error(f"Error exporting curves: {str(e)}")
        return False

# Add these methods to the ActionPotentialProcessor class in action_potential.py

def remove_extreme_outliers(self, data, times):
        """
        Remove obvious extreme outliers from the data.
        
        Args:
            data (np.array): Current values
            times (np.array): Time values
            
        Returns:
            tuple: (cleaned_data, cleaned_times)
        """
        try:
            if data is None or times is None:
                return None, None
                
            data = np.array(data)
            times = np.array(times)
            
            # Calculate median of the data
            median = np.median(data)
            
            # Set threshold for extreme outliers (5x median)
            threshold = 5 * abs(median)
            
            # Create mask for non-outlier points
            mask = np.abs(data) < threshold
            
            # Get cleaned data
            cleaned_data = data[mask]
            cleaned_times = times[mask]
            
            # Log removed points
            outlier_indices = np.where(~mask)[0]
            outlier_values = data[~mask]
            
            app_logger.info(f"Removed {len(outlier_values)} extreme outliers")
            app_logger.debug(f"Outlier indices: {outlier_indices}")
            app_logger.debug(f"Outlier values: {outlier_values}")
            
            return cleaned_data, cleaned_times
            
        except Exception as e:
            app_logger.error(f"Error removing outliers: {str(e)}")
            return data, times  # Return original data if cleaning fails

def calculate_cleaned_integrals(self):
        """
        Calculate integrals for hyperpolarization and depolarization segments
        after removing extreme outliers, with proper formatting for history.
        """
        try:
            # Remove extreme outliers from both segments
            hyperpol_clean, hyperpol_times_clean = self.remove_extreme_outliers(
                self.modified_hyperpol, 
                self.modified_hyperpol_times
            )
            
            depol_clean, depol_times_clean = self.remove_extreme_outliers(
                self.modified_depol,
                self.modified_depol_times
            )
            
            if hyperpol_clean is None or depol_clean is None:
                return None

            # Calculate time steps in milliseconds
            hyperpol_dt = np.diff(hyperpol_times_clean) * 1000  # Convert to ms
            depol_dt = np.diff(depol_times_clean) * 1000        # Convert to ms

            # Calculate absolute integrals using trapezoidal rule
            hyperpol_integral = np.abs(np.trapz(hyperpol_clean, 
                                              x=hyperpol_times_clean * 1000))  # time in ms
            depol_integral = np.abs(np.trapz(depol_clean,
                                           x=depol_times_clean * 1000))        # time in ms

            # Calculate capacitance
            voltage_diff = abs(self.params.get('V2', 0) - self.params.get('V0', 0))
            if voltage_diff > 0:
                capacitance = abs(hyperpol_integral - depol_integral) / voltage_diff
            else:
                capacitance = 0

            # Format values for history
            return {
                'integral_value': f"{(hyperpol_integral + depol_integral) / 2:.2f} pC",
                'hyperpol_area': f"{hyperpol_integral:.2f} pC",
                'depol_area': f"{depol_integral:.2f} pC",
                'capacitance_nF': f"{capacitance:.2f} nF",
                'purple_integral_value': (
                    f"Hyperpol={hyperpol_integral:.2f} pC, "
                    f"Depol={depol_integral:.2f} pC"
                )
            }

        except Exception as e:
            app_logger.error(f"Error calculating cleaned integrals: {str(e)}")
            return None

def integrate_curves_separately(self, ranges, method="direct", linreg_params=None):
        """
        Integrate hyperpolarization and depolarization curves with separate ranges.
        
        Args:
            ranges: dict with 'hyperpol' and 'depol' ranges, each containing 'start' and 'end'
            method: 'direct' or 'linreg'
            linreg_params: dict with linear regression parameters
            
        Returns:
            dict: Results containing both integrals and capacitance
        """
        if (not hasattr(self, 'modified_hyperpol') or 
            not hasattr(self, 'modified_depol') or
            self.modified_hyperpol is None or 
            self.modified_depol is None):
            raise ValueError("No purple curves available. Run analysis first.")

        # Get hyperpolarization range
        hyperpol_range = ranges.get('hyperpol', {'start': 0, 'end': 200})
        hyperpol_start = hyperpol_range['start']
        hyperpol_end = hyperpol_range['end']

        # Get depolarization range
        depol_range = ranges.get('depol', {'start': 0, 'end': 200})
        depol_start = depol_range['start']
        depol_end = depol_range['end']

        # Calculate integrals
        hyperpol_integral = self.integrate_segment(
            self.modified_hyperpol,
            self.modified_hyperpol_times,
            hyperpol_start,
            hyperpol_end,
            method,
            linreg_params,
            self.d3_factor if hasattr(self, 'd3_factor') else 1.0
        )

        depol_integral = self.integrate_segment(
            self.modified_depol,
            self.modified_depol_times,
            depol_start,
            depol_end,
            method,
            linreg_params,
            self.d3_factor if hasattr(self, 'd3_factor') else 1.0
        )

        # Calculate linear capacitance
        voltage_diff = abs(self.params['V2'] - self.params['V0'])
        if voltage_diff > 0:
            capacitance = abs(hyperpol_integral - depol_integral) / voltage_diff
        else:
            capacitance = 0

        return {
            'method': method,
            'hyperpol_range': f"{hyperpol_start}..{hyperpol_end}",
            'depol_range': f"{depol_start}..{depol_end}",
            'hyperpol_area': f"{hyperpol_integral:.6f}",
            'depol_area': f"{depol_integral:.6f}",
            'capacitance_nF': f"{capacitance:.6f}",
            'purple_integral_value': (
                f"Hyperpol={hyperpol_integral:.6f} pC, "
                f"Depol={depol_integral:.6f} pC, "
                f"Cap={capacitance:.6f} nF"
            )
        }
    
def integrate_curves_separately(self, ranges, method="direct", linreg_params=None):
    """
    Integrate hyperpolarization and depolarization curves with separate ranges.
    
    Args:
        ranges: dict with 'hyperpol' and 'depol' ranges, each containing 'start' and 'end'
        method: 'direct' or 'linreg'
        linreg_params: dict with linear regression parameters
        
    Returns:
        dict: Results containing both integrals and capacitance
    """
    if (not hasattr(self, 'modified_hyperpol') or 
        not hasattr(self, 'modified_depol') or
        self.modified_hyperpol is None or 
        self.modified_depol is None):
        raise ValueError("No purple curves available. Run analysis first.")

    # Get hyperpolarization range
    hyperpol_range = ranges.get('hyperpol', {'start': 0, 'end': 200})
    hyperpol_start = hyperpol_range['start']
    hyperpol_end = hyperpol_range['end']

    # Get depolarization range
    depol_range = ranges.get('depol', {'start': 0, 'end': 200})
    depol_start = depol_range['start']
    depol_end = depol_range['end']

    # Calculate integrals
    hyperpol_integral = self.integrate_segment(
        self.modified_hyperpol,
        self.modified_hyperpol_times,
        hyperpol_start,
        hyperpol_end,
        method,
        linreg_params,
        self.d3_factor if hasattr(self, 'd3_factor') else 1.0
    )

    depol_integral = self.integrate_segment(
        self.modified_depol,
        self.modified_depol_times,
        depol_start,
        depol_end,
        method,
        linreg_params,
        self.d3_factor if hasattr(self, 'd3_factor') else 1.0
    )

    # Calculate linear capacitance
    voltage_diff = abs(self.params['V2'] - self.params['V0'])
    if voltage_diff > 0:
        capacitance = abs(hyperpol_integral - depol_integral) / voltage_diff
    else:
        capacitance = 0

    return {
        'method': method,
        'hyperpol_range': f"{hyperpol_start}..{hyperpol_end}",
        'depol_range': f"{depol_start}..{depol_end}",
        'hyperpol_area': f"{hyperpol_integral:.6f}",
        'depol_area': f"{depol_integral:.6f}",
        'capacitance_nF': f"{capacitance:.6f}",
        'purple_integral_value': (
            f"Hyperpol={hyperpol_integral:.6f} pC, "
            f"Depol={depol_integral:.6f} pC, "
            f"Cap={capacitance:.6f} nF"
        )
    }





