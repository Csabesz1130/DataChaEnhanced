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

    def calculate_normalized_curve(self):
        """
        Calculate normalized curve showing conductance changes.
        G = |I/ΔV| where ΔV is the voltage step (±20mV)
        """
        try:
            if self.orange_curve is None:
                return None, None

            # Define segments and their types
            segments = [
                {"start": 35, "end": 234, "is_hyperpol": True},   # 35-234
                {"start": 235, "end": 434, "is_hyperpol": False}, # 235-434
                {"start": 435, "end": 633, "is_hyperpol": True},  # 435-633
                {"start": 635, "end": 834, "is_hyperpol": False}  # 634-834
            ]
            
            normalized_points = []
            normalized_times = []
            
            for segment in segments:
                start_idx = segment["start"]
                end_idx = segment["end"]
                is_hyperpol = segment["is_hyperpol"]
                
                if end_idx >= len(self.orange_curve):
                    app_logger.warning(f"Not enough points in orange curve. Length: {len(self.orange_curve)}")
                    continue

                # Extract points for this segment
                selected_points = self.orange_curve[start_idx:end_idx]
                selected_times = self.orange_curve_times[start_idx:end_idx]
                
                # Calculate voltage step (always 20mV magnitude)
                voltage_step = -20.0 if is_hyperpol else 20.0  # mV
                
                # Calculate conductance: |I/ΔV|
                # The absolute value ensures conductance is always positive
                #nincs abs()
                segment_normalized = selected_points / voltage_step
                
                # Add to total arrays
                normalized_points.extend(segment_normalized)
                normalized_times.extend(selected_times)
                
                app_logger.debug(f"Processed segment {start_idx+1}-{end_idx}: "
                            f"{'hyperpolarization' if is_hyperpol else 'depolarization'}")
                app_logger.debug(f"Voltage step: {voltage_step} mV")
                app_logger.debug(f"Current range: [{np.min(selected_points):.2f}, "
                            f"{np.max(selected_points):.2f}] pA")
                app_logger.debug(f"Conductance range: [{np.min(segment_normalized):.2f}, "
                            f"{np.max(segment_normalized):.2f}] nS")
            
            self.normalized_curve = np.array(normalized_points)
            self.normalized_curve_times = np.array(normalized_times)
            
            app_logger.info("Conductance values calculated for all segments")
            return np.array(normalized_points), np.array(normalized_times)
            
        except Exception as e:
            app_logger.error(f"Error calculating normalized curve: {str(e)}")
            return None, None

    # Add these methods to src/analysis/action_potential.py in the ActionPotentialProcessor class

    def calculate_alternative_integral(self):
        """
        Calculate integral using averaged normalized curve method.
        """
        try:
            # First ensure we have the normalized curves
            if self.normalized_curve is None:
                self.normalized_curve, self.normalized_curve_times = self.calculate_normalized_curve()
                
            if self.normalized_curve is None:
                return {
                    'integral_value': 'No normalized curve available',
                    'capacitance_uF_cm2': '0.0000 µF/cm²'
                }

            # Calculate the average of the normalized curves
            avg_norm = np.mean(self.normalized_curve)
            
            # Calculate time step
            dt = np.mean(np.diff(self.normalized_curve_times))
            
            # Calculate integral of averaged normalized curve
            integral = np.trapz(self.normalized_curve, dx=dt)
            
            # Multiply by voltage difference
            voltage_diff = abs(self.params['V2'] - self.params['V0'])  # mV
            total_integral = integral * voltage_diff
            
            # Calculate capacitance (similar to original method)
            total_cap_F = abs(total_integral * 1e-12)  # Convert to Farads
            total_cap_uF = total_cap_F * 1e6  # Convert to µF
            
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

    def process_signal(self, use_alternative_method=False):
        """
        Process the signal and store all results.
        Added parameter use_alternative_method to choose integration method.
        """
        try:
            self.baseline_correction_initial()
            self.advanced_baseline_normalization()
            self.generate_orange_curve()
            self.normalized_curve, self.normalized_curve_times = self.calculate_normalized_curve()
            self.average_curve, self.average_curve_times = self.calculate_segment_average()
            
            # Apply average to peaks and store results
            if self.average_curve is not None:
                (self.modified_hyperpol, self.modified_hyperpol_times,
                self.modified_depol, self.modified_depol_times) = self.apply_average_to_peaks()
                
            self.find_cycles()
            
            # Choose integration method
            if use_alternative_method:
                results = self.calculate_alternative_integral()
            else:
                results = self.calculate_integral()
            
            if not results:
                return None, None, None, None, None, {
                    'integral_value': 'No analysis performed',
                    'capacitance_uF_cm2': 'No analysis performed',
                    'cycle_indices': []
                }
                
            return (self.processed_data, self.orange_curve, self.orange_curve_times,
                    self.normalized_curve, self.normalized_curve_times, results)
                    
        except Exception as e:
            app_logger.error(f"Error in process_signal: {str(e)}")
            return None, None, None, None, None, {
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
        Advanced normalization to align segments and ensure both hyperpolarization 
        and depolarization events are consistently normalized.
        """
        try:
            sampling_rate = 1.0 / np.mean(np.diff(self.time_data))
            threshold_slope = 3 * np.std(np.diff(self.processed_data))
            
            # Start segment analysis
            segment_start = 0
            segments = []
            
            # Find segments using slope changes in both directions
            for i in range(1, len(self.processed_data)):
                slope = abs(self.processed_data[i] - self.processed_data[i-1])
                if slope > threshold_slope:
                    # Verify it's a real transition by checking surrounding points
                    if i + 1 < len(self.processed_data):
                        next_slope = abs(self.processed_data[i+1] - self.processed_data[i])
                        if next_slope > threshold_slope:
                            # End current segment if long enough
                            if i - segment_start > 50:  # Minimum segment length
                                segments.append((segment_start, i))
                            segment_start = i + 1
            
            # Add final segment if exists
            if segment_start < len(self.processed_data) - 50:
                segments.append((segment_start, len(self.processed_data)))
            
            # Process each segment
            baseline_window = 50  # Points to use for baseline calculation
            for start, end in segments:
                # Find stable regions before and after rapid changes
                pre_region = self.processed_data[max(start, start):min(start + baseline_window, end)]
                post_region = self.processed_data[max(start, end - baseline_window):end]
                
                if len(pre_region) > 0 and len(post_region) > 0:
                    # Calculate baseline from both pre and post regions
                    pre_baseline = np.median(pre_region)
                    post_baseline = np.median(post_region)
                    
                    # Use the more stable baseline (smaller std)
                    if np.std(pre_region) < np.std(post_region):
                        baseline = pre_baseline
                    else:
                        baseline = post_baseline
                    
                    # Subtract baseline from segment
                    self.processed_data[start:end] -= baseline
                    
                    # Ensure smooth transitions between segments
                    if start > 0:
                        # Blend over 10 points
                        blend_points = min(10, start)
                        weights = np.linspace(0, 1, blend_points)
                        self.processed_data[start-blend_points:start] = (
                            weights * self.processed_data[start] +
                            (1 - weights) * self.processed_data[start-blend_points]
                        )
            
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
        """
        Identify and extract consistent hyperpolarization cycles.
        Ensures similar peak timing and curve shapes across cycles.
        """
        try:
            sampling_rate = 1.0 / np.mean(np.diff(self.time_data))
            t1_samps = int(self.params['t1'] * sampling_rate / 1000)
            
            # Find significant negative peaks with more stringent criteria
            neg_peaks, properties = signal.find_peaks(-self.processed_data,
                                                    prominence=np.std(self.processed_data) * 1.5,  # Increased prominence
                                                    distance=t1_samps,
                                                    width=(t1_samps//4, t1_samps))  # Add width constraints
            
            # Filter peaks based on minimum depth
            min_depth = np.max(np.abs(self.processed_data)) * 0.4  # 40% of max amplitude
            valid_peaks = []
            for peak in neg_peaks:
                if abs(self.processed_data[peak]) > min_depth:
                    valid_peaks.append(peak)
            
            # Extract cycles with consistent window sizes
            window_before = t1_samps // 2  # Fixed window before peak
            window_after = int(t1_samps * 1.5)   # Fixed window after peak
            
            self.cycles = []
            self.cycle_times = []
            self.cycle_indices = []
            
            for i, peak in enumerate(valid_peaks[:self.params['n_cycles']]):
                # Define cycle boundaries
                start = max(0, peak - window_before)
                end = min(len(self.processed_data), peak + window_after)
                
                # Only include cycles with full windows
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
                
            # Verify cycle similarity
            if len(self.cycles) >= 2:
                correlations = []
                reference_cycle = self.cycles[0]
                for cycle in self.cycles[1:]:
                    corr = np.corrcoef(reference_cycle, cycle)[0, 1]
                    correlations.append(corr)
                    app_logger.debug(f"Cycle correlation: {corr:.3f}")
                
                # Warn if cycles are too different
                if any(corr < 0.8 for corr in correlations):
                    app_logger.warning("Detected cycles show significant differences")
            
        except Exception as e:
            app_logger.error(f"Error in cycle detection: {str(e)}")
            raise

    def calculate_segment_average(self):
        """
        Calculate average curve from the 4 normalized segments.
        Each segment is resized to 200 points before averaging.
        """
        try:
            if self.orange_curve is None:
                return None, None

            # Define segments
            segments = [
                {"start": 35, "end": 234},    # First hyperpol
                {"start": 235, "end": 434},   # First depol
                {"start": 435, "end": 634},   # Second hyperpol
                {"start": 635, "end": 834}    # Second depol
            ]
            
            # Initialize arrays for resampled segments
            resampled_segments = []
            target_length = 200
            
            # Resample each segment to 200 points
            for segment in segments:
                start_idx = segment["start"]
                end_idx = segment["end"]
                
                # Extract segment
                segment_data = self.orange_curve[start_idx:end_idx]
                segment_times = self.orange_curve_times[start_idx:end_idx]
                
                # Resample to 200 points using interpolation
                original_points = np.linspace(0, 1, len(segment_data))
                new_points = np.linspace(0, 1, target_length)
                resampled_data = np.interp(new_points, original_points, segment_data)
                resampled_segments.append(resampled_data)
            
            # Convert to numpy array for easier operations
            segments_array = np.array(resampled_segments)
            
            # Calculate average curve
            #kibontani mi történik, akár for-ral leprogramozni
            average_curve = np.mean(segments_array, axis=0)
            
            # Generate corresponding time points (using time range of first segment)
            avg_times = np.linspace(
                self.orange_curve_times[segments[0]["start"]],
                self.orange_curve_times[segments[0]["end"]],
                target_length
            )
            
            self.average_curve = average_curve
            self.average_curve_times = avg_times
            
            app_logger.info("Calculated average curve from 4 segments")
            app_logger.debug(f"Average curve range: [{np.min(average_curve):.2f}, {np.max(average_curve):.2f}]")
            
            return average_curve, avg_times
            
        except Exception as e:
            app_logger.error(f"Error calculating segment average: {str(e)}")
            return None, None

    # Replace apply_average_to_peaks method in ActionPotentialProcessor class

    def apply_average_to_peaks(self):
        """
        Add averaged normalized curve to high hyperpolarization
        and subtract it from high depolarization.
        Uses exact indices for the high voltage segments.
        """
        try:
            if self.average_curve is None:
                self.average_curve, _ = self.calculate_segment_average()
                if self.average_curve is None:
                    return None, None, None, None
            
            # Define exact indices for segments
            depol_start = 834    # High depolarization
            depol_end = 1034
            hyperpol_start = 1034  # High hyperpolarization (adjusted to match)
            hyperpol_end = 1234
            
            # Check if we have enough points
            if len(self.orange_curve) <= hyperpol_end:
                app_logger.error(f"Orange curve too short ({len(self.orange_curve)} points) for peak segments")
                return None, None, None, None
            
            # Extract segments (ensure same length)
            segment_length = depol_end - depol_start  # Should be 200
            
            depol_data = self.orange_curve[depol_start:depol_end]
            depol_times = self.orange_curve_times[depol_start:depol_end]
            hyperpol_data = self.orange_curve[hyperpol_start:hyperpol_start + segment_length]
            hyperpol_times = self.orange_curve_times[hyperpol_start:hyperpol_start + segment_length]
            
            # Ensure average curve matches segment length
            if len(self.average_curve) != segment_length:
                original_points = np.linspace(0, 1, len(self.average_curve))
                new_points = np.linspace(0, 1, segment_length)
                average_curve = np.interp(new_points, original_points, self.average_curve)
            else:
                average_curve = self.average_curve
            
            # Calculate voltage step for scaling
            voltage_diff = abs(self.params['V2'] - self.params['V0'])  # Should be around 90mV
            scaled_average = average_curve * voltage_diff * 0.2  # Scale factor to match original peaks
            
            # Add/subtract scaled average curve
            hyperpol_modified = hyperpol_data + scaled_average
            depol_modified = depol_data - scaled_average
            
            app_logger.info(f"Applied average curve to peak segments")
            app_logger.debug(f"Peak locations - hyperpol: {hyperpol_start}, depol: {depol_start}")
            app_logger.debug(f"Modified hyperpol range: [{np.min(hyperpol_modified):.2f}, {np.max(hyperpol_modified):.2f}]")
            app_logger.debug(f"Modified depol range: [{np.min(depol_modified):.2f}, {np.max(depol_modified):.2f}]")
            
            return (hyperpol_modified, hyperpol_times,
                    depol_modified, depol_times)
                    
        except Exception as e:
            app_logger.error(f"Error applying average to peaks: {str(e)}")
            return None, None, None, None

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
            voltage_diff_in_V = (self.params['V1'] - self.params['V0'])  # mV to V, FELESLEGES - kiejtik egymást a mV és ms, ygx nem kellenek ezek az átváltások
            
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