# data_export_for_chamo.py
# Add this file to your Python Signal Analyzer project
# This exports denoised red curve data for the VB.NET CHAMO app

import numpy as np
import os
from datetime import datetime
import logging

app_logger = logging.getLogger(__name__)

def export_denoised_data_for_chamo(processor, output_dir="chamo_data", filename=None):
    """
    Export denoised red curve data in format compatible with CHAMO VB.NET app.
    
    Args:
        processor: Your ActionPotentialProcessor instance
        output_dir: Directory to save the exported data
        filename: Custom filename (optional)
    
    Returns:
        str: Path to the exported file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"denoised_data_{timestamp}.txt"
        
        filepath = os.path.join(output_dir, filename)
        
        # Get the denoised data (red curve)
        # Adjust these based on your processor's attribute names
        if hasattr(processor, 'orange_curve') and processor.orange_curve is not None:
            # Using orange_curve as the main denoised data
            denoised_data = processor.orange_curve
            time_data = processor.orange_curve_times
        elif hasattr(processor, 'processed_data') and processor.processed_data is not None:
            # Fallback to processed_data
            denoised_data = processor.processed_data
            time_data = getattr(processor, 'time_data', None)
        else:
            raise ValueError("No denoised data available in processor")
        
        # Ensure we have time data
        if time_data is None:
            # Generate time data if not available
            sampling_rate = getattr(processor, 'sampling_rate', 20000)  # Default 20kHz
            time_data = np.arange(len(denoised_data)) / sampling_rate
            app_logger.warning("Time data not found, generated based on sampling rate")
        
        # Write data in CHAMO-compatible format
        with open(filepath, 'w') as f:
            # Write header with metadata
            f.write(f"# CHAMO Data Export from Python Signal Analyzer\n")
            f.write(f"# Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Number of Points: {len(denoised_data)}\n")
            f.write(f"# Sampling Rate: {getattr(processor, 'sampling_rate', 'Unknown')} Hz\n")
            f.write(f"# Time Unit: seconds\n")
            f.write(f"# Current Unit: pA (picoamps)\n")
            f.write("# Format: Index, Time(s), Current(pA)\n")
            f.write("#\n")
            
            # Write the actual data
            for i, (time_val, current_val) in enumerate(zip(time_data, denoised_data)):
                f.write(f"{i+1}\t{time_val:.6f}\t{current_val:.6f}\n")
        
        app_logger.info(f"Exported {len(denoised_data)} data points to {filepath}")
        return filepath
        
    except Exception as e:
        app_logger.error(f"Error exporting data for CHAMO: {str(e)}")
        raise


def export_denoised_data_chamo_binary(processor, output_dir="chamo_data", filename=None):
    """
    Export denoised data in CHAMO's native array format (alternative method).
    Creates a file that matches CHAMO's OCur array structure.
    
    Args:
        processor: Your ActionPotentialProcessor instance
        output_dir: Directory to save the exported data
        filename: Custom filename (optional)
    
    Returns:
        str: Path to the exported file
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chamo_array_{timestamp}.dat"
        
        filepath = os.path.join(output_dir, filename)
        
        # Get denoised data
        if hasattr(processor, 'orange_curve') and processor.orange_curve is not None:
            denoised_data = processor.orange_curve
        elif hasattr(processor, 'processed_data') and processor.processed_data is not None:
            denoised_data = processor.processed_data
        else:
            raise ValueError("No denoised data available in processor")
        
        # Convert to format matching CHAMO's OCur array
        # OCur(0) = number of points, OCur(1) to OCur(n) = data
        chamo_array = np.zeros(len(denoised_data) + 1, dtype=np.float32)
        chamo_array[0] = len(denoised_data)  # Number of points
        chamo_array[1:] = denoised_data.astype(np.float32)  # Data points
        
        # Save as binary file
        chamo_array.tofile(filepath)
        
        # Also create a text info file
        info_file = filepath.replace('.dat', '_info.txt')
        with open(info_file, 'w') as f:
            f.write(f"CHAMO Binary Data File Information\n")
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data points: {len(denoised_data)}\n")
            f.write(f"Array size: {len(chamo_array)}\n")
            f.write(f"Data type: float32\n")
            f.write(f"Format: [count, data1, data2, ..., dataN]\n")
        
        app_logger.info(f"Exported binary data to {filepath}")
        return filepath
        
    except Exception as e:
        app_logger.error(f"Error exporting binary data for CHAMO: {str(e)}")
        raise


def export_with_parameters(processor, output_dir="chamo_data", include_metadata=True):
    """
    Export denoised data with processing parameters for CHAMO.
    
    Args:
        processor: Your ActionPotentialProcessor instance
        output_dir: Directory to save files
        include_metadata: Whether to include processing metadata
    
    Returns:
        dict: Paths to exported files
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        exported_files = {}
        
        # Export main data
        data_file = export_denoised_data_for_chamo(
            processor, 
            output_dir, 
            f"denoised_for_chamo_{timestamp}.txt"
        )
        exported_files['data'] = data_file
        
        # Export parameters if available
        if include_metadata and hasattr(processor, '__dict__'):
            param_file = os.path.join(output_dir, f"parameters_{timestamp}.txt")
            with open(param_file, 'w') as f:
                f.write("# Processing Parameters for CHAMO\n")
                f.write(f"# Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("#\n")
                
                # Extract relevant parameters
                for attr in dir(processor):
                    if not attr.startswith('_') and not callable(getattr(processor, attr)):
                        try:
                            value = getattr(processor, attr)
                            if isinstance(value, (int, float, str, bool)):
                                f.write(f"{attr}: {value}\n")
                        except:
                            continue
            
            exported_files['parameters'] = param_file
        
        app_logger.info(f"Complete export finished: {exported_files}")
        return exported_files
        
    except Exception as e:
        app_logger.error(f"Error in complete export: {str(e)}")
        raise