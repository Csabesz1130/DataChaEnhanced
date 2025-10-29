"""
Convert matplotlib plots to JSON for frontend plotting
"""

import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, List, Tuple


def matplotlib_to_base64(fig) -> str:
    """
    Convert matplotlib figure to base64 encoded PNG
    
    Args:
        fig: Matplotlib figure object
    
    Returns:
        str: Base64 encoded PNG image
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{image_base64}"


def convert_matplotlib_to_json(fig) -> Dict:
    """
    Convert matplotlib figure to JSON-serializable format
    
    Args:
        fig: Matplotlib figure object
    
    Returns:
        dict: JSON-serializable plot data
    """
    # Get all axes from figure
    axes_data = []
    
    for ax in fig.get_axes():
        ax_data = {
            'xlabel': ax.get_xlabel(),
            'ylabel': ax.get_ylabel(),
            'title': ax.get_title(),
            'lines': []
        }
        
        # Extract line data
        for line in ax.get_lines():
            line_data = {
                'x': line.get_xdata().tolist(),
                'y': line.get_ydata().tolist(),
                'label': line.get_label(),
                'color': line.get_color(),
                'linestyle': line.get_linestyle(),
                'linewidth': line.get_linewidth()
            }
            ax_data['lines'].append(line_data)
        
        axes_data.append(ax_data)
    
    # Also include base64 image as fallback
    image_base64 = matplotlib_to_base64(fig)
    
    return {
        'axes': axes_data,
        'image': image_base64
    }


def numpy_to_plotly_trace(x: np.ndarray, y: np.ndarray, name: str = None, **kwargs) -> Dict:
    """
    Convert numpy arrays to Plotly trace format
    
    Args:
        x: X-axis data
        y: Y-axis data
        name: Trace name
        **kwargs: Additional Plotly trace parameters
    
    Returns:
        dict: Plotly trace object
    """
    trace = {
        'x': x.tolist() if isinstance(x, np.ndarray) else x,
        'y': y.tolist() if isinstance(y, np.ndarray) else y,
        'type': 'scatter',
        'mode': 'lines'
    }
    
    if name:
        trace['name'] = name
    
    # Add any additional parameters
    trace.update(kwargs)
    
    return trace


def create_plotly_layout(title: str = None, xlabel: str = None, ylabel: str = None, **kwargs) -> Dict:
    """
    Create Plotly layout object
    
    Args:
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        **kwargs: Additional layout parameters
    
    Returns:
        dict: Plotly layout object
    """
    layout = {
        'showlegend': True,
        'hovermode': 'closest'
    }
    
    if title:
        layout['title'] = title
    
    if xlabel:
        layout['xaxis'] = {'title': xlabel}
    
    if ylabel:
        layout['yaxis'] = {'title': ylabel}
    
    layout.update(kwargs)
    
    return layout


def analysis_results_to_plotly(results: Dict) -> Dict:
    """
    Convert analysis results to Plotly-compatible format
    
    Args:
        results: Analysis results dictionary from ActionPotentialProcessor
    
    Returns:
        dict: Plotly figure data (traces + layout)
    """
    traces = []
    
    # Orange curve
    if results.get('orange_curve') and results.get('orange_curve_times'):
        traces.append(numpy_to_plotly_trace(
            results['orange_curve_times'],
            results['orange_curve'],
            name='Orange Curve',
            line={'color': 'orange', 'width': 2}
        ))
    
    # Normalized curve
    if results.get('normalized_curve') and results.get('normalized_curve_times'):
        traces.append(numpy_to_plotly_trace(
            results['normalized_curve_times'],
            results['normalized_curve'],
            name='Normalized Curve',
            line={'color': 'blue', 'width': 2}
        ))
    
    # Average curve
    if results.get('average_curve') and results.get('average_curve_times'):
        traces.append(numpy_to_plotly_trace(
            results['average_curve_times'],
            results['average_curve'],
            name='Average Curve',
            line={'color': 'green', 'width': 2}
        ))
    
    # Modified hyperpolarization
    if results.get('modified_hyperpol') and results.get('modified_hyperpol_times'):
        traces.append(numpy_to_plotly_trace(
            results['modified_hyperpol_times'],
            results['modified_hyperpol'],
            name='Modified Hyperpol',
            line={'color': 'purple', 'width': 2, 'dash': 'dash'}
        ))
    
    # Modified depolarization
    if results.get('modified_depol') and results.get('modified_depol_times'):
        traces.append(numpy_to_plotly_trace(
            results['modified_depol_times'],
            results['modified_depol'],
            name='Modified Depol',
            line={'color': 'red', 'width': 2, 'dash': 'dash'}
        ))
    
    layout = create_plotly_layout(
        title='Signal Analysis Results',
        xlabel='Time (s)',
        ylabel='Current (pA)'
    )
    
    return {
        'data': traces,
        'layout': layout
    }

