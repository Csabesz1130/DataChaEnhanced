import React from 'react';
import Plot from 'react-plotly.js';
import { Box } from '@mui/material';

const PlotViewer = ({ data }) => {
    if (!data) return null;

    // Create traces for all available curves
    const traces = [];

    // Orange curve
    if (data.orange_curve && data.orange_curve_times) {
        traces.push({
            x: data.orange_curve_times,
            y: data.orange_curve,
            name: 'Orange Curve',
            type: 'scatter',
            mode: 'lines',
            line: { color: 'orange', width: 2 },
        });
    }

    // Normalized curve
    if (data.normalized_curve && data.normalized_curve_times) {
        traces.push({
            x: data.normalized_curve_times,
            y: data.normalized_curve,
            name: 'Normalized Curve',
            type: 'scatter',
            mode: 'lines',
            line: { color: 'blue', width: 2 },
        });
    }

    // Average curve
    if (data.average_curve && data.average_curve_times) {
        traces.push({
            x: data.average_curve_times,
            y: data.average_curve,
            name: 'Average Curve',
            type: 'scatter',
            mode: 'lines',
            line: { color: 'green', width: 2 },
        });
    }

    // Modified hyperpolarization
    if (data.modified_hyperpol && data.modified_hyperpol_times) {
        traces.push({
            x: data.modified_hyperpol_times,
            y: data.modified_hyperpol,
            name: 'Modified Hyperpol',
            type: 'scatter',
            mode: 'lines',
            line: { color: 'purple', width: 2, dash: 'dash' },
        });
    }

    // Modified depolarization
    if (data.modified_depol && data.modified_depol_times) {
        traces.push({
            x: data.modified_depol_times,
            y: data.modified_depol,
            name: 'Modified Depol',
            type: 'scatter',
            mode: 'lines',
            line: { color: 'red', width: 2, dash: 'dash' },
        });
    }

    const layout = {
        title: 'Signal Analysis',
        xaxis: {
            title: 'Time (s)',
            showgrid: true,
            zeroline: true,
        },
        yaxis: {
            title: 'Current (pA)',
            showgrid: true,
            zeroline: true,
        },
        showlegend: true,
        legend: {
            x: 1,
            xanchor: 'right',
            y: 1,
        },
        hovermode: 'closest',
        autosize: true,
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    };

    return (
        <Box sx={{ width: '100%', height: '600px' }}>
            <Plot
                data={traces}
                layout={layout}
                config={config}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler={true}
            />
        </Box>
    );
};

export default PlotViewer;

