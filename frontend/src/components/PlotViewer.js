import React, { useState, useCallback, useMemo } from 'react';
import Plot from 'react-plotly.js';
import Plotly from 'plotly.js';
import {
    Box,
    Paper,
    FormGroup,
    FormControlLabel,
    Checkbox,
    Button,
    ButtonGroup,
    TextField,
    Typography,
    IconButton,
    Tooltip,
    Menu,
    MenuItem,
} from '@mui/material';
import {
    ZoomIn,
    ZoomOut,
    FitScreen,
    Download,
    GridOn,
    GridOff,
} from '@mui/icons-material';

const PlotViewer = ({ data }) => {
    const [visibleCurves, setVisibleCurves] = useState({
        orange: true,
        normalized: true,
        average: true,
        hyperpol: true,
        depol: true,
    });

    const [showGrid, setShowGrid] = useState(true);
    const [customYLim, setCustomYLim] = useState({ enabled: false, min: '', max: '' });
    const [customXRange, setCustomXRange] = useState({ enabled: false, min: '', max: '' });
    const [plotRef, setPlotRef] = useState(null);
    const [exportMenuAnchor, setExportMenuAnchor] = useState(null);

    const handleCurveToggle = (curve) => {
        setVisibleCurves((prev) => ({
            ...prev,
            [curve]: !prev[curve],
        }));
    };

    const handleZoom = useCallback((action) => {
        if (!plotRef || !plotRef.el) return;

        const plotDiv = plotRef.el;
        const update = {};

        if (action === 'reset') {
            update['xaxis.autorange'] = true;
            update['yaxis.autorange'] = true;
        } else {
            // Get current ranges
            const xRange = plotDiv.layout?.xaxis?.range || plotDiv._fullLayout?.xaxis?.range;
            const yRange = plotDiv.layout?.yaxis?.range || plotDiv._fullLayout?.yaxis?.range;

            if (xRange && yRange && Array.isArray(xRange) && Array.isArray(yRange)) {
                const xCenter = (xRange[0] + xRange[1]) / 2;
                const yCenter = (yRange[0] + yRange[1]) / 2;
                
                if (action === 'in') {
                    // Zoom in by 20%
                    const xSpan = (xRange[1] - xRange[0]) * 0.8;
                    const ySpan = (yRange[1] - yRange[0]) * 0.8;
                    update['xaxis.range'] = [xCenter - xSpan / 2, xCenter + xSpan / 2];
                    update['yaxis.range'] = [yCenter - ySpan / 2, yCenter + ySpan / 2];
                } else if (action === 'out') {
                    // Zoom out by 25%
                    const xSpan = (xRange[1] - xRange[0]) * 1.25;
                    const ySpan = (yRange[1] - yRange[0]) * 1.25;
                    update['xaxis.range'] = [xCenter - xSpan / 2, xCenter + xSpan / 2];
                    update['yaxis.range'] = [yCenter - ySpan / 2, yCenter + ySpan / 2];
                }
            }
        }

        if (Object.keys(update).length > 0) {
            Plotly.relayout(plotDiv, update);
        }
    }, [plotRef]);

    const handleExport = useCallback((format) => {
        if (!plotRef) return;

        const plotDiv = plotRef.el;
        if (!plotDiv) return;

        const gd = plotDiv;
        const filename = `signal_analysis_${new Date().toISOString().split('T')[0]}.${format}`;

        if (format === 'png') {
            Plotly.downloadImage(gd, { format: 'png', width: 1200, height: 800, filename });
        } else if (format === 'svg') {
            Plotly.downloadImage(gd, { format: 'svg', width: 1200, height: 800, filename });
        } else if (format === 'pdf') {
            Plotly.downloadImage(gd, { format: 'pdf', width: 1200, height: 800, filename });
        }

        setExportMenuAnchor(null);
    }, [plotRef]);

    // Create traces for all available curves with visibility control
    const traces = useMemo(() => {
        const traceList = [];

        // Orange curve
        if (data?.orange_curve && data?.orange_curve_times && visibleCurves.orange) {
            traceList.push({
                x: data.orange_curve_times,
                y: data.orange_curve,
                name: 'Orange Curve',
                type: 'scatter',
                mode: 'lines',
                line: { color: 'orange', width: 2 },
                visible: visibleCurves.orange ? true : 'legendonly',
            });
        }

        // Normalized curve
        if (data?.normalized_curve && data?.normalized_curve_times && visibleCurves.normalized) {
            traceList.push({
                x: data.normalized_curve_times,
                y: data.normalized_curve,
                name: 'Normalized Curve',
                type: 'scatter',
                mode: 'lines',
                line: { color: 'blue', width: 2 },
                visible: visibleCurves.normalized ? true : 'legendonly',
            });
        }

        // Average curve
        if (data?.average_curve && data?.average_curve_times && visibleCurves.average) {
            traceList.push({
                x: data.average_curve_times,
                y: data.average_curve,
                name: 'Average Curve',
                type: 'scatter',
                mode: 'lines',
                line: { color: 'green', width: 2 },
                visible: visibleCurves.average ? true : 'legendonly',
            });
        }

        // Modified hyperpolarization
        if (data?.modified_hyperpol && data?.modified_hyperpol_times && visibleCurves.hyperpol) {
            traceList.push({
                x: data.modified_hyperpol_times,
                y: data.modified_hyperpol,
                name: 'Modified Hyperpol',
                type: 'scatter',
                mode: 'lines',
                line: { color: 'purple', width: 2, dash: 'dash' },
                visible: visibleCurves.hyperpol ? true : 'legendonly',
            });
        }

        // Modified depolarization
        if (data?.modified_depol && data?.modified_depol_times && visibleCurves.depol) {
            traceList.push({
                x: data.modified_depol_times,
                y: data.modified_depol,
                name: 'Modified Depol',
                type: 'scatter',
                mode: 'lines',
                line: { color: 'red', width: 2, dash: 'dash' },
                visible: visibleCurves.depol ? true : 'legendonly',
            });
        }

        return traceList;
    }, [data, visibleCurves]);

    const layout = useMemo(() => {
        const baseLayout = {
            title: 'Signal Analysis',
            xaxis: {
                title: 'Time (s)',
                showgrid: showGrid,
                zeroline: true,
                ...(customXRange.enabled && customXRange.min && customXRange.max
                    ? { range: [parseFloat(customXRange.min), parseFloat(customXRange.max)] }
                    : {}),
            },
            yaxis: {
                title: 'Current (pA)',
                showgrid: showGrid,
                zeroline: true,
                ...(customYLim.enabled && customYLim.min !== '' && customYLim.max !== ''
                    ? { range: [parseFloat(customYLim.min), parseFloat(customYLim.max)] }
                    : {}),
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

        return baseLayout;
    }, [showGrid, customYLim, customXRange]);

    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    };

    if (!data) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
                <Typography color="text.secondary">No data to display</Typography>
            </Box>
        );
    }

    return (
        <Box>
            {/* Controls Panel */}
            <Paper sx={{ p: 2, mb: 2 }}>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, alignItems: 'center' }}>
                    {/* Curve Visibility Toggles */}
                    <FormGroup row>
                        <Typography variant="body2" sx={{ mr: 1, alignSelf: 'center' }}>
                            Curves:
                        </Typography>
                        {data?.orange_curve && (
                            <FormControlLabel
                                control={
                                    <Checkbox
                                        checked={visibleCurves.orange}
                                        onChange={() => handleCurveToggle('orange')}
                                        size="small"
                                    />
                                }
                                label="Orange"
                            />
                        )}
                        {data?.normalized_curve && (
                            <FormControlLabel
                                control={
                                    <Checkbox
                                        checked={visibleCurves.normalized}
                                        onChange={() => handleCurveToggle('normalized')}
                                        size="small"
                                    />
                                }
                                label="Normalized"
                            />
                        )}
                        {data?.average_curve && (
                            <FormControlLabel
                                control={
                                    <Checkbox
                                        checked={visibleCurves.average}
                                        onChange={() => handleCurveToggle('average')}
                                        size="small"
                                    />
                                }
                                label="Average"
                            />
                        )}
                        {data?.modified_hyperpol && (
                            <FormControlLabel
                                control={
                                    <Checkbox
                                        checked={visibleCurves.hyperpol}
                                        onChange={() => handleCurveToggle('hyperpol')}
                                        size="small"
                                    />
                                }
                                label="Hyperpol"
                            />
                        )}
                        {data?.modified_depol && (
                            <FormControlLabel
                                control={
                                    <Checkbox
                                        checked={visibleCurves.depol}
                                        onChange={() => handleCurveToggle('depol')}
                                        size="small"
                                    />
                                }
                                label="Depol"
                            />
                        )}
                    </FormGroup>

                    {/* Zoom Controls */}
                    <ButtonGroup size="small" variant="outlined">
                        <Tooltip title="Zoom In">
                            <IconButton onClick={() => handleZoom('in')} size="small">
                                <ZoomIn />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Zoom Out">
                            <IconButton onClick={() => handleZoom('out')} size="small">
                                <ZoomOut />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Reset Zoom">
                            <IconButton onClick={() => handleZoom('reset')} size="small">
                                <FitScreen />
                            </IconButton>
                        </Tooltip>
                    </ButtonGroup>

                    {/* Grid Toggle */}
                    <Tooltip title={showGrid ? 'Hide Grid' : 'Show Grid'}>
                        <IconButton
                            onClick={() => setShowGrid(!showGrid)}
                            size="small"
                            color={showGrid ? 'primary' : 'default'}
                        >
                            {showGrid ? <GridOn /> : <GridOff />}
                        </IconButton>
                    </Tooltip>

                    {/* Export Menu */}
                    <Button
                        size="small"
                        variant="outlined"
                        startIcon={<Download />}
                        onClick={(e) => setExportMenuAnchor(e.currentTarget)}
                    >
                        Export
                    </Button>
                    <Menu
                        anchorEl={exportMenuAnchor}
                        open={Boolean(exportMenuAnchor)}
                        onClose={() => setExportMenuAnchor(null)}
                    >
                        <MenuItem onClick={() => handleExport('png')}>Export as PNG</MenuItem>
                        <MenuItem onClick={() => handleExport('svg')}>Export as SVG</MenuItem>
                        <MenuItem onClick={() => handleExport('pdf')}>Export as PDF</MenuItem>
                    </Menu>
                </Box>

                {/* Axis Controls */}
                <Box sx={{ display: 'flex', gap: 2, mt: 2, flexWrap: 'wrap' }}>
                    <FormControlLabel
                        control={
                            <Checkbox
                                checked={customYLim.enabled}
                                onChange={(e) =>
                                    setCustomYLim({ ...customYLim, enabled: e.target.checked })
                                }
                                size="small"
                            />
                        }
                        label="Custom Y Limits:"
                    />
                    {customYLim.enabled && (
                        <>
                            <TextField
                                size="small"
                                label="Y Min"
                                type="number"
                                value={customYLim.min}
                                onChange={(e) =>
                                    setCustomYLim({ ...customYLim, min: e.target.value })
                                }
                                sx={{ width: 100 }}
                            />
                            <TextField
                                size="small"
                                label="Y Max"
                                type="number"
                                value={customYLim.max}
                                onChange={(e) =>
                                    setCustomYLim({ ...customYLim, max: e.target.value })
                                }
                                sx={{ width: 100 }}
                            />
                        </>
                    )}

                    <FormControlLabel
                        control={
                            <Checkbox
                                checked={customXRange.enabled}
                                onChange={(e) =>
                                    setCustomXRange({ ...customXRange, enabled: e.target.checked })
                                }
                                size="small"
                            />
                        }
                        label="Custom X Range:"
                    />
                    {customXRange.enabled && (
                        <>
                            <TextField
                                size="small"
                                label="X Min"
                                type="number"
                                value={customXRange.min}
                                onChange={(e) =>
                                    setCustomXRange({ ...customXRange, min: e.target.value })
                                }
                                sx={{ width: 100 }}
                            />
                            <TextField
                                size="small"
                                label="X Max"
                                type="number"
                                value={customXRange.max}
                                onChange={(e) =>
                                    setCustomXRange({ ...customXRange, max: e.target.value })
                                }
                                sx={{ width: 100 }}
                            />
                        </>
                    )}
                </Box>
            </Paper>

            {/* Plot */}
            <Box sx={{ width: '100%', height: '600px' }}>
                <Plot
                    data={traces}
                    layout={layout}
                    config={config}
                    style={{ width: '100%', height: '100%' }}
                    useResizeHandler={true}
                    onInitialized={(figure, graphDiv) => setPlotRef({ el: graphDiv, figure })}
                    onUpdate={(figure, graphDiv) => setPlotRef({ el: graphDiv, figure })}
                />
            </Box>
        </Box>
    );
};

export default PlotViewer;

