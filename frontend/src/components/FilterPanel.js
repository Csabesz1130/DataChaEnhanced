import React, { useState } from 'react';
import {
    Box,
    Paper,
    Typography,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    TextField,
    Button,
    FormControlLabel,
    Switch,
    Grid,
    Alert,
    CircularProgress,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import {
    applySavgolFilter,
    applyButterworthFilter,
    applyWaveletFilter,
    applyCombinedFilter,
} from '../services/api';
import { getErrorMessage } from '../utils/errorHandler';

const FilterPanel = ({ data, timeData, onFiltered, loading: parentLoading }) => {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [filteredData, setFilteredData] = useState(null);
    const [metrics, setMetrics] = useState(null);

    // Savitzky-Golay filter state
    const [savgolEnabled, setSavgolEnabled] = useState(false);
    const [savgolWindow, setSavgolWindow] = useState(51);
    const [savgolOrder, setSavgolOrder] = useState(3);

    // Butterworth filter state
    const [butterEnabled, setButterEnabled] = useState(false);
    const [butterCutoff, setButterCutoff] = useState(100);
    const [butterOrder, setButterOrder] = useState(5);
    const [butterFs, setButterFs] = useState(1000);

    // Wavelet filter state
    const [waveletEnabled, setWaveletEnabled] = useState(false);
    const [waveletType, setWaveletType] = useState('db4');
    const [waveletLevel, setWaveletLevel] = useState(4);
    const [waveletThresholdMode, setWaveletThresholdMode] = useState('soft');

    const handleApplyFilter = async (filterType) => {
        if (!data || data.length === 0) {
            setError('No data available to filter');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            let result;
            let filterMetrics;

            switch (filterType) {
                case 'savgol':
                    if (!savgolEnabled) {
                        setError('Please enable Savitzky-Golay filter first');
                        setLoading(false);
                        return;
                    }
                    result = await applySavgolFilter(data, savgolWindow, savgolOrder);
                    filterMetrics = result.metrics;
                    break;

                case 'butterworth':
                    if (!butterEnabled) {
                        setError('Please enable Butterworth filter first');
                        setLoading(false);
                        return;
                    }
                    result = await applyButterworthFilter(data, butterCutoff, butterFs, butterOrder);
                    filterMetrics = result.metrics;
                    break;

                case 'wavelet':
                    if (!waveletEnabled) {
                        setError('Please enable Wavelet filter first');
                        setLoading(false);
                        return;
                    }
                    result = await applyWaveletFilter(data, waveletType, waveletLevel, waveletThresholdMode);
                    filterMetrics = result.metrics;
                    break;

                case 'combined':
                    const combinedParams = {};
                    if (savgolEnabled) {
                        combinedParams.savgol_params = {
                            window_length: savgolWindow,
                            polyorder: savgolOrder,
                        };
                    }
                    if (butterEnabled) {
                        combinedParams.butter_params = {
                            cutoff: butterCutoff,
                            fs: butterFs,
                            order: butterOrder,
                        };
                    }
                    if (waveletEnabled) {
                        combinedParams.wavelet_params = {
                            wavelet: waveletType,
                            level: waveletLevel,
                            threshold_mode: waveletThresholdMode,
                        };
                    }

                    if (Object.keys(combinedParams).length === 0) {
                        setError('Please enable at least one filter');
                        setLoading(false);
                        return;
                    }

                    result = await applyCombinedFilter(data, combinedParams);
                    filterMetrics = result.metrics;
                    break;

                default:
                    setError('Unknown filter type');
                    setLoading(false);
                    return;
            }

            setFilteredData(result.filtered_data);
            setMetrics(filterMetrics);

            // Notify parent component
            if (onFiltered) {
                onFiltered(result.filtered_data, timeData, filterMetrics);
            }
        } catch (err) {
            const errorMessage = getErrorMessage(err);
            setError(errorMessage);
            console.error('Filter error:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleReset = () => {
        setFilteredData(null);
        setMetrics(null);
        setError(null);
        if (onFiltered) {
            onFiltered(null, null, null);
        }
    };

    return (
        <Box>
            <Typography variant="h6" gutterBottom>
                Signal Filtering
            </Typography>

            {error && (
                <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
                    {error}
                </Alert>
            )}

            {metrics && (
                <Alert severity="info" sx={{ mb: 2 }}>
                    <Typography variant="body2">
                        <strong>Filter Metrics:</strong>
                    </Typography>
                    <Typography variant="body2">
                        SNR Improvement: {metrics.snr_improvement?.toFixed(2) || 'N/A'}
                    </Typography>
                    <Typography variant="body2">
                        Smoothness: {metrics.smoothness?.toFixed(2) || 'N/A'}
                    </Typography>
                </Alert>
            )}

            {/* Savitzky-Golay Filter */}
            <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                        <FormControlLabel
                            control={
                                <Switch
                                    checked={savgolEnabled}
                                    onChange={(e) => setSavgolEnabled(e.target.checked)}
                                    onClick={(e) => e.stopPropagation()}
                                />
                            }
                            label="Savitzky-Golay Filter"
                        />
                    </Box>
                </AccordionSummary>
                <AccordionDetails>
                    <Grid container spacing={2}>
                        <Grid item xs={12}>
                            <Typography variant="caption" color="text.secondary">
                                Polynomial smoothing filter. Good for preserving signal shape while reducing noise.
                            </Typography>
                        </Grid>
                        <Grid item xs={6}>
                            <TextField
                                fullWidth
                                size="small"
                                label="Window Length"
                                type="number"
                                value={savgolWindow}
                                onChange={(e) => setSavgolWindow(parseInt(e.target.value) || 51)}
                                inputProps={{ min: 5, max: 101, step: 2 }}
                                helperText="Must be odd (5-101)"
                                disabled={!savgolEnabled}
                            />
                        </Grid>
                        <Grid item xs={6}>
                            <TextField
                                fullWidth
                                size="small"
                                label="Polynomial Order"
                                type="number"
                                value={savgolOrder}
                                onChange={(e) => setSavgolOrder(parseInt(e.target.value) || 3)}
                                inputProps={{ min: 2, max: 5 }}
                                disabled={!savgolEnabled}
                            />
                        </Grid>
                        <Grid item xs={12}>
                            <Button
                                variant="outlined"
                                startIcon={loading ? <CircularProgress size={16} /> : <PlayArrowIcon />}
                                onClick={() => handleApplyFilter('savgol')}
                                disabled={!savgolEnabled || loading || parentLoading}
                                fullWidth
                            >
                                Apply Savitzky-Golay Filter
                            </Button>
                        </Grid>
                    </Grid>
                </AccordionDetails>
            </Accordion>

            {/* Butterworth Filter */}
            <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                        <FormControlLabel
                            control={
                                <Switch
                                    checked={butterEnabled}
                                    onChange={(e) => setButterEnabled(e.target.checked)}
                                    onClick={(e) => e.stopPropagation()}
                                />
                            }
                            label="Butterworth Lowpass Filter"
                        />
                    </Box>
                </AccordionSummary>
                <AccordionDetails>
                    <Grid container spacing={2}>
                        <Grid item xs={12}>
                            <Typography variant="caption" color="text.secondary">
                                Lowpass filter for removing high-frequency noise.
                            </Typography>
                        </Grid>
                        <Grid item xs={6}>
                            <TextField
                                fullWidth
                                size="small"
                                label="Cutoff Frequency (Hz)"
                                type="number"
                                value={butterCutoff}
                                onChange={(e) => setButterCutoff(parseFloat(e.target.value) || 100)}
                                inputProps={{ min: 0.1, step: 0.1 }}
                                disabled={!butterEnabled}
                            />
                        </Grid>
                        <Grid item xs={6}>
                            <TextField
                                fullWidth
                                size="small"
                                label="Sampling Rate (Hz)"
                                type="number"
                                value={butterFs}
                                onChange={(e) => setButterFs(parseFloat(e.target.value) || 1000)}
                                inputProps={{ min: 1, step: 1 }}
                                disabled={!butterEnabled}
                            />
                        </Grid>
                        <Grid item xs={12}>
                            <TextField
                                fullWidth
                                size="small"
                                label="Filter Order"
                                type="number"
                                value={butterOrder}
                                onChange={(e) => setButterOrder(parseInt(e.target.value) || 5)}
                                inputProps={{ min: 1, max: 10 }}
                                disabled={!butterEnabled}
                            />
                        </Grid>
                        <Grid item xs={12}>
                            <Button
                                variant="outlined"
                                startIcon={loading ? <CircularProgress size={16} /> : <PlayArrowIcon />}
                                onClick={() => handleApplyFilter('butterworth')}
                                disabled={!butterEnabled || loading || parentLoading}
                                fullWidth
                            >
                                Apply Butterworth Filter
                            </Button>
                        </Grid>
                    </Grid>
                </AccordionDetails>
            </Accordion>

            {/* Wavelet Filter */}
            <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                        <FormControlLabel
                            control={
                                <Switch
                                    checked={waveletEnabled}
                                    onChange={(e) => setWaveletEnabled(e.target.checked)}
                                    onClick={(e) => e.stopPropagation()}
                                />
                            }
                            label="Wavelet Filter"
                        />
                    </Box>
                </AccordionSummary>
                <AccordionDetails>
                    <Grid container spacing={2}>
                        <Grid item xs={12}>
                            <Typography variant="caption" color="text.secondary">
                                Wavelet denoising filter. Good for removing noise while preserving sharp features.
                            </Typography>
                        </Grid>
                        <Grid item xs={6}>
                            <FormControl fullWidth size="small" disabled={!waveletEnabled}>
                                <InputLabel>Wavelet Type</InputLabel>
                                <Select
                                    value={waveletType}
                                    label="Wavelet Type"
                                    onChange={(e) => setWaveletType(e.target.value)}
                                >
                                    <MenuItem value="db4">Daubechies 4 (db4)</MenuItem>
                                    <MenuItem value="db8">Daubechies 8 (db8)</MenuItem>
                                    <MenuItem value="haar">Haar</MenuItem>
                                    <MenuItem value="coif2">Coiflet 2</MenuItem>
                                </Select>
                            </FormControl>
                        </Grid>
                        <Grid item xs={6}>
                            <TextField
                                fullWidth
                                size="small"
                                label="Decomposition Level"
                                type="number"
                                value={waveletLevel}
                                onChange={(e) => setWaveletLevel(parseInt(e.target.value) || 4)}
                                inputProps={{ min: 1, max: 10 }}
                                disabled={!waveletEnabled}
                            />
                        </Grid>
                        <Grid item xs={12}>
                            <FormControl fullWidth size="small" disabled={!waveletEnabled}>
                                <InputLabel>Threshold Mode</InputLabel>
                                <Select
                                    value={waveletThresholdMode}
                                    label="Threshold Mode"
                                    onChange={(e) => setWaveletThresholdMode(e.target.value)}
                                >
                                    <MenuItem value="soft">Soft</MenuItem>
                                    <MenuItem value="hard">Hard</MenuItem>
                                </Select>
                            </FormControl>
                        </Grid>
                        <Grid item xs={12}>
                            <Button
                                variant="outlined"
                                startIcon={loading ? <CircularProgress size={16} /> : <PlayArrowIcon />}
                                onClick={() => handleApplyFilter('wavelet')}
                                disabled={!waveletEnabled || loading || parentLoading}
                                fullWidth
                            >
                                Apply Wavelet Filter
                            </Button>
                        </Grid>
                    </Grid>
                </AccordionDetails>
            </Accordion>

            {/* Combined Filter */}
            <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography>Combined Filters</Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Grid container spacing={2}>
                        <Grid item xs={12}>
                            <Typography variant="caption" color="text.secondary">
                                Apply multiple filters in sequence. Enable the filters you want to combine above.
                            </Typography>
                        </Grid>
                        <Grid item xs={12}>
                            <Button
                                variant="contained"
                                startIcon={loading ? <CircularProgress size={16} /> : <PlayArrowIcon />}
                                onClick={() => handleApplyFilter('combined')}
                                disabled={loading || parentLoading || (!savgolEnabled && !butterEnabled && !waveletEnabled)}
                                fullWidth
                            >
                                Apply Combined Filters
                            </Button>
                        </Grid>
                    </Grid>
                </AccordionDetails>
            </Accordion>

            {/* Reset Button */}
            {filteredData && (
                <Box sx={{ mt: 2 }}>
                    <Button
                        variant="outlined"
                        color="secondary"
                        onClick={handleReset}
                        fullWidth
                    >
                        Reset Filters
                    </Button>
                </Box>
            )}
        </Box>
    );
};

export default FilterPanel;

