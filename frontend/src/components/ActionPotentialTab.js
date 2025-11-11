import React, { useState, useEffect } from 'react';
import {
    Box,
    Typography,
    Button,
    TextField,
    Grid,
    FormControlLabel,
    Switch,
    Alert,
    CircularProgress,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    Divider,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import CalculateIcon from '@mui/icons-material/Calculate';
import { calculateIntegrals } from '../services/api';
import { getErrorMessage } from '../utils/errorHandler';

const ActionPotentialTab = ({ analysisId, analysisResults, onSpikeRemoval, onIntegrationChange }) => {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [integrals, setIntegrals] = useState(null);
    const [capacitance, setCapacitance] = useState(null);

    // Integration ranges
    const [hyperpolRange, setHyperpolRange] = useState({ start: 0, end: 199 });
    const [depolRange, setDepolRange] = useState({ start: 0, end: 199 });

    // Regression controls
    const [useRegressionHyperpol, setUseRegressionHyperpol] = useState(false);
    const [useRegressionDepol, setUseRegressionDepol] = useState(false);

    // Results display
    const [results, setResults] = useState(null);

    useEffect(() => {
        if (analysisResults) {
            setResults(analysisResults);
        }
    }, [analysisResults]);

    const handleSpikeRemoval = async () => {
        if (!analysisId) {
            setError('No analysis available. Please run analysis first.');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            if (onSpikeRemoval) {
                await onSpikeRemoval(analysisId);
                // Note: Spike removal would need a backend endpoint
                // For now, we'll just notify the parent component
            }
        } catch (err) {
            const errorMessage = getErrorMessage(err);
            setError(errorMessage);
            console.error('Spike removal error:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleCalculateIntegrals = async () => {
        if (!analysisId) {
            setError('No analysis available. Please run analysis first.');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const ranges = {
                hyperpol: hyperpolRange,
                depol: depolRange,
            };

            const linregParams = {};
            if (useRegressionHyperpol) {
                // TODO: Get regression parameters from curve fitting
                linregParams.hyperpol = { slope: 0, intercept: 0 };
            }
            if (useRegressionDepol) {
                // TODO: Get regression parameters from curve fitting
                linregParams.depol = { slope: 0, intercept: 0 };
            }

            const result = await calculateIntegrals(analysisId, ranges, linregParams);
            setIntegrals(result);
            setCapacitance(result.capacitance);

            // Notify parent of integration change
            if (onIntegrationChange) {
                onIntegrationChange(ranges, result);
            }
        } catch (err) {
            const errorMessage = getErrorMessage(err);
            setError(errorMessage);
            console.error('Integral calculation error:', err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <Box>
            <Typography variant="h6" gutterBottom>
                Action Potential Analysis
            </Typography>

            {error && (
                <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
                    {error}
                </Alert>
            )}

            {/* Results Display */}
            <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography>Results</Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Grid container spacing={2}>
                        {integrals && (
                            <>
                                <Grid item xs={12}>
                                    <Typography variant="subtitle2" gutterBottom>
                                        Integral Values
                                    </Typography>
                                </Grid>
                                <Grid item xs={6}>
                                    <Typography variant="body2" color="text.secondary">
                                        Hyperpolarization:
                                    </Typography>
                                    <Typography variant="h6">
                                        {integrals.hyperpol_integral?.toFixed(2) || 'N/A'}
                                    </Typography>
                                </Grid>
                                <Grid item xs={6}>
                                    <Typography variant="body2" color="text.secondary">
                                        Depolarization:
                                    </Typography>
                                    <Typography variant="h6">
                                        {integrals.depol_integral?.toFixed(2) || 'N/A'}
                                    </Typography>
                                </Grid>
                                {capacitance !== null && (
                                    <Grid item xs={12}>
                                        <Divider sx={{ my: 1 }} />
                                        <Typography variant="body2" color="text.secondary">
                                            Linear Capacitance:
                                        </Typography>
                                        <Typography variant="h6" color="primary">
                                            {capacitance.toFixed(2)} pF
                                        </Typography>
                                    </Grid>
                                )}
                            </>
                        )}
                        {!integrals && (
                            <Grid item xs={12}>
                                <Typography variant="body2" color="text.secondary">
                                    Calculate integrals to see results
                                </Typography>
                            </Grid>
                        )}
                    </Grid>
                </AccordionDetails>
            </Accordion>

            {/* Spike Removal */}
            <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography>Spike Removal</Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Box>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            Eliminate periodic spikes at (n + 200*i) from all curves.
                        </Typography>
                        <Button
                            variant="outlined"
                            onClick={handleSpikeRemoval}
                            disabled={!analysisId || loading}
                            startIcon={loading ? <CircularProgress size={16} /> : <PlayArrowIcon />}
                            fullWidth
                        >
                            Remove Spikes
                        </Button>
                    </Box>
                </AccordionDetails>
            </Accordion>

            {/* Integration Range Controls */}
            <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography>Integration Ranges</Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Grid container spacing={2}>
                        <Grid item xs={12}>
                            <Typography variant="subtitle2" gutterBottom>
                                Hyperpolarization Range
                            </Typography>
                        </Grid>
                        <Grid item xs={6}>
                            <TextField
                                fullWidth
                                size="small"
                                label="Start Index"
                                type="number"
                                value={hyperpolRange.start}
                                onChange={(e) =>
                                    setHyperpolRange({
                                        ...hyperpolRange,
                                        start: parseInt(e.target.value) || 0,
                                    })
                                }
                                inputProps={{ min: 0 }}
                            />
                        </Grid>
                        <Grid item xs={6}>
                            <TextField
                                fullWidth
                                size="small"
                                label="End Index"
                                type="number"
                                value={hyperpolRange.end}
                                onChange={(e) =>
                                    setHyperpolRange({
                                        ...hyperpolRange,
                                        end: parseInt(e.target.value) || 199,
                                    })
                                }
                                inputProps={{ min: 0 }}
                            />
                        </Grid>

                        <Grid item xs={12}>
                            <Divider sx={{ my: 1 }} />
                            <Typography variant="subtitle2" gutterBottom>
                                Depolarization Range
                            </Typography>
                        </Grid>
                        <Grid item xs={6}>
                            <TextField
                                fullWidth
                                size="small"
                                label="Start Index"
                                type="number"
                                value={depolRange.start}
                                onChange={(e) =>
                                    setDepolRange({
                                        ...depolRange,
                                        start: parseInt(e.target.value) || 0,
                                    })
                                }
                                inputProps={{ min: 0 }}
                            />
                        </Grid>
                        <Grid item xs={6}>
                            <TextField
                                fullWidth
                                size="small"
                                label="End Index"
                                type="number"
                                value={depolRange.end}
                                onChange={(e) =>
                                    setDepolRange({
                                        ...depolRange,
                                        end: parseInt(e.target.value) || 199,
                                    })
                                }
                                inputProps={{ min: 0 }}
                            />
                        </Grid>
                    </Grid>
                </AccordionDetails>
            </Accordion>

            {/* Regression Controls */}
            <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography>Regression Controls</Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Box>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            Use linear regression for baseline correction in integral calculations.
                        </Typography>
                        <FormControlLabel
                            control={
                                <Switch
                                    checked={useRegressionHyperpol}
                                    onChange={(e) => setUseRegressionHyperpol(e.target.checked)}
                                />
                            }
                            label="Use Regression for Hyperpolarization"
                        />
                        <FormControlLabel
                            control={
                                <Switch
                                    checked={useRegressionDepol}
                                    onChange={(e) => setUseRegressionDepol(e.target.checked)}
                                />
                            }
                            label="Use Regression for Depolarization"
                        />
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                            Note: Regression parameters are obtained from curve fitting. Enable curve fitting first.
                        </Typography>
                    </Box>
                </AccordionDetails>
            </Accordion>

            {/* Calculate Integrals Button */}
            <Box sx={{ mt: 2 }}>
                <Button
                    variant="contained"
                    fullWidth
                    size="large"
                    startIcon={loading ? <CircularProgress size={20} /> : <CalculateIcon />}
                    onClick={handleCalculateIntegrals}
                    disabled={!analysisId || loading}
                >
                    {loading ? 'Calculating...' : 'Calculate Integrals'}
                </Button>
            </Box>

            {/* Analysis Summary */}
            {results && (
                <Accordion sx={{ mt: 2 }}>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Typography>Analysis Summary</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                        <Grid container spacing={2}>
                            <Grid item xs={6}>
                                <Typography variant="body2" color="text.secondary">
                                    Baseline:
                                </Typography>
                                <Typography variant="body1">
                                    {results.baseline?.toFixed(2) || 'N/A'} pA
                                </Typography>
                            </Grid>
                            <Grid item xs={6}>
                                <Typography variant="body2" color="text.secondary">
                                    Number of Cycles:
                                </Typography>
                                <Typography variant="body1">
                                    {results.cycles || 'N/A'}
                                </Typography>
                            </Grid>
                        </Grid>
                    </AccordionDetails>
                </Accordion>
            )}
        </Box>
    );
};

export default ActionPotentialTab;

