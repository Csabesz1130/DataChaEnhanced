import React, { useState } from 'react';
import {
    Box,
    TextField,
    Button,
    Grid,
    Typography,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    FormControlLabel,
    Switch,
    RadioGroup,
    Radio,
    FormControl,
    FormLabel,
    Tooltip,
    IconButton,
    Alert,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';

const AnalysisControls = ({ onAnalyze, loading, disabled }) => {
    const [params, setParams] = useState({
        n_cycles: 2,
        t0: 20,
        t1: 100,
        t2: 100,
        t3: 1000,
        V0: -80,
        V1: -100,
        V2: -20,
        cell_area_cm2: 0.0001,
        n: 35, // Starting point - CRITICAL MISSING PARAMETER
        auto_optimize_starting_point: true,
        use_alternative_method: false,
        integration_method: 'traditional', // 'traditional' or 'alternative'
    });

    const [errors, setErrors] = useState({});

    const handleChange = (e) => {
        const { name, value, type, checked } = e.target;
        const newValue = type === 'checkbox' ? checked : 
                        type === 'radio' ? value :
                        type === 'number' ? (value === '' ? '' : parseFloat(value)) : value;
        
        setParams({
            ...params,
            [name]: newValue,
        });

        // Clear error for this field
        if (errors[name]) {
            setErrors({
                ...errors,
                [name]: null,
            });
        }
    };

    const validateParams = () => {
        const newErrors = {};

        // Validate n_cycles
        if (params.n_cycles < 1 || !Number.isInteger(params.n_cycles)) {
            newErrors.n_cycles = 'Number of cycles must be a positive integer';
        }

        // Validate time constants
        if (params.t0 <= 0) newErrors.t0 = 't0 must be positive';
        if (params.t1 <= 0) newErrors.t1 = 't1 must be positive';
        if (params.t2 <= 0) newErrors.t2 = 't2 must be positive';
        if (params.t3 <= 0) newErrors.t3 = 't3 must be positive';

        // Validate starting point
        if (params.n !== '' && (params.n < 1 || !Number.isInteger(params.n))) {
            newErrors.n = 'Starting point must be a positive integer';
        }

        // Validate cell area
        if (params.cell_area_cm2 <= 0) {
            newErrors.cell_area_cm2 = 'Cell area must be positive';
        }

        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        
        if (!validateParams()) {
            return;
        }

        // Prepare params for API (convert empty n to null for auto-optimization)
        const submitParams = {
            ...params,
            n: params.n === '' ? null : params.n,
        };

        if (onAnalyze) {
            onAnalyze(submitParams);
        }
    };

    const ParameterTooltip = ({ title, children }) => (
        <Tooltip title={title} arrow>
            <IconButton size="small" sx={{ ml: 0.5, p: 0.5 }}>
                <HelpOutlineIcon fontSize="small" />
            </IconButton>
        </Tooltip>
    );

    return (
        <Box component="form" onSubmit={handleSubmit}>
            {/* Integration Method */}
            <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography>Integration Method</Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <FormControl component="fieldset">
                        <RadioGroup
                            name="integration_method"
                            value={params.integration_method}
                            onChange={handleChange}
                        >
                            <FormControlLabel
                                value="traditional"
                                control={<Radio />}
                                label="Traditional Method"
                            />
                            <FormControlLabel
                                value="alternative"
                                control={<Radio />}
                                label="Averaged Normalized Method"
                            />
                        </RadioGroup>
                    </FormControl>
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                        Alternative method uses averaged normalized curves
                    </Typography>
                </AccordionDetails>
            </Accordion>

            {/* Basic Parameters */}
            <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography>Basic Parameters</Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Grid container spacing={2}>
                        <Grid item xs={12}>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                <TextField
                                    fullWidth
                                    size="small"
                                    label="Number of Cycles"
                                    name="n_cycles"
                                    type="number"
                                    value={params.n_cycles}
                                    onChange={handleChange}
                                    error={!!errors.n_cycles}
                                    helperText={errors.n_cycles}
                                    inputProps={{ min: 1, step: 1 }}
                                />
                                <ParameterTooltip title="Number of cycles to analyze in the signal" />
                            </Box>
                        </Grid>
                        <Grid item xs={4}>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                <TextField
                                    fullWidth
                                    size="small"
                                    label="t0 (ms)"
                                    name="t0"
                                    type="number"
                                    value={params.t0}
                                    onChange={handleChange}
                                    error={!!errors.t0}
                                    helperText={errors.t0}
                                    inputProps={{ min: 0.1, step: 0.1 }}
                                />
                                <ParameterTooltip title="Baseline time constant" />
                            </Box>
                        </Grid>
                        <Grid item xs={4}>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                <TextField
                                    fullWidth
                                    size="small"
                                    label="t1 (ms)"
                                    name="t1"
                                    type="number"
                                    value={params.t1}
                                    onChange={handleChange}
                                    error={!!errors.t1}
                                    helperText={errors.t1}
                                    inputProps={{ min: 0.1, step: 0.1 }}
                                />
                                <ParameterTooltip title="Hyperpolarization time constant" />
                            </Box>
                        </Grid>
                        <Grid item xs={4}>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                <TextField
                                    fullWidth
                                    size="small"
                                    label="t2 (ms)"
                                    name="t2"
                                    type="number"
                                    value={params.t2}
                                    onChange={handleChange}
                                    error={!!errors.t2}
                                    helperText={errors.t2}
                                    inputProps={{ min: 0.1, step: 0.1 }}
                                />
                                <ParameterTooltip title="Depolarization time constant" />
                            </Box>
                        </Grid>
                        <Grid item xs={12}>
                            <TextField
                                fullWidth
                                size="small"
                                label="t3 (ms)"
                                name="t3"
                                type="number"
                                value={params.t3}
                                onChange={handleChange}
                                error={!!errors.t3}
                                helperText={errors.t3}
                                inputProps={{ min: 0.1, step: 0.1 }}
                            />
                        </Grid>
                        <Grid item xs={12}>
                            <Typography variant="caption" color="text.secondary">
                                t0: baseline, t1: hyperpolarization, t2: depolarization
                            </Typography>
                        </Grid>
                    </Grid>
                </AccordionDetails>
            </Accordion>

            {/* Starting Point */}
            <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography>Starting Point (Normalization)</Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Grid container spacing={2}>
                        <Grid item xs={12}>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                <TextField
                                    fullWidth
                                    size="small"
                                    label="Starting Point (n)"
                                    name="n"
                                    type="number"
                                    value={params.n}
                                    onChange={handleChange}
                                    error={!!errors.n}
                                    helperText={errors.n || 'Leave blank to use default (35)'}
                                    inputProps={{ min: 1, step: 1 }}
                                    placeholder="35"
                                />
                                <ParameterTooltip title="Starting point for normalization segments. Leave blank for default (35) or use auto-optimization." />
                            </Box>
                        </Grid>
                        <Grid item xs={12}>
                            <FormControlLabel
                                control={
                                    <Switch
                                        checked={params.auto_optimize_starting_point}
                                        onChange={handleChange}
                                        name="auto_optimize_starting_point"
                                    />
                                }
                                label="Auto-optimize starting point"
                            />
                            <ParameterTooltip title="Automatically find the optimal starting point for smooth curves" />
                        </Grid>
                        <Grid item xs={12}>
                            <Typography variant="caption" color="text.secondary">
                                Auto-optimization finds the best starting point automatically. Leave starting point blank to use default value (35).
                            </Typography>
                        </Grid>
                    </Grid>
                </AccordionDetails>
            </Accordion>

            {/* Voltage Parameters */}
            <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography>Voltage Parameters</Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Grid container spacing={2}>
                        <Grid item xs={4}>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                <TextField
                                    fullWidth
                                    size="small"
                                    label="V0 (mV)"
                                    name="V0"
                                    type="number"
                                    value={params.V0}
                                    onChange={handleChange}
                                    inputProps={{ step: 1 }}
                                />
                                <ParameterTooltip title="Baseline voltage" />
                            </Box>
                        </Grid>
                        <Grid item xs={4}>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                <TextField
                                    fullWidth
                                    size="small"
                                    label="V1 (mV)"
                                    name="V1"
                                    type="number"
                                    value={params.V1}
                                    onChange={handleChange}
                                    inputProps={{ step: 1 }}
                                />
                                <ParameterTooltip title="Hyperpolarization voltage" />
                            </Box>
                        </Grid>
                        <Grid item xs={4}>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                <TextField
                                    fullWidth
                                    size="small"
                                    label="V2 (mV)"
                                    name="V2"
                                    type="number"
                                    value={params.V2}
                                    onChange={handleChange}
                                    inputProps={{ step: 1 }}
                                />
                                <ParameterTooltip title="Depolarization voltage" />
                            </Box>
                        </Grid>
                        <Grid item xs={12}>
                            <Typography variant="caption" color="text.secondary">
                                V0: baseline, V1: hyperpolarization, V2: depolarization
                            </Typography>
                        </Grid>
                        <Grid item xs={12}>
                            <TextField
                                fullWidth
                                size="small"
                                label="Cell Area (cm²)"
                                name="cell_area_cm2"
                                type="number"
                                step="0.0001"
                                value={params.cell_area_cm2}
                                onChange={handleChange}
                                error={!!errors.cell_area_cm2}
                                helperText={errors.cell_area_cm2}
                                inputProps={{ min: 0.0001, step: 0.0001 }}
                            />
                        </Grid>
                    </Grid>
                </AccordionDetails>
            </Accordion>

            {/* Error Display */}
            {Object.keys(errors).length > 0 && (
                <Alert severity="error" sx={{ mt: 2 }}>
                    Please fix the errors above before running analysis.
                </Alert>
            )}

            <Button
                type="submit"
                variant="contained"
                fullWidth
                size="large"
                startIcon={<PlayArrowIcon />}
                disabled={disabled || loading || Object.keys(errors).length > 0}
                sx={{ mt: 2 }}
            >
                {loading ? 'Analyzing...' : 'Run Analysis'}
            </Button>
        </Box>
    );
};

export default AnalysisControls;

