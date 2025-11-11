import React, { useState } from 'react';
import {
    Box,
    TextField,
    Button,
    Grid,
    Typography,
    Stepper,
    Step,
    StepLabel,
    StepContent,
    FormControlLabel,
    Switch,
    RadioGroup,
    Radio,
    FormControl,
    Tooltip,
    IconButton,
    Alert,
    Paper,
} from '@mui/material';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

const ParameterTooltip = ({ title, children }) => (
    <Tooltip title={title} arrow>
        <IconButton size="small" sx={{ ml: 0.5, p: 0.5 }}>
            <HelpOutlineIcon fontSize="small" />
        </IconButton>
    </Tooltip>
);

const AnalysisControlsStepper = ({ onAnalyze, loading, disabled }) => {
    const [activeStep, setActiveStep] = useState(0);
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
        n: 35,
        auto_optimize_starting_point: true,
        use_alternative_method: false,
        integration_method: 'traditional',
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

        if (errors[name]) {
            setErrors({
                ...errors,
                [name]: undefined,
            });
        }
    };

    const validateParams = () => {
        const newErrors = {};

        if (params.n_cycles < 1 || !Number.isInteger(params.n_cycles)) {
            newErrors.n_cycles = 'Number of cycles must be a positive integer';
        }

        if (params.t0 <= 0) newErrors.t0 = 't0 must be positive';
        if (params.t1 <= 0) newErrors.t1 = 't1 must be positive';
        if (params.t2 <= 0) newErrors.t2 = 't2 must be positive';
        if (params.t3 <= 0) newErrors.t3 = 't3 must be positive';

        if (params.n !== '' && (params.n < 1 || !Number.isInteger(params.n))) {
            newErrors.n = 'Starting point must be a positive integer';
        }

        if (params.cell_area_cm2 <= 0) {
            newErrors.cell_area_cm2 = 'Cell area must be positive';
        }

        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const handleNext = () => {
        if (activeStep === 0) {
            // Validate basic params before moving to advanced
            if (validateParams()) {
                setActiveStep(activeStep + 1);
            }
        } else {
            setActiveStep(activeStep + 1);
        }
    };

    const handleBack = () => {
        setActiveStep(activeStep - 1);
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        
        if (!validateParams()) {
            setActiveStep(0);
            return;
        }

        const submitParams = {
            ...params,
            n: params.n === '' ? null : params.n,
        };

        if (onAnalyze) {
            onAnalyze(submitParams);
        }
    };

    const steps = [
        {
            label: 'Integration Method',
            content: (
                <Box sx={{ mt: 2 }}>
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
                            <Typography variant="caption" color="text.secondary" sx={{ ml: 4, mb: 2, display: 'block' }}>
                                Standard integration method
                            </Typography>
                            
                            <FormControlLabel
                                value="alternative"
                                control={<Radio />}
                                label="Averaged Normalized Method"
                            />
                            <Typography variant="caption" color="text.secondary" sx={{ ml: 4, display: 'block' }}>
                                Alternative method uses averaged normalized curves
                            </Typography>
                        </RadioGroup>
                    </FormControl>
                </Box>
            ),
        },
        {
            label: 'Basic Parameters',
            content: (
                <Grid container spacing={2} sx={{ mt: 1 }}>
                    <Grid item xs={12} sm={6}>
                        <TextField
                            fullWidth
                            label="Number of Cycles"
                            name="n_cycles"
                            type="number"
                            value={params.n_cycles}
                            onChange={handleChange}
                            error={!!errors.n_cycles}
                            helperText={errors.n_cycles}
                            inputProps={{ min: 1 }}
                        />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                        <TextField
                            fullWidth
                            label="Starting Point (n)"
                            name="n"
                            type="number"
                            value={params.n}
                            onChange={handleChange}
                            error={!!errors.n}
                            helperText={errors.n || 'Leave empty for auto-optimization'}
                            inputProps={{ min: 1 }}
                        />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <TextField
                                fullWidth
                                label="t0 (ms)"
                                name="t0"
                                type="number"
                                value={params.t0}
                                onChange={handleChange}
                                error={!!errors.t0}
                                helperText={errors.t0}
                                inputProps={{ min: 0.1, step: 0.1 }}
                            />
                            <ParameterTooltip title="Time constant for baseline correction" />
                        </Box>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <TextField
                                fullWidth
                                label="t1 (ms)"
                                name="t1"
                                type="number"
                                value={params.t1}
                                onChange={handleChange}
                                error={!!errors.t1}
                                helperText={errors.t1}
                                inputProps={{ min: 0.1, step: 0.1 }}
                            />
                            <ParameterTooltip title="Time constant for peak detection" />
                        </Box>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <TextField
                                fullWidth
                                label="t2 (ms)"
                                name="t2"
                                type="number"
                                value={params.t2}
                                onChange={handleChange}
                                error={!!errors.t2}
                                helperText={errors.t2}
                                inputProps={{ min: 0.1, step: 0.1 }}
                            />
                            <ParameterTooltip title="Time constant for normalization" />
                        </Box>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                        <TextField
                            fullWidth
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
                </Grid>
            ),
        },
        {
            label: 'Voltage & Cell Parameters',
            content: (
                <Grid container spacing={2} sx={{ mt: 1 }}>
                    <Grid item xs={12} sm={4}>
                        <TextField
                            fullWidth
                            label="V0 (mV)"
                            name="V0"
                            type="number"
                            value={params.V0}
                            onChange={handleChange}
                            inputProps={{ step: 0.1 }}
                        />
                    </Grid>
                    <Grid item xs={12} sm={4}>
                        <TextField
                            fullWidth
                            label="V1 (mV)"
                            name="V1"
                            type="number"
                            value={params.V1}
                            onChange={handleChange}
                            inputProps={{ step: 0.1 }}
                        />
                    </Grid>
                    <Grid item xs={12} sm={4}>
                        <TextField
                            fullWidth
                            label="V2 (mV)"
                            name="V2"
                            type="number"
                            value={params.V2}
                            onChange={handleChange}
                            inputProps={{ step: 0.1 }}
                        />
                    </Grid>
                    <Grid item xs={12}>
                        <TextField
                            fullWidth
                            label="Cell Area (cm²)"
                            name="cell_area_cm2"
                            type="number"
                            value={params.cell_area_cm2}
                            onChange={handleChange}
                            error={!!errors.cell_area_cm2}
                            helperText={errors.cell_area_cm2}
                            inputProps={{ min: 0, step: 0.0001 }}
                        />
                    </Grid>
                </Grid>
            ),
        },
        {
            label: 'Advanced Options',
            content: (
                <Box sx={{ mt: 2 }}>
                    <FormControlLabel
                        control={
                            <Switch
                                checked={params.auto_optimize_starting_point}
                                onChange={handleChange}
                                name="auto_optimize_starting_point"
                            />
                        }
                        label="Auto-optimize Starting Point"
                    />
                    <Typography variant="caption" color="text.secondary" sx={{ ml: 4, display: 'block' }}>
                        Automatically find optimal starting point if not specified
                    </Typography>
                </Box>
            ),
        },
        {
            label: 'Review & Run',
            content: (
                <Box sx={{ mt: 2 }}>
                    <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
                        <Typography variant="subtitle2" gutterBottom>Selected Parameters:</Typography>
                        <Typography variant="body2" component="div">
                            <strong>Method:</strong> {params.integration_method === 'traditional' ? 'Traditional' : 'Averaged Normalized'}<br />
                            <strong>Cycles:</strong> {params.n_cycles}<br />
                            <strong>Starting Point:</strong> {params.n || 'Auto'}<br />
                            <strong>Time Constants:</strong> t0={params.t0}ms, t1={params.t1}ms, t2={params.t2}ms, t3={params.t3}ms<br />
                            <strong>Voltages:</strong> V0={params.V0}mV, V1={params.V1}mV, V2={params.V2}mV<br />
                            <strong>Cell Area:</strong> {params.cell_area_cm2} cm²<br />
                            <strong>Auto-optimize:</strong> {params.auto_optimize_starting_point ? 'Yes' : 'No'}
                        </Typography>
                    </Paper>
                    {Object.keys(errors).length > 0 && (
                        <Alert severity="error" sx={{ mt: 2 }}>
                            Please fix validation errors before running analysis.
                        </Alert>
                    )}
                </Box>
            ),
        },
    ];

    return (
        <Box component="form" onSubmit={handleSubmit}>
            <Stepper activeStep={activeStep} orientation="vertical">
                {steps.map((step, index) => (
                    <Step key={step.label}>
                        <StepLabel
                            optional={
                                index === steps.length - 1 ? (
                                    <Typography variant="caption">Ready to analyze</Typography>
                                ) : null
                            }
                        >
                            {step.label}
                        </StepLabel>
                        <StepContent>
                            {step.content}
                            <Box sx={{ mb: 2, mt: 2 }}>
                                <div>
                                    <Button
                                        variant="contained"
                                        onClick={handleNext}
                                        sx={{ mt: 1, mr: 1 }}
                                        disabled={index === steps.length - 1}
                                    >
                                        {index === steps.length - 1 ? 'Finish' : 'Continue'}
                                    </Button>
                                    <Button
                                        disabled={index === 0}
                                        onClick={handleBack}
                                        sx={{ mt: 1, mr: 1 }}
                                    >
                                        Back
                                    </Button>
                                </div>
                            </Box>
                        </StepContent>
                    </Step>
                ))}
            </Stepper>
            
            {activeStep === steps.length && (
                <Box sx={{ mt: 3 }}>
                    <Alert icon={<CheckCircleIcon />} severity="success" sx={{ mb: 2 }}>
                        All steps completed. Click "Run Analysis" to proceed.
                    </Alert>
                    <Button
                        type="submit"
                        variant="contained"
                        size="large"
                        startIcon={loading ? <PlayArrowIcon /> : <PlayArrowIcon />}
                        disabled={loading || disabled || Object.keys(errors).length > 0}
                        fullWidth
                    >
                        {loading ? 'Analyzing...' : 'Run Analysis'}
                    </Button>
                </Box>
            )}
        </Box>
    );
};

export default AnalysisControlsStepper;


