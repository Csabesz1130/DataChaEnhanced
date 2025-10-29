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
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';

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
    });

    const handleChange = (e) => {
        setParams({
            ...params,
            [e.target.name]: parseFloat(e.target.value) || 0,
        });
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if (onAnalyze) {
            onAnalyze(params);
        }
    };

    return (
        <Box component="form" onSubmit={handleSubmit}>
            <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography>Basic Parameters</Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Grid container spacing={2}>
                        <Grid item xs={12}>
                            <TextField
                                fullWidth
                                size="small"
                                label="Number of Cycles"
                                name="n_cycles"
                                type="number"
                                value={params.n_cycles}
                                onChange={handleChange}
                            />
                        </Grid>
                        <Grid item xs={6}>
                            <TextField
                                fullWidth
                                size="small"
                                label="t0 (ms)"
                                name="t0"
                                type="number"
                                value={params.t0}
                                onChange={handleChange}
                            />
                        </Grid>
                        <Grid item xs={6}>
                            <TextField
                                fullWidth
                                size="small"
                                label="t1 (ms)"
                                name="t1"
                                type="number"
                                value={params.t1}
                                onChange={handleChange}
                            />
                        </Grid>
                        <Grid item xs={6}>
                            <TextField
                                fullWidth
                                size="small"
                                label="t2 (ms)"
                                name="t2"
                                type="number"
                                value={params.t2}
                                onChange={handleChange}
                            />
                        </Grid>
                        <Grid item xs={6}>
                            <TextField
                                fullWidth
                                size="small"
                                label="t3 (ms)"
                                name="t3"
                                type="number"
                                value={params.t3}
                                onChange={handleChange}
                            />
                        </Grid>
                    </Grid>
                </AccordionDetails>
            </Accordion>

            <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography>Voltage Parameters</Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Grid container spacing={2}>
                        <Grid item xs={4}>
                            <TextField
                                fullWidth
                                size="small"
                                label="V0 (mV)"
                                name="V0"
                                type="number"
                                value={params.V0}
                                onChange={handleChange}
                            />
                        </Grid>
                        <Grid item xs={4}>
                            <TextField
                                fullWidth
                                size="small"
                                label="V1 (mV)"
                                name="V1"
                                type="number"
                                value={params.V1}
                                onChange={handleChange}
                            />
                        </Grid>
                        <Grid item xs={4}>
                            <TextField
                                fullWidth
                                size="small"
                                label="V2 (mV)"
                                name="V2"
                                type="number"
                                value={params.V2}
                                onChange={handleChange}
                            />
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
                            />
                        </Grid>
                    </Grid>
                </AccordionDetails>
            </Accordion>

            <Button
                type="submit"
                variant="contained"
                fullWidth
                size="large"
                startIcon={<PlayArrowIcon />}
                disabled={disabled || loading}
                sx={{ mt: 2 }}
            >
                {loading ? 'Analyzing...' : 'Run Analysis'}
            </Button>
        </Box>
    );
};

export default AnalysisControls;

