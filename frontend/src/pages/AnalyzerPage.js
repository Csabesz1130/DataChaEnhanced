import React, { useState } from 'react';
import { Container, Typography, Box, Paper, Grid } from '@mui/material';
import FileUpload from '../components/FileUpload';
import PlotViewer from '../components/PlotViewer';
import AnalysisControlsStepper from '../components/AnalysisControlsStepper';
import ExportButton from '../components/ExportButton';
import ActionPotentialTab from '../components/ActionPotentialTab';
import { analyzeFile } from '../services/api';
import { getErrorMessage } from '../utils/errorHandler';
import { useNotification } from '../contexts/NotificationContext';

function AnalyzerPage() {
    const { showSuccess, showError, showInfo } = useNotification();
    // eslint-disable-next-line no-unused-vars
    const [file, setFile] = useState(null);
    const [fileId, setFileId] = useState(null);
    const [analysisResults, setAnalysisResults] = useState(null);
    const [analysisId, setAnalysisId] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleFileUpload = async (uploadedFile, uploadedFileId) => {
        // For now, use the first uploaded file
        // In the future, we could support selecting which file to analyze
        setFile(uploadedFile);
        setFileId(uploadedFileId);
        setAnalysisResults(null);
        setAnalysisId(null);
        setError(null);
    };

    const handleAnalyze = async (params) => {
        if (!fileId) {
            setError('Please upload a file first');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const apiParams = {
                ...params,
                use_alternative_method: params.integration_method === 'alternative',
            };
            
            const result = await analyzeFile(fileId, apiParams);
            setAnalysisResults(result.results);
            setAnalysisId(result.analysis_id);
            setError(null);
            showSuccess('Analysis completed successfully');
        } catch (err) {
            const errorMessage = getErrorMessage(err);
            setError(errorMessage);
            showError(errorMessage);
            console.error('Analysis error:', err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <Container maxWidth="xl">
            {error && (
                <Box sx={{ mb: 2, p: 2, bgcolor: 'error.light', borderRadius: 1 }}>
                    <Typography color="error.contrastText">{error}</Typography>
                </Box>
            )}

            <Grid container spacing={3}>
                {/* Left Panel - Controls */}
                <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 3, height: 'fit-content', position: 'sticky', top: 80 }}>
                        <Typography variant="h6" gutterBottom>
                            1. Upload File
                        </Typography>
                        <FileUpload onFileUpload={handleFileUpload} />

                        <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
                            2. Set Parameters
                        </Typography>
                        <AnalysisControlsStepper
                            onAnalyze={handleAnalyze}
                            loading={loading}
                            disabled={!fileId}
                        />

                        {analysisId && (
                            <>
                                <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
                                    3. Action Potential Analysis
                                </Typography>
                                <ActionPotentialTab
                                    analysisId={analysisId}
                                    analysisResults={analysisResults}
                                    onSpikeRemoval={async (id) => {
                                        showInfo('Spike removal feature coming soon');
                                    }}
                                    onIntegrationChange={(ranges, result) => {
                                        showSuccess('Integrals calculated successfully');
                                    }}
                                />

                                <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
                                    4. Export Results
                                </Typography>
                                <ExportButton analysisId={analysisId} />
                            </>
                        )}
                    </Paper>
                </Grid>

                {/* Right Panel - Plot */}
                <Grid item xs={12} md={8}>
                    <Paper sx={{ p: 3, minHeight: 600 }}>
                        <Typography variant="h6" gutterBottom>
                            Analysis Results
                        </Typography>
                        {loading ? (
                            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 500 }}>
                                <Typography>Analyzing...</Typography>
                            </Box>
                        ) : analysisResults ? (
                            <PlotViewer data={analysisResults} />
                        ) : (
                            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 500 }}>
                                <Typography color="text.secondary">
                                    Upload a file and run analysis to see results
                                </Typography>
                            </Box>
                        )}
                    </Paper>
                </Grid>
            </Grid>
        </Container>
    );
}

export default AnalyzerPage;

