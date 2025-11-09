import React, { useState } from 'react';
import { Container, Typography, Box, AppBar, Toolbar, Paper } from '@mui/material';
import FileUpload from './components/FileUpload';
import PlotViewer from './components/PlotViewer';
import AnalysisControls from './components/AnalysisControls';
import ExportButton from './components/ExportButton';
import ActionPotentialTab from './components/ActionPotentialTab';
import { analyzeFile } from './services/api';
import { getErrorMessage } from './utils/errorHandler';
import { useNotification } from './contexts/NotificationContext';

function App() {
    const { showSuccess, showError, showInfo } = useNotification();
    const [file, setFile] = useState(null);
    const [fileId, setFileId] = useState(null);
    const [analysisResults, setAnalysisResults] = useState(null);
    const [analysisId, setAnalysisId] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleFileUpload = async (uploadedFile, uploadedFileId) => {
        setFile(uploadedFile);
        setFileId(uploadedFileId);
        setAnalysisResults(null);
        setAnalysisId(null);
        setError(null);
        // Success notification handled here to avoid duplicate
    };

    const handleAnalyze = async (params) => {
        if (!fileId) {
            setError('Please upload a file first');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            // Map integration_method to use_alternative_method for API
            const apiParams = {
                ...params,
                use_alternative_method: params.integration_method === 'alternative',
            };
            
            const result = await analyzeFile(fileId, apiParams);
            setAnalysisResults(result.results);
            setAnalysisId(result.analysis_id);
            setError(null); // Clear any previous errors
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
        <div className="App">
            <AppBar position="static">
                <Toolbar>
                    <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                        Signal Analyzer
                    </Typography>
                </Toolbar>
            </AppBar>

            <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
                {error && (
                    <Box sx={{ mb: 2, p: 2, bgcolor: 'error.light', borderRadius: 1 }}>
                        <Typography color="error.contrastText">{error}</Typography>
                    </Box>
                )}

                <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 2 }}>
                    {/* Left Panel - Controls */}
                    <Paper sx={{ p: 2 }}>
                        <Typography variant="h6" gutterBottom>
                            1. Upload File
                        </Typography>
                        <FileUpload onFileUpload={handleFileUpload} />

                        <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
                            2. Set Parameters
                        </Typography>
                        <AnalysisControls
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
                                        // TODO: Implement spike removal API call
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

                    {/* Right Panel - Plot */}
                    <Paper sx={{ p: 2 }}>
                        <Typography variant="h6" gutterBottom>
                            Analysis Results
                        </Typography>
                        {loading ? (
                            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
                                <Typography>Analyzing...</Typography>
                            </Box>
                        ) : analysisResults ? (
                            <PlotViewer data={analysisResults} />
                        ) : (
                            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
                                <Typography color="text.secondary">
                                    Upload a file and run analysis to see results
                                </Typography>
                            </Box>
                        )}
                    </Paper>
                </Box>
            </Container>
        </div>
    );
}

export default App;

