import React, { useState } from 'react';
import {
    Button,
    ButtonGroup,
    CircularProgress,
    Snackbar,
    Alert,
} from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import { exportToExcel, exportToCSV, downloadExport } from '../services/api';

const ExportButton = ({ analysisId }) => {
    const [loading, setLoading] = useState(false);
    const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });

    const handleExport = async (format) => {
        setLoading(true);

        try {
            let result;
            if (format === 'excel') {
                result = await exportToExcel(analysisId);
            } else if (format === 'csv') {
                result = await exportToCSV(analysisId);
            }

            // Trigger download
            const downloadUrl = downloadExport(result.export_id);
            window.open(downloadUrl, '_blank');

            setSnackbar({
                open: true,
                message: `${format.toUpperCase()} export successful!`,
                severity: 'success',
            });
        } catch (err) {
            setSnackbar({
                open: true,
                message: `Export failed: ${err.message}`,
                severity: 'error',
            });
            console.error('Export error:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleCloseSnackbar = () => {
        setSnackbar({ ...snackbar, open: false });
    };

    return (
        <>
            <ButtonGroup fullWidth variant="contained" disabled={loading}>
                <Button
                    onClick={() => handleExport('excel')}
                    startIcon={loading ? <CircularProgress size={20} /> : <DownloadIcon />}
                >
                    Export Excel
                </Button>
                <Button
                    onClick={() => handleExport('csv')}
                    startIcon={loading ? <CircularProgress size={20} /> : <DownloadIcon />}
                >
                    Export CSV
                </Button>
            </ButtonGroup>

            <Snackbar
                open={snackbar.open}
                autoHideDuration={6000}
                onClose={handleCloseSnackbar}
                anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
            >
                <Alert onClose={handleCloseSnackbar} severity={snackbar.severity}>
                    {snackbar.message}
                </Alert>
            </Snackbar>
        </>
    );
};

export default ExportButton;

