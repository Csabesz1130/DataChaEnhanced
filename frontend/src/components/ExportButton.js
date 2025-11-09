import React, { useState } from 'react';
import {
    Button,
    ButtonGroup,
    CircularProgress,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Checkbox,
    FormControlLabel,
    FormGroup,
    Typography,
} from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import SettingsIcon from '@mui/icons-material/Settings';
import { exportToExcel, exportToCSV, downloadExport } from '../services/api';
import { useNotification } from '../contexts/NotificationContext';
import { getErrorMessage } from '../utils/errorHandler';

const ExportButton = ({ analysisId }) => {
    const { showSuccess, showError } = useNotification();
    const [loading, setLoading] = useState(false);
    const [optionsOpen, setOptionsOpen] = useState(false);
    const [exportOptions, setExportOptions] = useState({
        includeCharts: true,
        includeRawData: true,
        includeCurves: {
            orange: true,
            normalized: true,
            average: true,
            hyperpol: true,
            depol: true,
        },
    });

    const handleExport = async (format, options = {}) => {
        setLoading(true);

        try {
            let result;
            if (format === 'excel') {
                result = await exportToExcel(analysisId, {
                    include_charts: options.includeCharts ?? exportOptions.includeCharts,
                    include_raw_data: options.includeRawData ?? exportOptions.includeRawData,
                });
            } else if (format === 'csv') {
                result = await exportToCSV(analysisId, {
                    include_raw_data: options.includeRawData ?? exportOptions.includeRawData,
                });
            }

            // Trigger download
            const downloadUrl = downloadExport(result.export_id);
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.download = result.filename || `export.${format === 'excel' ? 'xlsx' : 'csv'}`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            showSuccess(`${format.toUpperCase()} export successful!`);
            setOptionsOpen(false);
        } catch (err) {
            const errorMessage = getErrorMessage(err);
            showError(`Export failed: ${errorMessage}`);
            console.error('Export error:', err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <>
            <ButtonGroup fullWidth variant="contained" disabled={loading}>
                <Button
                    onClick={() => handleExport('excel')}
                    startIcon={loading ? <CircularProgress size={20} /> : <DownloadIcon />}
                    disabled={loading}
                >
                    Export Excel
                </Button>
                <Button
                    onClick={() => handleExport('csv')}
                    startIcon={loading ? <CircularProgress size={20} /> : <DownloadIcon />}
                    disabled={loading}
                >
                    Export CSV
                </Button>
                <Button
                    onClick={() => setOptionsOpen(true)}
                    startIcon={<SettingsIcon />}
                    disabled={loading}
                >
                    Options
                </Button>
            </ButtonGroup>

            {/* Export Options Dialog */}
            <Dialog open={optionsOpen} onClose={() => setOptionsOpen(false)} maxWidth="sm" fullWidth>
                <DialogTitle>Export Options</DialogTitle>
                <DialogContent>
                    <FormGroup>
                        <FormControlLabel
                            control={
                                <Checkbox
                                    checked={exportOptions.includeCharts}
                                    onChange={(e) =>
                                        setExportOptions({
                                            ...exportOptions,
                                            includeCharts: e.target.checked,
                                        })
                                    }
                                />
                            }
                            label="Include Charts (Excel only)"
                        />
                        <FormControlLabel
                            control={
                                <Checkbox
                                    checked={exportOptions.includeRawData}
                                    onChange={(e) =>
                                        setExportOptions({
                                            ...exportOptions,
                                            includeRawData: e.target.checked,
                                        })
                                    }
                                />
                            }
                            label="Include Raw Data"
                        />
                        <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
                            Include Curves:
                        </Typography>
                        <FormControlLabel
                            control={
                                <Checkbox
                                    checked={exportOptions.includeCurves.orange}
                                    onChange={(e) =>
                                        setExportOptions({
                                            ...exportOptions,
                                            includeCurves: {
                                                ...exportOptions.includeCurves,
                                                orange: e.target.checked,
                                            },
                                        })
                                    }
                                />
                            }
                            label="Orange Curve"
                        />
                        <FormControlLabel
                            control={
                                <Checkbox
                                    checked={exportOptions.includeCurves.normalized}
                                    onChange={(e) =>
                                        setExportOptions({
                                            ...exportOptions,
                                            includeCurves: {
                                                ...exportOptions.includeCurves,
                                                normalized: e.target.checked,
                                            },
                                        })
                                    }
                                />
                            }
                            label="Normalized Curve"
                        />
                        <FormControlLabel
                            control={
                                <Checkbox
                                    checked={exportOptions.includeCurves.average}
                                    onChange={(e) =>
                                        setExportOptions({
                                            ...exportOptions,
                                            includeCurves: {
                                                ...exportOptions.includeCurves,
                                                average: e.target.checked,
                                            },
                                        })
                                    }
                                />
                            }
                            label="Average Curve"
                        />
                        <FormControlLabel
                            control={
                                <Checkbox
                                    checked={exportOptions.includeCurves.hyperpol}
                                    onChange={(e) =>
                                        setExportOptions({
                                            ...exportOptions,
                                            includeCurves: {
                                                ...exportOptions.includeCurves,
                                                hyperpol: e.target.checked,
                                            },
                                        })
                                    }
                                />
                            }
                            label="Modified Hyperpol"
                        />
                        <FormControlLabel
                            control={
                                <Checkbox
                                    checked={exportOptions.includeCurves.depol}
                                    onChange={(e) =>
                                        setExportOptions({
                                            ...exportOptions,
                                            includeCurves: {
                                                ...exportOptions.includeCurves,
                                                depol: e.target.checked,
                                            },
                                        })
                                    }
                                />
                            }
                            label="Modified Depol"
                        />
                    </FormGroup>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setOptionsOpen(false)}>Cancel</Button>
                    <Button
                        onClick={() => handleExport('excel', exportOptions)}
                        variant="contained"
                        disabled={loading}
                        startIcon={loading ? <CircularProgress size={16} /> : <DownloadIcon />}
                    >
                        Export with Options
                    </Button>
                </DialogActions>
            </Dialog>
        </>
    );
};

export default ExportButton;

