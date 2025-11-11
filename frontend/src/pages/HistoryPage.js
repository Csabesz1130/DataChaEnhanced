import React, { useState, useEffect } from 'react';
import {
    Box,
    Typography,
    Paper,
    CircularProgress,
    Alert,
    Button,
    Chip,
} from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';
import RefreshIcon from '@mui/icons-material/Refresh';
import { useNotification } from '../contexts/NotificationContext';
import { getAnalysisHistory, getAnalysisRun } from '../services/api';

function HistoryPage() {
    const { showError } = useNotification();
    const [runs, setRuns] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const loadHistory = async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await getAnalysisHistory();
            setRuns(data.runs || []);
        } catch (err) {
            setError('Failed to load analysis history');
            showError('Failed to load analysis history');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadHistory();
    }, []);

    const columns = [
        {
            field: 'id',
            headerName: 'ID',
            width: 100,
            renderCell: (params) => (
                <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                    {params.value.substring(0, 8)}...
                </Typography>
            ),
        },
        {
            field: 'created_at',
            headerName: 'Date',
            width: 180,
            valueFormatter: (params) => {
                if (!params.value) return '';
                return new Date(params.value).toLocaleString();
            },
        },
        {
            field: 'file_name',
            headerName: 'File',
            width: 200,
            flex: 1,
        },
        {
            field: 'processing_time',
            headerName: 'Time (s)',
            width: 120,
            valueFormatter: (params) => {
                if (!params.value) return '';
                return params.value.toFixed(2);
            },
        },
        {
            field: 'params',
            headerName: 'Parameters',
            width: 200,
            renderCell: (params) => {
                const p = params.value || {};
                return (
                    <Chip
                        label={`n_cycles: ${p.n_cycles || 'N/A'}`}
                        size="small"
                        variant="outlined"
                    />
                );
            },
        },
        {
            field: 'actions',
            headerName: 'Actions',
            width: 150,
            sortable: false,
            renderCell: (params) => (
                <Button
                    size="small"
                    variant="outlined"
                    onClick={() => handleLoadRun(params.row.id)}
                >
                    Load
                </Button>
            ),
        },
    ];

    const handleLoadRun = async (runId) => {
        try {
            const run = await getAnalysisRun(runId);
            // TODO: Navigate to analyzer page with run data loaded
            // For now, just log it
            console.log('Load run:', run);
        } catch (err) {
            showError('Failed to load run');
        }
    };

    if (loading) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 400 }}>
                <CircularProgress />
            </Box>
        );
    }

    return (
        <Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Typography variant="h4">Analysis History</Typography>
                <Button
                    variant="outlined"
                    startIcon={<RefreshIcon />}
                    onClick={loadHistory}
                >
                    Refresh
                </Button>
            </Box>

            {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                    {error}
                </Alert>
            )}

            <Paper sx={{ height: 600, width: '100%' }}>
                <DataGrid
                    rows={runs}
                    columns={columns}
                    pageSize={10}
                    rowsPerPageOptions={[10, 25, 50]}
                    disableSelectionOnClick
                    sx={{
                        '& .MuiDataGrid-cell:focus': {
                            outline: 'none',
                        },
                    }}
                />
            </Paper>
        </Box>
    );
}

export default HistoryPage;


