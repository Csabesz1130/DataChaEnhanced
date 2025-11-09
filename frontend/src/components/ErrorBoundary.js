import React from 'react';
import { Box, Typography, Button, Paper, Alert } from '@mui/material';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';

class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null, errorInfo: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true };
    }

    componentDidCatch(error, errorInfo) {
        this.setState({
            error,
            errorInfo,
        });

        // Log error to console in development
        if (process.env.NODE_ENV === 'development') {
            console.error('ErrorBoundary caught an error:', error, errorInfo);
        }
    }

    handleReset = () => {
        this.setState({ hasError: false, error: null, errorInfo: null });
    };

    render() {
        if (this.state.hasError) {
            return (
                <Box
                    sx={{
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                        minHeight: '400px',
                        p: 4,
                    }}
                >
                    <Paper sx={{ p: 4, maxWidth: 600 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                            <ErrorOutlineIcon color="error" sx={{ fontSize: 40, mr: 2 }} />
                            <Typography variant="h5" component="h2">
                                Something went wrong
                            </Typography>
                        </Box>

                        <Alert severity="error" sx={{ mb: 2 }}>
                            <Typography variant="body1" gutterBottom>
                                An unexpected error occurred. Please try refreshing the page.
                            </Typography>
                            {process.env.NODE_ENV === 'development' && this.state.error && (
                                <Typography variant="body2" component="pre" sx={{ mt: 1, fontSize: '0.75rem' }}>
                                    {this.state.error.toString()}
                                    {this.state.errorInfo?.componentStack}
                                </Typography>
                            )}
                        </Alert>

                        <Box sx={{ display: 'flex', gap: 2 }}>
                            <Button variant="contained" onClick={this.handleReset}>
                                Try Again
                            </Button>
                            <Button variant="outlined" onClick={() => window.location.reload()}>
                                Refresh Page
                            </Button>
                        </Box>
                    </Paper>
                </Box>
            );
        }

        return this.props.children;
    }
}

export default ErrorBoundary;

