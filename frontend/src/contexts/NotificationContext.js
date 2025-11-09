import React, { createContext, useContext, useState, useCallback } from 'react';
import { Snackbar, Alert } from '@mui/material';

const NotificationContext = createContext();

export const useNotification = () => {
    const context = useContext(NotificationContext);
    if (!context) {
        throw new Error('useNotification must be used within NotificationProvider');
    }
    return context;
};

export const NotificationProvider = ({ children }) => {
    const [snackbar, setSnackbar] = useState({
        open: false,
        message: '',
        severity: 'info', // 'success', 'error', 'warning', 'info'
    });

    const showNotification = useCallback((message, severity = 'info') => {
        setSnackbar({
            open: true,
            message,
            severity,
        });
    }, []);

    const showSuccess = useCallback((message) => {
        showNotification(message, 'success');
    }, [showNotification]);

    const showError = useCallback((message) => {
        showNotification(message, 'error');
    }, [showNotification]);

    const showWarning = useCallback((message) => {
        showNotification(message, 'warning');
    }, [showNotification]);

    const showInfo = useCallback((message) => {
        showNotification(message, 'info');
    }, [showNotification]);

    const handleClose = useCallback((event, reason) => {
        if (reason === 'clickaway') {
            return;
        }
        setSnackbar((prev) => ({ ...prev, open: false }));
    }, []);

    const value = {
        showNotification,
        showSuccess,
        showError,
        showWarning,
        showInfo,
    };

    return (
        <NotificationContext.Provider value={value}>
            {children}
            <Snackbar
                open={snackbar.open}
                autoHideDuration={6000}
                onClose={handleClose}
                anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
            >
                <Alert
                    onClose={handleClose}
                    severity={snackbar.severity}
                    variant="filled"
                    sx={{ width: '100%' }}
                >
                    {snackbar.message}
                </Alert>
            </Snackbar>
        </NotificationContext.Provider>
    );
};

