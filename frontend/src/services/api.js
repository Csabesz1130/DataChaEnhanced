/**
 * API service for backend communication
 */

import axios from 'axios';
import { handleError, retryWithBackoff, AppError } from '../utils/errorHandler';

// Use environment variable or default to localhost
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

// Create axios instance with defaults
const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
    timeout: 30000, // 30 seconds timeout
});

// Request interceptor
api.interceptors.request.use(
    (config) => {
        // Add any auth tokens here if needed
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

// Response interceptor for global error handling
api.interceptors.response.use(
    (response) => {
        return response;
    },
    async (error) => {
        const handledError = handleError(error, {
            url: error.config?.url,
            method: error.config?.method,
        });

        // Convert to AppError for consistent handling
        const appError = new AppError(
            handledError.message,
            error.response?.status >= 500 ? 'SERVER_ERROR' :
            error.response?.status === 408 || error.code === 'ECONNABORTED' ? 'TIMEOUT' :
            error.request && !error.response ? 'NETWORK_ERROR' :
            'UNKNOWN_ERROR',
            error
        );

        return Promise.reject(appError);
    }
);

// File Upload
export const uploadFile = async (file, onProgress = null) => {
    const formData = new FormData();
    formData.append('file', file);

    const config = {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    };

    if (onProgress) {
        config.onUploadProgress = (progressEvent) => {
            const percentCompleted = Math.round(
                (progressEvent.loaded * 100) / progressEvent.total
            );
            onProgress(percentCompleted);
        };
    }

    const response = await retryWithBackoff(() =>
        api.post('/files/upload', formData, config)
    );

    return response.data;
};

// Analysis
export const analyzeFile = async (fileId, params) => {
    // Extract options from params if provided, otherwise use defaults
    const {
        use_alternative_method = false,
        auto_optimize_starting_point = true,
        n,
        ...analysisParams
    } = params;

    const response = await retryWithBackoff(() =>
        api.post('/analysis/process', {
            file_id: fileId,
            params: {
                ...analysisParams,
                n: n || undefined, // Only include n if provided
            },
            options: {
                use_alternative_method: use_alternative_method,
                auto_optimize_starting_point: auto_optimize_starting_point,
            },
        })
    );

    return response.data;
};

export const getAnalysisResults = async (analysisId) => {
    const response = await api.get(`/analysis/results/${analysisId}`);
    return response.data;
};

export const calculateIntegrals = async (analysisId, ranges, linregParams = null) => {
    const payload = {
        analysis_id: analysisId,
        ranges: ranges,
        method: linregParams ? 'linreg' : 'direct',
    };

    if (linregParams) {
        payload.linreg_params = linregParams;
    }

    const response = await retryWithBackoff(() =>
        api.post('/analysis/integrals', payload)
    );

    return response.data;
};

// Filtering
export const applySavgolFilter = async (data, windowLength = 51, polyorder = 3) => {
    const response = await retryWithBackoff(() =>
        api.post('/filter/savgol', {
            data: data,
            window_length: windowLength,
            polyorder: polyorder,
        })
    );

    return response.data;
};

export const applyButterworthFilter = async (data, cutoff = 100, fs = 1000, order = 5) => {
    const response = await retryWithBackoff(() =>
        api.post('/filter/butterworth', {
            data: data,
            cutoff: cutoff,
            fs: fs,
            order: order,
        })
    );

    return response.data;
};

export const applyWaveletFilter = async (data, wavelet = 'db4', level = 4, thresholdMode = 'soft') => {
    const response = await retryWithBackoff(() =>
        api.post('/filter/wavelet', {
            data: data,
            wavelet: wavelet,
            level: level,
            threshold_mode: thresholdMode,
        })
    );

    return response.data;
};

export const applyCombinedFilter = async (data, filterParams) => {
    const response = await retryWithBackoff(() =>
        api.post('/filter/combined', {
            data: data,
            ...filterParams,
        })
    );

    return response.data;
};

// Export
export const exportToExcel = async (analysisId, options = {}) => {
    const response = await retryWithBackoff(() =>
        api.post('/export/excel', {
            analysis_id: analysisId,
            include_charts: options.include_charts ?? true,
            include_raw_data: options.include_raw_data ?? true,
        })
    );

    return response.data;
};

export const exportToCSV = async (analysisId, options = {}) => {
    const response = await retryWithBackoff(() =>
        api.post('/export/csv', {
            analysis_id: analysisId,
            include_raw_data: options.include_raw_data ?? true,
        })
    );

    return response.data;
};

export const downloadExport = (exportId) => {
    return `${API_BASE_URL}/export/download/${exportId}`;
};

// Health check
export const healthCheck = async () => {
    const response = await api.get('/health');
    return response.data;
};

export default api;

