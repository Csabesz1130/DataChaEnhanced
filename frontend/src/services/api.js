/**
 * API service for backend communication
 */

import axios from 'axios';

// Use environment variable or default to localhost
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

// Create axios instance with defaults
const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// File Upload
export const uploadFile = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post('/files/upload', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });

    return response.data;
};

// Analysis
export const analyzeFile = async (fileId, params) => {
    const response = await api.post('/analysis/process', {
        file_id: fileId,
        params: params,
        options: {
            use_alternative_method: false,
            auto_optimize_starting_point: true,
        },
    });

    return response.data;
};

export const getAnalysisResults = async (analysisId) => {
    const response = await api.get(`/analysis/results/${analysisId}`);
    return response.data;
};

export const calculateIntegrals = async (analysisId, ranges) => {
    const response = await api.post('/analysis/integrals', {
        analysis_id: analysisId,
        ranges: ranges,
        method: 'direct',
    });

    return response.data;
};

// Filtering
export const applySavgolFilter = async (data, windowLength = 51, polyorder = 3) => {
    const response = await api.post('/filter/savgol', {
        data: data,
        window_length: windowLength,
        polyorder: polyorder,
    });

    return response.data;
};

export const applyButterworthFilter = async (data, cutoff = 100, fs = 1000, order = 5) => {
    const response = await api.post('/filter/butterworth', {
        data: data,
        cutoff: cutoff,
        fs: fs,
        order: order,
    });

    return response.data;
};

// Export
export const exportToExcel = async (analysisId) => {
    const response = await api.post('/export/excel', {
        analysis_id: analysisId,
        include_charts: true,
        include_raw_data: true,
    });

    return response.data;
};

export const exportToCSV = async (analysisId) => {
    const response = await api.post('/export/csv', {
        analysis_id: analysisId,
    });

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

