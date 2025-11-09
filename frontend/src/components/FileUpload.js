import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Box, Typography, Paper, CircularProgress, LinearProgress } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import { uploadFile } from '../services/api';
import { getErrorMessage } from '../utils/errorHandler';
import { useNotification } from '../contexts/NotificationContext';

const FileUpload = ({ onFileUpload }) => {
    const { showSuccess, showError } = useNotification();
    const [uploading, setUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [uploadedFile, setUploadedFile] = useState(null);
    const [error, setError] = useState(null);

    const onDrop = useCallback(async (acceptedFiles) => {
        if (acceptedFiles.length === 0) return;

        const file = acceptedFiles[0];
        setUploading(true);
        setUploadProgress(0);
        setError(null);

        try {
            const result = await uploadFile(file, (progress) => {
                setUploadProgress(progress);
            });
            setUploadedFile(result);
            setUploadProgress(100);
            if (onFileUpload) {
                onFileUpload(file, result.file_id);
            }
            // Success notification is handled in App.js
        } catch (err) {
            const errorMessage = getErrorMessage(err);
            setError(errorMessage);
            showError(errorMessage);
            console.error('Upload error:', err);
        } finally {
            setUploading(false);
            // Reset progress after a short delay
            setTimeout(() => setUploadProgress(0), 1000);
        }
    }, [onFileUpload]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'text/plain': ['.atf', '.txt'],
        },
        maxFiles: 1,
        disabled: uploading,
    });

    return (
        <Box>
            <Paper
                {...getRootProps()}
                sx={{
                    p: 3,
                    textAlign: 'center',
                    cursor: uploading ? 'default' : 'pointer',
                    border: '2px dashed',
                    borderColor: isDragActive ? 'primary.main' : 'grey.400',
                    bgcolor: isDragActive ? 'action.hover' : 'background.paper',
                    '&:hover': {
                        borderColor: 'primary.main',
                        bgcolor: 'action.hover',
                    },
                }}
            >
                <input {...getInputProps()} />

                {uploading ? (
                    <>
                        <CircularProgress size={48} />
                        <Typography sx={{ mt: 2 }}>Uploading...</Typography>
                        <Box sx={{ width: '100%', mt: 2 }}>
                            <LinearProgress variant="determinate" value={uploadProgress} />
                            <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
                                {uploadProgress}%
                            </Typography>
                        </Box>
                    </>
                ) : uploadedFile ? (
                    <>
                        <CheckCircleIcon color="success" sx={{ fontSize: 48 }} />
                        <Typography sx={{ mt: 2 }} variant="body1">
                            {uploadedFile.filename}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                            {(uploadedFile.file_size / 1024).toFixed(2)} KB
                        </Typography>
                        <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                            Click or drag to upload another file
                        </Typography>
                    </>
                ) : (
                    <>
                        <CloudUploadIcon sx={{ fontSize: 48, color: 'text.secondary' }} />
                        {isDragActive ? (
                            <Typography sx={{ mt: 2 }}>Drop file here...</Typography>
                        ) : (
                            <>
                                <Typography sx={{ mt: 2 }}>
                                    Drag & drop ATF file here
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                    or click to select file
                                </Typography>
                            </>
                        )}
                    </>
                )}
            </Paper>

            {error && (
                <Typography color="error" variant="caption" sx={{ mt: 1, display: 'block' }}>
                    {error}
                </Typography>
            )}
        </Box>
    );
};

export default FileUpload;

