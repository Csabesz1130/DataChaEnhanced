import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import {
    Box,
    Typography,
    Paper,
    CircularProgress,
    LinearProgress,
    List,
    ListItem,
    ListItemText,
    ListItemSecondaryAction,
    IconButton,
    Chip,
    Divider,
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import DeleteIcon from '@mui/icons-material/Delete';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';
import { uploadFile } from '../services/api';
import { getErrorMessage } from '../utils/errorHandler';
import { useNotification } from '../contexts/NotificationContext';

const FileUpload = ({ onFileUpload, maxFiles = 5 }) => {
    const { showError, showSuccess } = useNotification();
    const [uploading, setUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState({});
    const [uploadedFiles, setUploadedFiles] = useState([]);
    const [error, setError] = useState(null);

    const onDrop = useCallback(async (acceptedFiles) => {
        if (acceptedFiles.length === 0) return;

        // Check if adding these files would exceed maxFiles
        if (uploadedFiles.length + acceptedFiles.length > maxFiles) {
            showError(`Maximum ${maxFiles} file(s) allowed`);
            return;
        }

        setUploading(true);
        setError(null);

        const uploadPromises = acceptedFiles.map(async (file) => {
            const fileId = `${Date.now()}-${Math.random()}`;
            setUploadProgress((prev) => ({ ...prev, [fileId]: 0 }));

            try {
                const result = await uploadFile(file, (progress) => {
                    setUploadProgress((prev) => ({ ...prev, [fileId]: progress }));
                });

                const fileInfo = {
                    id: fileId,
                    file: file,
                    fileId: result.file_id,
                    filename: result.filename || file.name,
                    fileSize: result.file_size || file.size,
                    uploadedAt: new Date(),
                };

                setUploadedFiles((prev) => [...prev, fileInfo]);
                setUploadProgress((prev) => ({ ...prev, [fileId]: 100 }));

                if (onFileUpload) {
                    onFileUpload(file, result.file_id);
                }

                showSuccess(`File "${file.name}" uploaded successfully`);
                return fileInfo;
            } catch (err) {
                const errorMessage = getErrorMessage(err);
                showError(`Failed to upload "${file.name}": ${errorMessage}`);
                setUploadProgress((prev) => {
                    const newPrev = { ...prev };
                    delete newPrev[fileId];
                    return newPrev;
                });
                throw err;
            }
        });

        try {
            await Promise.all(uploadPromises);
        } catch (err) {
            console.error('Upload error:', err);
        } finally {
            setUploading(false);
            // Clear progress after a delay
            setTimeout(() => {
                setUploadProgress({});
            }, 2000);
        }
    }, [onFileUpload, showError, showSuccess, uploadedFiles.length, maxFiles]);

    const handleDelete = (fileId) => {
        setUploadedFiles((prev) => prev.filter((f) => f.id !== fileId));
        setUploadProgress((prev) => {
            const newPrev = { ...prev };
            delete newPrev[fileId];
            return newPrev;
        });
    };

    const formatFileSize = (bytes) => {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
    };

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'text/plain': ['.atf', '.txt'],
        },
        maxFiles: maxFiles,
        disabled: uploading || uploadedFiles.length >= maxFiles,
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
                        {Object.keys(uploadProgress).length > 0 && (
                            <Box sx={{ width: '100%', mt: 2 }}>
                                {Object.entries(uploadProgress).map(([fileId, progress]) => (
                                    <Box key={fileId} sx={{ mb: 1 }}>
                                        <LinearProgress variant="determinate" value={progress} />
                                        <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
                                            {progress}%
                                        </Typography>
                                    </Box>
                                ))}
                            </Box>
                        )}
                    </>
                ) : uploadedFiles.length >= maxFiles ? (
                    <>
                        <CheckCircleIcon color="success" sx={{ fontSize: 48 }} />
                        <Typography sx={{ mt: 2 }} variant="body1">
                            Maximum files reached ({maxFiles})
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                            Delete a file to upload another
                        </Typography>
                    </>
                ) : (
                    <>
                        <CloudUploadIcon sx={{ fontSize: 48, color: 'text.secondary' }} />
                        {isDragActive ? (
                            <Typography sx={{ mt: 2 }}>Drop file(s) here...</Typography>
                        ) : (
                            <>
                                <Typography sx={{ mt: 2 }}>
                                    Drag & drop ATF file(s) here
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                    or click to select file(s)
                                </Typography>
                                {maxFiles > 1 && (
                                    <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5 }}>
                                        Up to {maxFiles} files ({uploadedFiles.length}/{maxFiles} uploaded)
                                    </Typography>
                                )}
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

            {uploadedFiles.length > 0 && (
                <Box sx={{ mt: 2 }}>
                    <Divider sx={{ mb: 2 }} />
                    <Typography variant="subtitle2" gutterBottom>
                        Uploaded Files ({uploadedFiles.length})
                    </Typography>
                    <List dense>
                        {uploadedFiles.map((fileInfo) => (
                            <ListItem
                                key={fileInfo.id}
                                sx={{
                                    border: '1px solid',
                                    borderColor: 'divider',
                                    borderRadius: 1,
                                    mb: 1,
                                    bgcolor: 'background.paper',
                                }}
                            >
                                <InsertDriveFileIcon sx={{ mr: 1, color: 'text.secondary' }} />
                                <ListItemText
                                    primary={fileInfo.filename}
                                    secondary={
                                        <Box>
                                            <Typography variant="caption" component="span">
                                                {formatFileSize(fileInfo.fileSize)}
                                            </Typography>
                                            <Chip
                                                label="Ready"
                                                size="small"
                                                color="success"
                                                sx={{ ml: 1 }}
                                            />
                                            <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5 }}>
                                                ID: {fileInfo.fileId.substring(0, 8)}...
                                            </Typography>
                                        </Box>
                                    }
                                />
                                <ListItemSecondaryAction>
                                    <IconButton
                                        edge="end"
                                        aria-label="delete"
                                        onClick={() => handleDelete(fileInfo.id)}
                                        size="small"
                                    >
                                        <DeleteIcon />
                                    </IconButton>
                                </ListItemSecondaryAction>
                            </ListItem>
                        ))}
                    </List>
                </Box>
            )}
        </Box>
    );
};

export default FileUpload;

