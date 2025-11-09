/**
 * Global error handler for the application
 * Provides user-friendly error messages and retry logic
 */

export class AppError extends Error {
    constructor(message, code, originalError = null) {
        super(message);
        this.name = 'AppError';
        this.code = code;
        this.originalError = originalError;
        this.timestamp = new Date().toISOString();
    }
}

/**
 * Map error codes to user-friendly messages
 */
const ERROR_MESSAGES = {
    NETWORK_ERROR: 'Network connection failed. Please check your internet connection and try again.',
    TIMEOUT: 'Request timed out. The server may be busy. Please try again.',
    SERVER_ERROR: 'Server error occurred. Please try again later or contact support.',
    NOT_FOUND: 'The requested resource was not found.',
    UNAUTHORIZED: 'You are not authorized to perform this action.',
    VALIDATION_ERROR: 'Invalid input. Please check your parameters and try again.',
    FILE_TOO_LARGE: 'File is too large. Maximum size is 50MB.',
    INVALID_FILE_TYPE: 'Invalid file type. Please upload .atf, .txt, or .csv files.',
    ANALYSIS_FAILED: 'Analysis failed. Please check your parameters and try again.',
    UNKNOWN_ERROR: 'An unexpected error occurred. Please try again.',
};

/**
 * Extract user-friendly error message from error object
 */
export const getErrorMessage = (error) => {
    if (error instanceof AppError) {
        return ERROR_MESSAGES[error.code] || error.message;
    }

    if (error.response) {
        // Axios error with response
        const status = error.response.status;
        const data = error.response.data;

        if (data?.error || data?.message) {
            return data.error || data.message;
        }

        switch (status) {
            case 400:
                return ERROR_MESSAGES.VALIDATION_ERROR;
            case 401:
                return ERROR_MESSAGES.UNAUTHORIZED;
            case 404:
                return ERROR_MESSAGES.NOT_FOUND;
            case 413:
                return ERROR_MESSAGES.FILE_TOO_LARGE;
            case 500:
            case 502:
            case 503:
                return ERROR_MESSAGES.SERVER_ERROR;
            default:
                return ERROR_MESSAGES.UNKNOWN_ERROR;
        }
    }

    if (error.request) {
        // Axios error without response (network error)
        if (error.code === 'ECONNABORTED') {
            return ERROR_MESSAGES.TIMEOUT;
        }
        return ERROR_MESSAGES.NETWORK_ERROR;
    }

    if (error.message) {
        return error.message;
    }

    return ERROR_MESSAGES.UNKNOWN_ERROR;
};

/**
 * Check if error is retryable
 */
export const isRetryable = (error) => {
    if (error instanceof AppError) {
        return ['NETWORK_ERROR', 'TIMEOUT', 'SERVER_ERROR'].includes(error.code);
    }

    if (error.response) {
        const status = error.response.status;
        // Retry on server errors and timeouts
        return status >= 500 || status === 408 || status === 429;
    }

    if (error.request) {
        // Network errors are retryable
        return true;
    }

    return false;
};

/**
 * Retry function with exponential backoff
 */
export const retryWithBackoff = async (fn, maxRetries = 3, initialDelay = 1000) => {
    let lastError;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
            return await fn();
        } catch (error) {
            lastError = error;

            if (!isRetryable(error) || attempt === maxRetries) {
                throw error;
            }

            // Exponential backoff: delay = initialDelay * 2^attempt
            const delay = initialDelay * Math.pow(2, attempt);
            await new Promise((resolve) => setTimeout(resolve, delay));
        }
    }

    throw lastError;
};

/**
 * Log error for debugging
 */
export const logError = (error, context = {}) => {
    const errorInfo = {
        message: error.message,
        code: error.code || 'UNKNOWN',
        stack: error.stack,
        timestamp: new Date().toISOString(),
        context,
    };

    // In production, send to error tracking service (e.g., Sentry)
    if (process.env.NODE_ENV === 'production') {
        // TODO: Integrate with error tracking service
        console.error('Error logged:', errorInfo);
    } else {
        console.error('Error:', errorInfo);
    }

    return errorInfo;
};

/**
 * Handle error with user-friendly message and logging
 */
export const handleError = (error, context = {}) => {
    const userMessage = getErrorMessage(error);
    logError(error, context);

    return {
        message: userMessage,
        code: error.code || 'UNKNOWN',
        retryable: isRetryable(error),
        originalError: error,
    };
};

