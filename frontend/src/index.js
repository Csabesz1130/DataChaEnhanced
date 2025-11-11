import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import './index.css';
import App from './App';
import ErrorBoundary from './components/ErrorBoundary';
import { NotificationProvider } from './contexts/NotificationContext';
import { ThemeContextProvider } from './contexts/ThemeContext';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
    <React.StrictMode>
        <ErrorBoundary>
            <ThemeContextProvider>
                <NotificationProvider>
                    <BrowserRouter>
                        <App />
                    </BrowserRouter>
                </NotificationProvider>
            </ThemeContextProvider>
        </ErrorBoundary>
    </React.StrictMode>
);

