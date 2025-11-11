import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import AnalyzerPage from './pages/AnalyzerPage';
import HistoryPage from './pages/HistoryPage';
import SettingsPage from './pages/SettingsPage';

function App() {
    return (
        <Layout>
            <Routes>
                <Route path="/" element={<AnalyzerPage />} />
                <Route path="/history" element={<HistoryPage />} />
                <Route path="/settings" element={<SettingsPage />} />
            </Routes>
        </Layout>
    );
}

export default App;

