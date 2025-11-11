import React from 'react';
import {
    Box,
    Typography,
    Paper,
    Grid,
    Switch,
    FormControlLabel,
    Divider,
    List,
    ListItem,
    ListItemText,
    ListItemSecondaryAction,
} from '@mui/material';
import { useTheme } from '../contexts/ThemeContext';

function SettingsPage() {
    const { mode, toggleTheme } = useTheme();

    return (
        <Box>
            <Typography variant="h4" gutterBottom>
                Settings
            </Typography>

            <Grid container spacing={3} sx={{ mt: 2 }}>
                <Grid item xs={12} md={6}>
                    <Paper sx={{ p: 3 }}>
                        <Typography variant="h6" gutterBottom>
                            Appearance
                        </Typography>
                        <Divider sx={{ my: 2 }} />
                        <List>
                            <ListItem>
                                <ListItemText
                                    primary="Dark Mode"
                                    secondary="Toggle between light and dark theme"
                                />
                                <ListItemSecondaryAction>
                                    <FormControlLabel
                                        control={
                                            <Switch
                                                checked={mode === 'dark'}
                                                onChange={toggleTheme}
                                            />
                                        }
                                        label=""
                                    />
                                </ListItemSecondaryAction>
                            </ListItem>
                        </List>
                    </Paper>
                </Grid>

                <Grid item xs={12} md={6}>
                    <Paper sx={{ p: 3 }}>
                        <Typography variant="h6" gutterBottom>
                            About
                        </Typography>
                        <Divider sx={{ my: 2 }} />
                        <Typography variant="body2" color="text.secondary" paragraph>
                            Signal Analyzer Web Application
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                            Version 1.0.0
                        </Typography>
                    </Paper>
                </Grid>
            </Grid>
        </Box>
    );
}

export default SettingsPage;


