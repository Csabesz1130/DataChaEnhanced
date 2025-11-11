# Signal Analyzer Frontend

React-based web frontend for the Signal Analyzer application.

## Prerequisites

- Node.js 16+ and npm
- Backend API running on `http://localhost:5000` (or configure `REACT_APP_API_URL`)

## Setup

1. Install dependencies:
```bash
npm install
```

2. Configure environment (optional):
```bash
cp .env.example .env
# Edit .env with your API URL if needed
```

3. Start development server:
```bash
npm start
```

The app will open at `http://localhost:3001` (or the port specified in `.env`).

## Available Scripts

- `npm start` - Start development server
- `npm build` - Build for production
- `npm test` - Run tests
- `npm run eject` - Eject from Create React App (irreversible)

## Project Structure

```
src/
├── components/     # Reusable UI components
├── pages/         # Page components (Analyzer, History, Settings)
├── contexts/      # React Context providers (Theme, Notifications)
├── services/      # API client and services
├── utils/         # Utility functions
└── App.js         # Main app component with routing
```

## Features

- **Modern UI**: Material-UI v5 with dark/light theme support
- **Responsive Layout**: Drawer navigation with mobile support
- **Analysis Workflow**: Stepper-based parameter configuration
- **History Tracking**: View and reload past analysis runs
- **Interactive Plots**: Plotly.js charts with zoom, export, and curve toggles

## Development Notes

- Uses `react-app-rewired` for webpack configuration overrides
- Axios is patched post-install to fix ESM import issues
- Theme preference is saved to localStorage
- All API calls include retry logic with exponential backoff

