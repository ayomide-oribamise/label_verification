# Label Verification Frontend

React frontend for the AI-powered alcohol label verification application.

## Quick Start

```bash
# Install dependencies
npm install

# Copy environment template
cp .env.example .env

# Start development server
npm run dev
```

Open http://localhost:5173

## Configuration

Create a `.env` file with:

```
VITE_API_URL=http://localhost:8000
```

For production, set this to the deployed backend URL.

## Build

```bash
# Production build
npm run build

# Preview build locally
npm run preview
```

Output is in the `dist/` folder.

## Features

### Single Label Verification
- Drag-and-drop image upload
- Load sample labels for quick testing
- Application data form with validation
- Real-time verification results
- Field-by-field status display (match/review/mismatch)

### Batch Verification
- Multiple image upload
- CSV file upload with application data
- Auto-matching of images to CSV rows by filename
- Download CSV template with uploaded filenames
- Progress tracking and results summary
- Export results to CSV

### Sample Data
Pre-loaded sample labels and application data for demonstration:
- Bourbon (Old Tom Distillery)
- Wine (Silver Oak Cabernet Sauvignon)  
- Beer (Mountain Brew Co IPA)

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── SingleVerification.jsx   # Single label verification mode
│   │   ├── BatchVerification.jsx    # Batch verification mode
│   │   ├── ApplicationForm.jsx      # Application data form
│   │   ├── VerificationResults.jsx  # Single verification results
│   │   ├── BatchResults.jsx         # Batch results display
│   │   └── LoadingSpinner.jsx       # Loading indicator
│   ├── assets/                      # Static assets and sample images
│   ├── App.jsx                      # Main application component
│   ├── App.css                      # Application styles
│   ├── main.jsx                     # Entry point
│   └── index.css                    # Global styles
├── public/
│   └── samples/                     # Sample data for testing
├── .env.example                     # Environment template
├── package.json                     # Dependencies
└── vite.config.js                   # Vite configuration
```

## Deployment

### Azure Static Web Apps

1. Build the application:
   ```bash
   npm run build
   ```

2. Deploy using Azure CLI or GitHub Actions

3. Set environment variable in Azure portal:
   - `VITE_API_URL` = backend Container App URL

### Manual Deployment

1. Run `npm run build`
2. Deploy contents of `dist/` folder to any static hosting service

## Design

- Clean, accessible interface
- High contrast colors for readability
- Responsive layout for desktop and tablet
- Clear status indicators (green/yellow/red)
- Loading states and error handling
