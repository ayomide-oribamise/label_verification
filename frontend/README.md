# Label Verification Frontend

React frontend for the AI-powered alcohol label verification tool.

## Quick Start

### Prerequisites

- Node.js 18+
- npm or yarn

### Development

1. Install dependencies:
```bash
npm install
```

2. Copy environment file:
```bash
cp .env.example .env
```

3. Update `.env` with your backend URL:
```
VITE_API_URL=http://localhost:8000
```

4. Start development server:
```bash
npm run dev
```

5. Open http://localhost:5173

### Building for Production

```bash
npm run build
```

Output will be in the `dist/` folder.

### Preview Production Build

```bash
npm run preview
```

## Features

### Single Label Verification
- Drag-and-drop image upload
- Form fields for application data
- Real-time verification results
- Field-by-field status display

### Batch Verification
- Multiple image upload
- CSV file with application data
- Parallel processing
- Export results to CSV

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── SingleVerification.jsx  # Single label mode
│   │   ├── BatchVerification.jsx   # Batch mode
│   │   ├── ApplicationForm.jsx     # Form fields
│   │   ├── VerificationResults.jsx # Single results
│   │   ├── BatchResults.jsx        # Batch results
│   │   └── LoadingSpinner.jsx      # Loading indicator
│   ├── App.jsx                     # Main app
│   ├── App.css                     # Styles
│   ├── main.jsx                    # Entry point
│   └── index.css                   # Reset styles
├── .env.example                    # Environment template
├── package.json
└── vite.config.js
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Backend API URL | `http://localhost:8000` |

## Deployment

### Vercel (Recommended)

1. Connect your GitHub repository
2. Set environment variable:
   - `VITE_API_URL` = your backend URL
3. Deploy

### Manual

1. Build: `npm run build`
2. Deploy `dist/` folder to any static hosting

## Design Principles

- **Accessibility**: Large fonts, high contrast, keyboard navigation
- **Simplicity**: Clean UI, obvious actions
- **Responsive**: Works on desktop and tablet
- **Error Handling**: Clear error messages and guidance
