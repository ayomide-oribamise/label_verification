import { useState } from 'react'
import './App.css'
import SingleVerification from './components/SingleVerification'
import BatchVerification from './components/BatchVerification'

function App() {
  const [mode, setMode] = useState('single') // 'single' or 'batch'

  return (
    <div className="app">
      <header className="app-header">
        <h1>🍷 Label Verification Tool</h1>
        <p className="subtitle">AI-powered alcohol label compliance checker</p>
      </header>

      <nav className="mode-tabs" role="tablist">
        <button
          role="tab"
          aria-selected={mode === 'single'}
          className={`tab ${mode === 'single' ? 'active' : ''}`}
          onClick={() => setMode('single')}
        >
          Single Label
        </button>
        <button
          role="tab"
          aria-selected={mode === 'batch'}
          className={`tab ${mode === 'batch' ? 'active' : ''}`}
          onClick={() => setMode('batch')}
        >
          Batch Upload
        </button>
      </nav>

      <main className="main-content">
        {mode === 'single' ? <SingleVerification /> : <BatchVerification />}
      </main>

      <footer className="app-footer">
        <p>Prototype for TTB Label Compliance Review</p>
      </footer>
    </div>
  )
}

export default App
