import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'
import LoadingSpinner from './LoadingSpinner'
import BatchResults from './BatchResults'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function BatchVerification() {
  const [images, setImages] = useState([])
  const [csvFile, setCsvFile] = useState(null)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [progress, setProgress] = useState(0)

  // Image dropzone
  const onDropImages = useCallback((acceptedFiles, rejectedFiles) => {
    if (rejectedFiles.length > 0) {
      setError('Some files were rejected. Only PNG, JPG, JPEG, WEBP images are allowed.')
    }
    
    // Filter and validate files
    const validFiles = acceptedFiles.filter(file => {
      if (file.size > 5 * 1024 * 1024) {
        setError(`${file.name} is too large (max 5MB). It will be skipped.`)
        return false
      }
      return true
    })

    setImages(prev => [...prev, ...validFiles])
    setResults(null)
  }, [])

  const { getRootProps: getImageRootProps, getInputProps: getImageInputProps, isDragActive: isImageDragActive } = useDropzone({
    onDrop: onDropImages,
    accept: {
      'image/png': ['.png'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/webp': ['.webp'],
    },
  })

  // CSV dropzone
  const onDropCSV = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0]
    if (file) {
      setCsvFile(file)
      setError(null)
      setResults(null)
    }
  }, [])

  const { getRootProps: getCsvRootProps, getInputProps: getCsvInputProps, isDragActive: isCsvDragActive } = useDropzone({
    onDrop: onDropCSV,
    accept: {
      'text/csv': ['.csv'],
    },
    maxFiles: 1,
  })

  const removeImage = (index) => {
    setImages(prev => prev.filter((_, i) => i !== index))
  }

  const handleVerifyBatch = async () => {
    if (images.length === 0) {
      setError('Please upload at least one label image.')
      return
    }

    if (!csvFile) {
      setError('Please upload a CSV file with application data.')
      return
    }

    setLoading(true)
    setError(null)
    setResults(null)
    setProgress(0)

    const submitData = new FormData()
    
    // Add all images
    images.forEach(img => {
      submitData.append('images', img)
    })
    
    // Add CSV file
    submitData.append('csv_file', csvFile)

    try {
      const response = await axios.post(`${API_URL}/api/v1/verify/batch`, submitData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 300000, // 5 minute timeout for batch
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          setProgress(percentCompleted)
        },
      })

      if (response.data.success) {
        setResults(response.data)
      } else {
        setError(response.data.error || 'Batch verification failed.')
      }
    } catch (err) {
      console.error('Batch verification error:', err)
      if (err.code === 'ECONNABORTED') {
        setError('Request timed out. The batch may be too large.')
      } else if (err.response?.data?.detail) {
        setError(err.response.data.detail)
      } else {
        setError('Unable to process batch. Please check the connection and try again.')
      }
    } finally {
      setLoading(false)
      setProgress(0)
    }
  }

  const handleClear = () => {
    setImages([])
    setCsvFile(null)
    setResults(null)
    setError(null)
  }

  const downloadTemplate = () => {
    const csvContent = `filename,brand_name,class_type,abv_percent,net_contents_ml,has_warning
label_01.png,OLD TOM DISTILLERY,Kentucky Straight Bourbon Whiskey,45,750,true
label_02.png,JACK DANIELS,Tennessee Whiskey,40,1000,true`
    
    const blob = new Blob([csvContent], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'batch_template.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="batch-verification">
      <div className="batch-info">
        <h2>Batch Verification</h2>
        <p>Upload multiple label images and a CSV file with the expected application data.</p>
        <button className="btn btn-link" onClick={downloadTemplate}>
          📥 Download CSV Template
        </button>
      </div>

      <div className="batch-layout">
        {/* Images upload */}
        <div className="batch-section">
          <h3>1. Upload Label Images</h3>
          <div
            {...getImageRootProps()}
            className={`dropzone batch-dropzone ${isImageDragActive ? 'active' : ''}`}
          >
            <input {...getImageInputProps()} />
            <div className="dropzone-content">
              <div className="dropzone-icon">📷</div>
              <p>Drag & drop label images here</p>
              <p className="dropzone-subtext">or click to select files</p>
            </div>
          </div>

          {images.length > 0 && (
            <div className="image-list">
              <h4>{images.length} image(s) selected:</h4>
              <ul>
                {images.map((img, index) => (
                  <li key={index}>
                    <span className="image-name">{img.name}</span>
                    <span className="image-size">({(img.size / 1024).toFixed(1)} KB)</span>
                    <button
                      className="btn-remove"
                      onClick={() => removeImage(index)}
                      aria-label={`Remove ${img.name}`}
                    >
                      ×
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* CSV upload */}
        <div className="batch-section">
          <h3>2. Upload CSV Data</h3>
          <div
            {...getCsvRootProps()}
            className={`dropzone batch-dropzone ${isCsvDragActive ? 'active' : ''} ${csvFile ? 'has-file' : ''}`}
          >
            <input {...getCsvInputProps()} />
            {csvFile ? (
              <div className="csv-preview">
                <span className="csv-icon">📄</span>
                <span className="csv-name">{csvFile.name}</span>
                <p className="change-hint">Click or drag to change</p>
              </div>
            ) : (
              <div className="dropzone-content">
                <div className="dropzone-icon">📄</div>
                <p>Drag & drop CSV file here</p>
                <p className="dropzone-subtext">or click to select</p>
              </div>
            )}
          </div>

          <div className="csv-format">
            <h4>Required CSV columns:</h4>
            <code>filename, brand_name, class_type, abv_percent, net_contents_ml, has_warning</code>
          </div>
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="error-banner" role="alert">
          <span className="error-icon">⚠️</span>
          <span>{error}</span>
          <button className="error-close" onClick={() => setError(null)}>×</button>
        </div>
      )}

      {/* Progress bar */}
      {loading && progress > 0 && (
        <div className="progress-container">
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progress}%` }}></div>
          </div>
          <span className="progress-text">Uploading... {progress}%</span>
        </div>
      )}

      {/* Action buttons */}
      <div className="action-buttons">
        <button
          className="btn btn-primary btn-large"
          onClick={handleVerifyBatch}
          disabled={loading || images.length === 0 || !csvFile}
        >
          {loading ? <LoadingSpinner /> : `✓ Verify ${images.length} Label(s)`}
        </button>
        <button
          className="btn btn-secondary"
          onClick={handleClear}
          disabled={loading}
        >
          Clear All
        </button>
      </div>

      {/* Results */}
      {results && <BatchResults results={results} />}
    </div>
  )
}

export default BatchVerification
