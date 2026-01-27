import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'
import ApplicationForm from './ApplicationForm'
import VerificationResults from './VerificationResults'
import LoadingSpinner from './LoadingSpinner'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Sample application data for quick testing
const SAMPLE_APPLICATIONS = [
  {
    id: 'old-tom',
    name: 'OLD TOM DISTILLERY',
    data: {
      brand_name: 'OLD TOM DISTILLERY',
      class_type: 'Kentucky Straight Bourbon Whiskey',
      abv_percent: '45',
      net_contents_ml: '750',
      has_warning: true,
    }
  },
  {
    id: 'silver-oak',
    name: 'SILVER OAK VINEYARDS',
    data: {
      brand_name: 'SILVER OAK',
      class_type: 'Cabernet Sauvignon',
      abv_percent: '14.5',
      net_contents_ml: '750',
      has_warning: true,
    }
  },
  {
    id: 'mountain-brew',
    name: 'MOUNTAIN BREW CO',
    data: {
      brand_name: 'MOUNTAIN BREW',
      class_type: 'India Pale Ale',
      abv_percent: '6.5',
      net_contents_ml: '355',
      has_warning: true,
    }
  },
]

function SingleVerification() {
  const [image, setImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [formData, setFormData] = useState({
    brand_name: '',
    class_type: '',
    abv_percent: '',
    net_contents_ml: '',
    has_warning: true,
  })
  const [selectedSample, setSelectedSample] = useState(null)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    if (rejectedFiles.length > 0) {
      setError('Invalid file type. Please upload PNG, JPG, JPEG, or WEBP images.')
      return
    }
    
    const file = acceptedFiles[0]
    if (file) {
      // Check file size (max 5MB)
      if (file.size > 5 * 1024 * 1024) {
        setError('Image too large. Maximum size is 5MB. Please resize or compress the image.')
        return
      }
      
      setImage(file)
      setImagePreview(URL.createObjectURL(file))
      setError(null)
      setResults(null)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/png': ['.png'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/webp': ['.webp'],
    },
    maxFiles: 1,
  })

  const handleFormChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }))
    setSelectedSample(null) // Clear sample selection when manually editing
  }

  const loadSampleApplication = (sample) => {
    setFormData(sample.data)
    setSelectedSample(sample.id)
    setError(null)
  }

  const handleVerify = async () => {
    if (!image) {
      setError('Please upload a label image first.')
      return
    }

    if (!formData.brand_name.trim()) {
      setError('Brand name is required.')
      return
    }

    setLoading(true)
    setError(null)
    setResults(null)

    const submitData = new FormData()
    submitData.append('image', image)
    submitData.append('brand_name', formData.brand_name)
    
    if (formData.class_type) {
      submitData.append('class_type', formData.class_type)
    }
    if (formData.abv_percent) {
      submitData.append('abv_percent', parseFloat(formData.abv_percent))
    }
    if (formData.net_contents_ml) {
      submitData.append('net_contents_ml', parseFloat(formData.net_contents_ml))
    }
    submitData.append('has_warning', formData.has_warning)

    try {
      const response = await axios.post(`${API_URL}/api/v1/verify`, submitData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000, // 30 second timeout
      })

      if (response.data.success) {
        setResults(response.data)
      } else {
        setError(response.data.error || 'Verification failed. Please try again.')
      }
    } catch (err) {
      console.error('Verification error:', err)
      if (err.code === 'ECONNABORTED') {
        setError('Request timed out. The server may be busy. Please try again.')
      } else if (err.response?.data?.error) {
        setError(err.response.data.error)
      } else if (err.response?.data?.detail) {
        setError(err.response.data.detail)
      } else {
        setError('Unable to connect to the verification server. Please check the connection.')
      }
    } finally {
      setLoading(false)
    }
  }

  const handleClear = () => {
    setImage(null)
    setImagePreview(null)
    setFormData({
      brand_name: '',
      class_type: '',
      abv_percent: '',
      net_contents_ml: '',
      has_warning: true,
    })
    setSelectedSample(null)
    setResults(null)
    setError(null)
  }

  return (
    <div className="single-verification">
      <div className="verification-layout">
        {/* Left side: Image upload */}
        <div className="upload-section">
          <h2>1. Upload Label Image</h2>
          
          <div
            {...getRootProps()}
            className={`dropzone ${isDragActive ? 'active' : ''} ${imagePreview ? 'has-image' : ''}`}
          >
            <input {...getInputProps()} />
            
            {imagePreview ? (
              <div className="image-preview">
                <img src={imagePreview} alt="Label preview" />
                <p className="change-hint">Click or drag to change image</p>
              </div>
            ) : (
              <div className="dropzone-content">
                <div className="dropzone-icon">📷</div>
                <p className="dropzone-text">
                  {isDragActive
                    ? 'Drop the label image here...'
                    : 'Drag & drop a label image here'}
                </p>
                <p className="dropzone-subtext">or click to select a file</p>
                <p className="dropzone-formats">PNG, JPG, JPEG, WEBP (max 5MB)</p>
              </div>
            )}
          </div>
        </div>

        {/* Right side: Form */}
        <div className="form-section">
          <h2>2. Enter Application Data</h2>
          
          {/* Sample Application Selector */}
          <div className="sample-selector">
            <p className="sample-label">Quick load sample application:</p>
            <div className="sample-buttons">
              {SAMPLE_APPLICATIONS.map(sample => (
                <button
                  key={sample.id}
                  className={`btn btn-sample ${selectedSample === sample.id ? 'active' : ''}`}
                  onClick={() => loadSampleApplication(sample)}
                  disabled={loading}
                  title={`Load ${sample.name} application data`}
                >
                  {sample.name}
                </button>
              ))}
            </div>
            <p className="sample-hint">Or enter your own application data below:</p>
          </div>

          <ApplicationForm
            formData={formData}
            onChange={handleFormChange}
            disabled={loading}
          />
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

      {/* Action buttons */}
      <div className="action-buttons">
        <button
          className="btn btn-primary btn-large"
          onClick={handleVerify}
          disabled={loading || !image}
        >
          {loading ? <LoadingSpinner /> : '✓ Verify Label'}
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
      {results && <VerificationResults results={results} />}
    </div>
  )
}

export default SingleVerification
