import { useState, useCallback, useMemo } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'
import LoadingSpinner from './LoadingSpinner'
import BatchResults from './BatchResults'

// Import sample images
import sampleBourbon from '../assets/sample_bourbon.png'
import sampleBeer from '../assets/sample_beer.png'
import sampleWine from '../assets/sample_wine.png'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Helper: find best matching CSV row for an image filename
const findMatchingCsvRow = (imageName, csvRows, imageIndex = -1) => {
  if (!csvRows || csvRows.length === 0) return null
  
  // 1. Exact match
  const exact = csvRows.find(row => row.filename === imageName)
  if (exact) return exact
  
  // 2. Case-insensitive match
  const lowerName = imageName.toLowerCase()
  const caseInsensitive = csvRows.find(row => row.filename?.toLowerCase() === lowerName)
  if (caseInsensitive) return caseInsensitive
  
  // 3. Partial match (CSV filename contained in image name or vice versa)
  const partial = csvRows.find(row => {
    if (!row.filename) return false
    const csvLower = row.filename.toLowerCase()
    return lowerName.includes(csvLower) || csvLower.includes(lowerName)
  })
  if (partial) return partial
  
  // 4. Match by row order (if same number of rows)
  if (imageIndex >= 0 && imageIndex < csvRows.length) {
    return csvRows[imageIndex]
  }
  
  return null
}

// Sample batch data with images for quick testing
const SAMPLE_LABELS = [
  {
    id: 'bourbon',
    filename: 'sample_bourbon.png',
    image: sampleBourbon,
    category: 'Bourbon',
    data: {
      brand_name: 'OLD TOM DISTILLERY',
      class_type: 'Kentucky Straight Bourbon Whiskey',
      abv_percent: '45',
      net_contents_ml: '750',
      has_warning: true,
    }
  },
  {
    id: 'beer',
    filename: 'sample_beer.png',
    image: sampleBeer,
    category: 'Beer',
    data: {
      brand_name: 'MOUNTAIN BREW CO',
      class_type: 'India Pale Ale',
      abv_percent: '6.8',
      net_contents_ml: '355',
      has_warning: true,
    }
  },
  {
    id: 'wine',
    filename: 'sample_wine.png',
    image: sampleWine,
    category: 'Wine',
    data: {
      brand_name: 'SILVER OAK',
      class_type: 'Cabernet Sauvignon',
      abv_percent: '14.5',
      net_contents_ml: '750',
      has_warning: true,
    }
  },
]

// Legacy lookup for backward compatibility
const SAMPLE_BATCH_DATA = Object.fromEntries(
  SAMPLE_LABELS.map(s => [s.filename, s.data])
)

function BatchVerification() {
  const [images, setImages] = useState([])
  const [csvFile, setCsvFile] = useState(null)
  const [csvData, setCsvData] = useState(null) // Parsed CSV data
  const [imageData, setImageData] = useState({}) // Editable data per image
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [progress, setProgress] = useState(0)
  const [mode, setMode] = useState('upload') // 'upload', 'edit', 'review'

  // Image dropzone
  const onDropImages = useCallback((acceptedFiles, rejectedFiles) => {
    if (rejectedFiles.length > 0) {
      setError('Some files were rejected. Only PNG, JPG, JPEG, WEBP images are allowed.')
    }
    
    const validFiles = acceptedFiles.filter(file => {
      if (file.size > 5 * 1024 * 1024) {
        setError(`${file.name} is too large (max 5MB). It will be skipped.`)
        return false
      }
      return true
    })

    setImages(prev => [...prev, ...validFiles])
    setResults(null)
    // Don't clear csvData - keep it for matching
    
    // Initialize data for new images, using CSV data if available
    setImageData(prev => {
      const newData = { ...prev }
      const existingCount = Object.keys(prev).length
      validFiles.forEach((file, idx) => {
        if (!newData[file.name]) {
          // Try to match with existing CSV data (use index for row-order matching)
          const csvMatch = findMatchingCsvRow(file.name, csvData, existingCount + idx)
          if (csvMatch) {
            newData[file.name] = {
              brand_name: csvMatch.brand_name || '',
              class_type: csvMatch.class_type || '',
              abv_percent: csvMatch.abv_percent || '',
              net_contents_ml: csvMatch.net_contents_ml || '',
              has_warning: csvMatch.has_warning !== 'false' && csvMatch.has_warning !== false,
            }
          } else {
            newData[file.name] = {
              brand_name: '',
              class_type: '',
              abv_percent: '',
              net_contents_ml: '',
              has_warning: true,
            }
          }
        }
      })
      return newData
    })
  }, [csvData])

  const { getRootProps: getImageRootProps, getInputProps: getImageInputProps, isDragActive: isImageDragActive } = useDropzone({
    onDrop: onDropImages,
    accept: {
      'image/png': ['.png'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/webp': ['.webp'],
    },
  })

  // CSV dropzone and parsing
  const onDropCSV = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0]
    if (file) {
      setCsvFile(file)
      setError(null)
      setResults(null)
      
      // Parse CSV
      const reader = new FileReader()
      reader.onload = (e) => {
        try {
          const text = e.target.result
          const parsed = parseCSV(text)
          setCsvData(parsed)
          
          // Update imageData with CSV values using fuzzy matching
          const newImageData = { ...imageData }
          
          // For each uploaded image, try to find matching CSV row
          // Use index for row-order matching when names don't match
          images.forEach((img, index) => {
            const csvMatch = findMatchingCsvRow(img.name, parsed, index)
            if (csvMatch) {
              newImageData[img.name] = {
                brand_name: csvMatch.brand_name || '',
                class_type: csvMatch.class_type || '',
                abv_percent: csvMatch.abv_percent || '',
                net_contents_ml: csvMatch.net_contents_ml || '',
                has_warning: csvMatch.has_warning !== 'false' && csvMatch.has_warning !== false,
              }
            }
          })
          
          // Also store CSV data keyed by original filename (for future image uploads)
          parsed.forEach(row => {
            if (row.filename && !newImageData[row.filename]) {
              newImageData[row.filename] = {
                brand_name: row.brand_name || '',
                class_type: row.class_type || '',
                abv_percent: row.abv_percent || '',
                net_contents_ml: row.net_contents_ml || '',
                has_warning: row.has_warning !== 'false' && row.has_warning !== false,
              }
            }
          })
          
          setImageData(newImageData)
          
          // Show how many images were matched
          const matchedCount = images.filter(img => findMatchingCsvRow(img.name, parsed)).length
          if (images.length > 0 && matchedCount > 0) {
            console.log(`CSV matched ${matchedCount}/${images.length} images`)
          }
        } catch (err) {
          setError('Failed to parse CSV file. Please check the format.')
        }
      }
      reader.readAsText(file)
    }
  }, [imageData, images])

  const { getRootProps: getCsvRootProps, getInputProps: getCsvInputProps, isDragActive: isCsvDragActive } = useDropzone({
    onDrop: onDropCSV,
    accept: {
      'text/csv': ['.csv'],
    },
    maxFiles: 1,
  })

  // Parse CSV string to array of objects
  const parseCSV = (text) => {
    const lines = text.trim().split('\n')
    if (lines.length < 2) return []
    
    const headers = lines[0].split(',').map(h => h.trim().toLowerCase())
    
    return lines.slice(1).map(line => {
      const values = line.split(',').map(v => v.trim())
      const row = {}
      headers.forEach((header, i) => {
        row[header] = values[i] || ''
      })
      return row
    })
  }

  // Calculate matching status
  const matchStatus = useMemo(() => {
    const imageNames = new Set(images.map(img => img.name))
    const csvNames = csvData ? new Set(csvData.map(row => row.filename)) : new Set()
    
    return {
      matched: images.filter(img => csvNames.has(img.name) || imageData[img.name]?.brand_name),
      unmatchedImages: images.filter(img => !csvNames.has(img.name) && !imageData[img.name]?.brand_name),
      unmatchedCsv: csvData ? csvData.filter(row => !imageNames.has(row.filename)) : [],
    }
  }, [images, csvData, imageData])

  const removeImage = (index) => {
    const img = images[index]
    setImages(prev => prev.filter((_, i) => i !== index))
    setImageData(prev => {
      const newData = { ...prev }
      delete newData[img.name]
      return newData
    })
  }

  const updateImageData = (filename, field, value) => {
    setImageData(prev => ({
      ...prev,
      [filename]: {
        ...prev[filename],
        [field]: value,
      }
    }))
  }

  const handleVerifyBatch = async () => {
    if (images.length === 0) {
      setError('Please upload at least one label image.')
      return
    }

    // Check if all images have at least brand_name
    const missingData = images.filter(img => !imageData[img.name]?.brand_name)
    if (missingData.length > 0) {
      setError(`Missing brand name for: ${missingData.map(img => img.name).join(', ')}`)
      return
    }

    setLoading(true)
    setError(null)
    setResults(null)
    setProgress(0)

    // Generate CSV from imageData
    const csvContent = generateCSV()
    const csvBlob = new Blob([csvContent], { type: 'text/csv' })
    const generatedCsvFile = new File([csvBlob], 'batch_data.csv', { type: 'text/csv' })

    const submitData = new FormData()
    
    images.forEach(img => {
      submitData.append('images', img)
    })
    
    submitData.append('csv_file', generatedCsvFile)

    try {
      const response = await axios.post(`${API_URL}/api/v1/verify/batch`, submitData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 300000,
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          setProgress(percentCompleted)
        },
      })

      if (response.data.success) {
        setResults(response.data)
        setMode('upload')
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

  const generateCSV = () => {
    const headers = ['filename', 'brand_name', 'class_type', 'abv_percent', 'net_contents_ml', 'has_warning']
    const rows = images.map(img => {
      const data = imageData[img.name] || {}
      return [
        img.name,
        data.brand_name || '',
        data.class_type || '',
        data.abv_percent || '',
        data.net_contents_ml || '',
        data.has_warning !== false ? 'true' : 'false',
      ].join(',')
    })
    return [headers.join(','), ...rows].join('\n')
  }

  const downloadSmartTemplate = () => {
    const csvContent = generateCSV()
    const blob = new Blob([csvContent], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'batch_template.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  const downloadGenericTemplate = () => {
    const csvContent = `filename,brand_name,class_type,abv_percent,net_contents_ml,has_warning
label_01.png,BRAND NAME,Beverage Type,45,750,true
label_02.png,ANOTHER BRAND,Another Type,40,1000,true`
    
    const blob = new Blob([csvContent], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'batch_template.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  const loadSampleData = (filename) => {
    if (SAMPLE_BATCH_DATA[filename]) {
      setImageData(prev => ({
        ...prev,
        [filename]: { ...SAMPLE_BATCH_DATA[filename] }
      }))
    }
  }

  const handleClear = () => {
    setImages([])
    setCsvFile(null)
    setCsvData(null)
    setImageData({})
    setResults(null)
    setError(null)
    setMode('upload')
  }

  // Load all sample labels for quick batch testing
  const loadAllSamples = async () => {
    try {
      const sampleFiles = []
      const sampleData = {}
      
      for (const sample of SAMPLE_LABELS) {
        // Fetch the image and convert to File object
        const response = await fetch(sample.image)
        const blob = await response.blob()
        const file = new File([blob], sample.filename, { type: 'image/png' })
        sampleFiles.push(file)
        sampleData[sample.filename] = sample.data
      }
      
      setImages(sampleFiles)
      setImageData(sampleData)
      setResults(null)
      setError(null)
    } catch (err) {
      console.error('Failed to load sample images:', err)
      setError('Failed to load sample images. Please try again.')
    }
  }

  const allImagesHaveData = images.every(img => imageData[img.name]?.brand_name)

  return (
    <div className="batch-verification">
      <div className="batch-info">
        <h2>Batch Verification</h2>
        <p>Upload multiple label images and provide application data for each.</p>
      </div>

      {/* Quick Test Section */}
      <div className="quick-test-section">
        <h3>🚀 Try Batch Verification in less than 10 seconds</h3>
        <p>Load all 3 sample labels with pre-filled application data:</p>
        <div className="sample-cards">
          {SAMPLE_LABELS.map(sample => (
            <div key={sample.id} className="sample-card-preview">
              <img src={sample.image} alt={sample.data.brand_name} className="sample-card-image" />
              <div className="sample-card-info">
                <span className="sample-card-category">{sample.category}</span>
                <span className="sample-card-name">{sample.data.brand_name}</span>
              </div>
            </div>
          ))}
        </div>
        <button 
          className="btn btn-primary" 
          onClick={loadAllSamples}
          disabled={loading}
          style={{ marginTop: '1rem' }}
        >
          📦 Load All Sample Labels
        </button>
      </div>

      <div className="section-divider">
        <span>or upload your own</span>
      </div>

      {/* Step 1: Upload Images */}
      <div className="batch-section">
        <h3>Step 1: Upload Label Images</h3>
        <div
          {...getImageRootProps()}
          className={`dropzone batch-dropzone ${isImageDragActive ? 'active' : ''}`}
        >
          <input {...getImageInputProps()} />
          <div className="dropzone-content">
            <div className="dropzone-icon">📷</div>
            <p>Drag & drop label images here</p>
            <p className="dropzone-subtext">or click to select files</p>
            <p className="dropzone-formats">PNG, JPG, JPEG, WEBP (max 5MB each)</p>
          </div>
        </div>
      </div>

      {/* Step 2: Provide Data - shows after images uploaded */}
      {images.length > 0 && (
        <div className="batch-section">
          <h3>Step 2: Provide Application Data</h3>
          <p className="section-hint">
            Choose how to provide data for your {images.length} image(s):
          </p>
          
          <div className="data-options">
            <div className="data-option">
              <h4>Option A: Upload CSV</h4>
              <p>Upload a CSV file with application data matching your image filenames.</p>
              <div
                {...getCsvRootProps()}
                className={`dropzone mini-dropzone ${isCsvDragActive ? 'active' : ''} ${csvFile ? 'has-file' : ''}`}
              >
                <input {...getCsvInputProps()} />
                {csvFile ? (
                  <span className="csv-selected">✓ {csvFile.name}</span>
                ) : (
                  <span>Drop CSV here or click to select</span>
                )}
              </div>
              <button className="btn btn-link" onClick={downloadGenericTemplate}>
                📥 Download blank template
              </button>
              {images.length > 0 && (
                <button className="btn btn-link" onClick={downloadSmartTemplate}>
                  📥 Download template with your filenames
                </button>
              )}
            </div>
            
            <div className="data-option-divider">OR</div>
            
            <div className="data-option">
              <h4>Option B: Enter Data Below</h4>
              <p>Fill in the application data directly for each image.</p>
            </div>
          </div>

          {/* Image-by-image data entry */}
          <div className="image-data-table">
            <table className="data-entry-table">
              <thead>
                <tr>
                  <th>Image</th>
                  <th>Brand Name <span className="required">*</span></th>
                  <th>Class/Type</th>
                  <th>ABV %</th>
                  <th>mL</th>
                  <th>Warning</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {images.map((img, index) => (
                  <tr key={img.name} className={imageData[img.name]?.brand_name ? 'has-data' : 'missing-data'}>
                    <td className="image-cell">
                      <span className="image-filename" title={img.name}>
                        {img.name.length > 20 ? img.name.slice(0, 17) + '...' : img.name}
                      </span>
                      {SAMPLE_BATCH_DATA[img.name] && (
                        <button 
                          className="btn-mini btn-sample-load"
                          onClick={() => loadSampleData(img.name)}
                          title="Load sample data"
                        >
                          Load sample
                        </button>
                      )}
                    </td>
                    <td>
                      <input
                        type="text"
                        value={imageData[img.name]?.brand_name || ''}
                        onChange={(e) => updateImageData(img.name, 'brand_name', e.target.value)}
                        placeholder="Required"
                        className="table-input"
                      />
                    </td>
                    <td>
                      <input
                        type="text"
                        value={imageData[img.name]?.class_type || ''}
                        onChange={(e) => updateImageData(img.name, 'class_type', e.target.value)}
                        placeholder="e.g., Bourbon"
                        className="table-input"
                      />
                    </td>
                    <td>
                      <input
                        type="number"
                        value={imageData[img.name]?.abv_percent || ''}
                        onChange={(e) => updateImageData(img.name, 'abv_percent', e.target.value)}
                        placeholder="e.g., 45"
                        className="table-input table-input-small"
                        step="0.1"
                      />
                    </td>
                    <td>
                      <input
                        type="number"
                        value={imageData[img.name]?.net_contents_ml || ''}
                        onChange={(e) => updateImageData(img.name, 'net_contents_ml', e.target.value)}
                        placeholder="e.g., 750"
                        className="table-input table-input-small"
                      />
                    </td>
                    <td>
                      <input
                        type="checkbox"
                        checked={imageData[img.name]?.has_warning !== false}
                        onChange={(e) => updateImageData(img.name, 'has_warning', e.target.checked)}
                        className="table-checkbox"
                      />
                    </td>
                    <td>
                      <button
                        className="btn-remove"
                        onClick={() => removeImage(index)}
                        aria-label={`Remove ${img.name}`}
                      >
                        ×
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Status indicator */}
          <div className="data-status">
            {allImagesHaveData ? (
              <span className="status-ready">✓ All images have required data</span>
            ) : (
              <span className="status-incomplete">
                ⚠️ {images.filter(img => !imageData[img.name]?.brand_name).length} image(s) missing brand name
              </span>
            )}
          </div>
        </div>
      )}

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
          disabled={loading || images.length === 0 || !allImagesHaveData}
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
