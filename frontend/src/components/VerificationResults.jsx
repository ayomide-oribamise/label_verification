function VerificationResults({ results }) {
  if (!results || !results.result) {
    return null
  }

  const { result, extracted } = results
  const { overall_status, fields, summary, processing_time_ms } = result

  const getStatusIcon = (status) => {
    switch (status) {
      case 'match':
        return '✅'
      case 'review':
        return '⚠️'
      case 'mismatch':
      case 'not_found':
        return '❌'
      default:
        return '❓'
    }
  }

  const getStatusClass = (status) => {
    switch (status) {
      case 'match':
        return 'status-match'
      case 'review':
        return 'status-review'
      case 'mismatch':
      case 'not_found':
        return 'status-mismatch'
      default:
        return ''
    }
  }

  const getOverallStatusText = (status) => {
    switch (status) {
      case 'match':
        return 'All Fields Verified'
      case 'review':
        return 'Review Recommended'
      case 'mismatch':
        return 'Verification Failed'
      default:
        return 'Unknown Status'
    }
  }

  return (
    <div className="verification-results">
      <h2>Verification Results</h2>

      {/* Overall status banner */}
      <div className={`overall-status ${getStatusClass(overall_status)}`}>
        <span className="status-icon">{getStatusIcon(overall_status)}</span>
        <div className="status-content">
          <span className="status-title">{getOverallStatusText(overall_status)}</span>
          {processing_time_ms && (
            <span className="processing-time">
              Processed in {(processing_time_ms / 1000).toFixed(2)}s
            </span>
          )}
        </div>
      </div>

      {/* Summary */}
      {summary && (
        <div className="results-summary">
          <pre>{summary}</pre>
        </div>
      )}

      {/* Field-by-field results */}
      <div className="field-results">
        <h3>Field Details</h3>
        <table className="results-table">
          <thead>
            <tr>
              <th>Field</th>
              <th>Status</th>
              <th>Extracted</th>
              <th>Expected</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody>
            {fields.map((field, index) => (
              <tr key={index} className={getStatusClass(field.status)}>
                <td className="field-name">{field.field_name}</td>
                <td className="field-status">
                  <span className="status-badge">
                    {getStatusIcon(field.status)} {field.status}
                  </span>
                </td>
                <td className="field-extracted">
                  {field.extracted_value || <span className="not-found">Not detected</span>}
                </td>
                <td className="field-expected">
                  {field.expected_value || '-'}
                </td>
                <td className="field-message">{field.message}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Extracted text (collapsible) */}
      {extracted && (
        <details className="extracted-details">
          <summary>View Extracted Data</summary>
          <div className="extracted-content">
            <h4>Extracted Fields</h4>
            <dl className="extracted-fields">
              <dt>Brand Name</dt>
              <dd>{extracted.brand_name || 'Not detected'}</dd>
              
              <dt>Class/Type</dt>
              <dd>{extracted.class_type || 'Not detected'}</dd>
              
              <dt>ABV</dt>
              <dd>{extracted.abv_percent ? `${extracted.abv_percent}%` : 'Not detected'}</dd>
              
              <dt>Net Contents</dt>
              <dd>{extracted.net_contents_ml ? `${extracted.net_contents_ml} mL` : 'Not detected'}</dd>
              
              <dt>Government Warning</dt>
              <dd>{extracted.government_warning || 'Not detected'}</dd>
              
              <dt>OCR Confidence</dt>
              <dd>{extracted.ocr_confidence ? `${(extracted.ocr_confidence * 100).toFixed(1)}%` : 'N/A'}</dd>
            </dl>

            {extracted.raw_text && (
              <>
                <h4>Raw OCR Text</h4>
                <pre className="raw-text">{extracted.raw_text}</pre>
              </>
            )}
          </div>
        </details>
      )}
    </div>
  )
}

export default VerificationResults
