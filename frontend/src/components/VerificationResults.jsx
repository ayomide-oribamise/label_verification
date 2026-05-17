function VerificationResults({ results }) {
  if (!results || !results.result) {
    return null
  }

  const { result, extracted } = results
  const { overall_status, fields, summary, processing_time_ms, issues = [] } = result
  const metFiveSecondTarget = processing_time_ms && processing_time_ms <= 5000
  const contradictionFields = fields.filter((field) => field.status === 'mismatch')
  const primaryMissingFields = fields.filter((field) => (
    field.status === 'not_visible_on_label' && field.category === 'required_on_primary'
  ))
  const reviewFields = fields.filter((field) => (
    field.status === 'review'
    || field.status === 'low_confidence'
    || (field.status === 'not_visible_on_label' && field.category !== 'required_on_primary')
  ))

  const getStatusIcon = (status) => {
    switch (status) {
      case 'match':
        return '✅'
      case 'review':
      case 'not_visible_on_label':
      case 'low_confidence':
        return '⚠️'
      case 'incomplete':
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
      case 'not_visible_on_label':
      case 'low_confidence':
        return 'status-review'
      case 'incomplete':
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
        return 'Verified'
      case 'review':
        return 'Review Required'
      case 'incomplete':
        return 'Cannot Verify'
      case 'mismatch':
        return 'Cannot Verify'
      default:
        return 'Unknown Status'
    }
  }

  const formatStatusLabel = (status) => {
    switch (status) {
      case 'not_visible_on_label':
        return 'not visible'
      case 'low_confidence':
        return 'low confidence'
      default:
        return status?.replaceAll('_', ' ') || 'unknown'
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
              {' '}
              ({metFiveSecondTarget ? 'meets' : 'exceeds'} 5s target)
            </span>
          )}
        </div>
      </div>

      {/* Summary */}
      {(summary || issues.length > 0) && (
        <div className={`results-summary ${getStatusClass(overall_status)}`}>
          {issues.length > 0 && (
            <div className="issue-groups">
              {contradictionFields.length > 0 && (
                <div className="issue-group">
                  <h4>Contradictions</h4>
                  <ul>
                    {contradictionFields.map((field, index) => (
                      <li key={index}>{field.message}</li>
                    ))}
                  </ul>
                </div>
              )}
              {primaryMissingFields.length > 0 && (
                <div className="issue-group">
                  <h4>Missing Primary-Label Fields</h4>
                  <ul>
                    {primaryMissingFields.map((field, index) => (
                      <li key={index}>{field.field_name}: {field.guidance || field.message}</li>
                    ))}
                  </ul>
                </div>
              )}
              {reviewFields.length > 0 && (
                <div className="issue-group">
                  <h4>Review Checklist</h4>
                  <ul>
                    {reviewFields.map((field, index) => (
                      <li key={index}>{field.field_name}: {field.guidance || field.message}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
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
                    {getStatusIcon(field.status)} {formatStatusLabel(field.status)}
                  </span>
                </td>
                <td className="field-extracted">
                  {field.extracted_value || <span className="not-found">Not detected</span>}
                </td>
                <td className="field-expected">
                  {field.expected_value || '-'}
                </td>
                <td className="field-message">
                  <span>{field.message}</span>
                  {field.guidance && (
                    <span className="field-guidance">{field.guidance}</span>
                  )}
                </td>
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

              <dt>Bottler / Producer</dt>
              <dd>{extracted.bottler_producer || 'Not detected'}</dd>

              <dt>Country of Origin</dt>
              <dd>{extracted.country_of_origin || 'Not detected'}</dd>
              
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
