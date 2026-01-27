function BatchResults({ results }) {
  if (!results) return null

  const { total, processed, passed, needs_review, failed, results: rowResults, processing_time_ms } = results

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

  const exportResults = () => {
    const csvRows = [
      ['Filename', 'Status', 'Summary', 'Error'].join(','),
      ...rowResults.map(r => [
        r.filename,
        r.success ? r.result?.overall_status : 'error',
        r.success ? r.result?.summary?.replace(/[\n,]/g, ' ') : '',
        r.error || ''
      ].map(v => `"${v}"`).join(','))
    ]
    
    const blob = new Blob([csvRows.join('\n')], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'verification_results.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="batch-results">
      <div className="results-header">
        <h2>Batch Results</h2>
        <button className="btn btn-secondary" onClick={exportResults}>
          📥 Export CSV
        </button>
      </div>

      {/* Summary stats */}
      <div className="batch-summary">
        <div className="stat">
          <span className="stat-value">{total}</span>
          <span className="stat-label">Total</span>
        </div>
        <div className="stat stat-passed">
          <span className="stat-value">{passed}</span>
          <span className="stat-label">✅ Passed</span>
        </div>
        <div className="stat stat-review">
          <span className="stat-value">{needs_review}</span>
          <span className="stat-label">⚠️ Review</span>
        </div>
        <div className="stat stat-failed">
          <span className="stat-value">{failed}</span>
          <span className="stat-label">❌ Failed</span>
        </div>
        {processing_time_ms && (
          <div className="stat">
            <span className="stat-value">{(processing_time_ms / 1000).toFixed(1)}s</span>
            <span className="stat-label">Total Time</span>
          </div>
        )}
      </div>

      {/* Results table */}
      <div className="batch-table-container">
        <table className="batch-table">
          <thead>
            <tr>
              <th>Filename</th>
              <th>Status</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody>
            {rowResults.map((row, index) => (
              <tr key={index} className={row.success ? getStatusClass(row.result?.overall_status) : 'status-error'}>
                <td className="filename">{row.filename}</td>
                <td className="status">
                  {row.success ? (
                    <span className="status-badge">
                      {getStatusIcon(row.result?.overall_status)} {row.result?.overall_status}
                    </span>
                  ) : (
                    <span className="status-badge status-error">❌ Error</span>
                  )}
                </td>
                <td className="details">
                  {row.success ? (
                    <details>
                      <summary>
                        {row.result?.fields?.length || 0} fields verified
                        {row.result?.processing_time_ms && ` (${row.result.processing_time_ms}ms)`}
                      </summary>
                      <div className="row-details">
                        {row.result?.fields?.map((field, i) => (
                          <div key={i} className={`field-row ${getStatusClass(field.status)}`}>
                            <span className="field-icon">{getStatusIcon(field.status)}</span>
                            <span className="field-name">{field.field_name}:</span>
                            <span className="field-message">{field.message}</span>
                          </div>
                        ))}
                      </div>
                    </details>
                  ) : (
                    <span className="error-message">{row.error}</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default BatchResults
