function ApplicationForm({ formData, onChange, disabled }) {
  return (
    <form className="application-form" onSubmit={(e) => e.preventDefault()}>
      <div className="form-group">
        <label htmlFor="brand_name">
          Brand Name <span className="required">*</span>
        </label>
        <input
          type="text"
          id="brand_name"
          value={formData.brand_name}
          onChange={(e) => onChange('brand_name', e.target.value)}
          placeholder="e.g., OLD TOM DISTILLERY"
          disabled={disabled}
          required
        />
        <p className="field-hint">The brand name as it appears on the application</p>
      </div>

      <div className="form-group">
        <label htmlFor="class_type">Class/Type</label>
        <input
          type="text"
          id="class_type"
          value={formData.class_type}
          onChange={(e) => onChange('class_type', e.target.value)}
          placeholder="e.g., Kentucky Straight Bourbon Whiskey"
          disabled={disabled}
        />
        <p className="field-hint">The product class or type designation</p>
      </div>

      <div className="form-row">
        <div className="form-group">
          <label htmlFor="abv_percent">ABV %</label>
          <input
            type="number"
            id="abv_percent"
            value={formData.abv_percent}
            onChange={(e) => onChange('abv_percent', e.target.value)}
            placeholder="e.g., 45"
            min="0"
            max="100"
            step="0.1"
            disabled={disabled}
          />
          <p className="field-hint">Alcohol by volume percentage</p>
        </div>

        <div className="form-group">
          <label htmlFor="net_contents_ml">Net Contents (mL)</label>
          <input
            type="number"
            id="net_contents_ml"
            value={formData.net_contents_ml}
            onChange={(e) => onChange('net_contents_ml', e.target.value)}
            placeholder="e.g., 750"
            min="0"
            step="1"
            disabled={disabled}
          />
          <p className="field-hint">Volume in milliliters</p>
        </div>
      </div>

      <div className="form-group checkbox-group">
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={formData.has_warning}
            onChange={(e) => onChange('has_warning', e.target.checked)}
            disabled={disabled}
          />
          <span className="checkbox-text">
            Label should include Government Health Warning
          </span>
        </label>
        <p className="field-hint">Required for all alcohol beverages sold in the US</p>
      </div>
    </form>
  )
}

export default ApplicationForm
