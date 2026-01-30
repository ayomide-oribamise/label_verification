function LoadingSpinner({ size = 'small' }) {
  return (
    <span className={`spinner spinner-${size}`} aria-label="Loading">
      <span className="spinner-dot"></span>
      <span className="spinner-dot"></span>
      <span className="spinner-dot"></span>
    </span>
  )
}

export default LoadingSpinner
