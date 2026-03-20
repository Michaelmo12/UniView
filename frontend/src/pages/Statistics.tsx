import "./Statistics.css";

function Statistics() {
  return (
    <div className="stats-page">
      <div className="stats-header">
        <h1 className="stats-title">Statistics</h1>
        <p className="stats-subtitle">System Analytics &amp; Performance Metrics</p>
      </div>

      <div className="stats-section">
        <div className="stats-section-label">Tracking Overview</div>
        <div className="stats-grid">
          <div className="stat-card stat-card--placeholder stat-card--green">
            <div className="stat-card__label">Total Tracks</div>
            <div className="stat-card__value">—</div>
            <div className="stat-card__desc">Unique identities tracked across all cameras</div>
          </div>
          <div className="stat-card stat-card--placeholder stat-card--green">
            <div className="stat-card__label">Active Drones</div>
            <div className="stat-card__value">—</div>
            <div className="stat-card__desc">Live camera nodes currently streaming</div>
          </div>
          <div className="stat-card stat-card--placeholder">
            <div className="stat-card__label">Cross-Camera Matches</div>
            <div className="stat-card__value">—</div>
            <div className="stat-card__desc">Re-identifications across different views</div>
          </div>
        </div>
      </div>

      <div className="stats-section">
        <div className="stats-section-label">Detection Performance</div>
        <div className="stats-grid">
          <div className="stat-card stat-card--placeholder stat-card--purple">
            <div className="stat-card__label">Avg. Confidence</div>
            <div className="stat-card__value">—</div>
            <div className="stat-card__desc">Mean detection confidence score</div>
          </div>
          <div className="stat-card stat-card--placeholder stat-card--orange">
            <div className="stat-card__label">Detections / sec</div>
            <div className="stat-card__value">—</div>
            <div className="stat-card__desc">Current detection throughput</div>
          </div>
          <div className="stat-card stat-card--placeholder">
            <div className="stat-card__label">Data Processed</div>
            <div className="stat-card__value">—</div>
            <div className="stat-card__desc">Total frames processed this session</div>
          </div>
        </div>
      </div>

      <div className="stats-section">
        <div className="stats-section-label">System Health</div>
        <div className="stats-grid stats-grid--wide">
          <div className="stat-card stat-card--placeholder">
            <div className="stat-card__label">Uptime</div>
            <div className="stat-card__value">—</div>
            <div className="stat-card__desc">Session duration since last restart</div>
          </div>
          <div className="stat-card stat-card--placeholder">
            <div className="stat-card__label">Pipeline Latency</div>
            <div className="stat-card__value">—</div>
            <div className="stat-card__desc">End-to-end processing delay</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Statistics;
