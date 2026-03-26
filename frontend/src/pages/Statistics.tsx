import { Radio, Crosshair, Activity, ShieldCheck, AlertTriangle, RefreshCw } from "lucide-react";
import { useAlgorithmStats } from "../hooks/useAlgorithmStats";
import { MetricWidget } from "../components/MetricWidget";
import "./Statistics.css";

function Statistics() {
  const { data, loading, error, refetch } = useAlgorithmStats();

  const statusColor = data
    ? data.system_status === "Optimal" ? "green"
    : data.system_status === "Warning" ? "orange"
    : "red"
    : "green";

  const StatusIcon = data?.system_status === "Critical" ? AlertTriangle : ShieldCheck;

  return (
    <div className="stats-page">
      <div className="stats-header">
        <div className="stats-header-row">
          <div>
            <h1 className="stats-title">Algorithm System Status</h1>
            <p className="stats-subtitle">Live pipeline health metrics</p>
          </div>
          <button
            className="stats-refresh-btn"
            onClick={refetch}
            disabled={loading}
          >
            <RefreshCw />
            Refresh
          </button>
        </div>
      </div>

      {loading && !data && (
        <div className="stats-loading">Loading status...</div>
      )}

      {error && !data && (
        <div className="stats-error">
          <span className="stats-error__message">{error}</span>
          <button className="stats-error__retry" onClick={refetch}>
            Retry
          </button>
        </div>
      )}

      {data && (
        <div className="stats-section">
          <div className="stats-section-label">Pipeline Metrics</div>
          <div className="stats-grid stats-grid--quad">
            <MetricWidget
              icon={Radio}
              label="Active Drones"
              value={data.active_drones}
              description="Camera nodes currently streaming"
              color="green"
            />
            <MetricWidget
              icon={Crosshair}
              label="Active Tracks"
              value={data.active_tracks}
              description="Persons currently being tracked"
              color="purple"
            />
            <MetricWidget
              icon={Activity}
              label="Server FPS"
              value={data.server_fps.toFixed(1)}
              description="Pipeline processing framerate"
              color="orange"
            />
            <MetricWidget
              icon={StatusIcon}
              label="System Status"
              value={data.system_status}
              description="Overall pipeline health"
              color={statusColor}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default Statistics;
