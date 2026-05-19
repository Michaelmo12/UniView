import { Radio, Crosshair, Activity, ShieldCheck, AlertTriangle, RefreshCw, Timer, Target } from "lucide-react";
import { useAlgorithmStats } from "../hooks/useAlgorithmStats";
import { MetricWidget } from "../components/MetricWidget";
import "./Statistics.css";

function Statistics() {
  const { data, loading, error, refetch } = useAlgorithmStats();

  const isPipelineOffline = data !== null && data.pipeline_online === false;

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

      {data && isPipelineOffline && (
        <div className="stats-offline">
          <span className="stats-offline__dot" />
          PIPELINE OFFLINE — NO ACTIVE DRONES DETECTED
        </div>
      )}

      {data && (
        <div className={isPipelineOffline ? "stats-dimmed" : ""}>
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
              <MetricWidget
                icon={Target}
                label="Confidence Avg"
                value={`${((data.avg_confidence ?? 0) * 100).toFixed(1)}%`}
                description="Avg YOLO detection confidence"
                color="orange"
              />
            </div>
          </div>

          <div className="stats-section">
            <div className="stats-section-label">
              <Timer size={13} style={{ display: "inline", marginRight: 6, verticalAlign: "middle" }} />
              Per-Stage Timings (avg ms)
            </div>
            <div className="stats-timings">
              {(["detection", "features", "fusion", "reconstruction", "tracking", "total"] as const).map((stage, i) => {
                const ms = data.stage_timings_ms?.[stage] ?? 0;
                const total = data.stage_timings_ms?.total || 1;
                const pct = stage === "total" ? 100 : Math.round((ms / total) * 100);
                const isTotal = stage === "total";
                // Color shifts green → amber → red as percentage rises
                const barColor = isTotal
                  ? "rgba(0, 255, 136, 0.9)"
                  : pct < 30
                    ? "rgba(0, 255, 136, 0.65)"
                    : pct < 60
                      ? "rgba(251, 191, 36, 0.75)"
                      : "rgba(255, 68, 68, 0.75)";
                return (
                  <div key={stage} className={`stats-timing-row${isTotal ? " stats-timing-row--total" : ""}`}>
                    <span className="stats-timing-label">{stage}</span>
                    <div className="stats-timing-bar-wrap">
                      <div
                        className="stats-timing-bar"
                        style={{
                          "--bar-width": `${pct}%`,
                          "--bar-color": barColor,
                          animationDelay: `${i * 0.08}s`,
                        } as React.CSSProperties}
                      />
                    </div>
                    <span className="stats-timing-value">{ms.toFixed(1)} ms</span>
                    {!isTotal && <span className="stats-timing-pct">{pct}%</span>}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Statistics;
