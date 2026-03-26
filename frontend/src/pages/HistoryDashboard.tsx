/**
 * HistoryDashboard.tsx
 *
 * C2 mission-history analytics page for UniView.
 * Displays a rolling timeline of tracking sessions with summary stat cards,
 * a recharts line chart, and a raw-data log table.
 *
 * Fetches real data from GET /api/history.
 */

import { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import {
  Users,
  Radio,
  GitMerge,
  TrendingUp,
  Activity,
  Clock,
} from "lucide-react";
import { apiRequest } from "../services/api/client";
import "./HistoryDashboard.css";

/* ─── Types ───────────────────────────────────────────────────── */

interface HistoryLogAPI {
  id: number;
  timestamp: string; // ISO 8601 from backend
  avg_people_count: number;
  peak_people_count: number;
  active_drones_count: number;
  total_reid_matches: number;
}

interface HistoryLog {
  id: string;
  timestamp: string; // e.g. "14:32"
  totalPersons: number;
  peakPersons: number;
  activeDrones: number;
  crossCameraMatches: number;
}

/* ─── Stat-card configuration ─────────────────────────────────── */

interface StatCardConfig {
  label: string;
  value: string | number;
  desc: string;
  modifier: "green" | "cyan" | "orange" | "purple" | "pink";
  Icon: React.ElementType;
}

/* ─── Custom Tooltip ──────────────────────────────────────────── */

interface TooltipPayload {
  value: number;
  name: string;
  dataKey: string;
  color: string;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: TooltipPayload[];
  label?: string;
}

function ChartTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload?.length) return null;
  return (
    <div className="history-chart__tooltip">
      <div className="history-chart__tooltip-time">{label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="history-chart__tooltip-row">
          <span
            className="history-chart__tooltip-dot"
            style={{ background: p.color }}
          />
          <span className="history-chart__tooltip-label">
            {p.name === "totalPersons"
              ? "Avg Persons"
              : p.name === "peakPersons"
              ? "Peak Persons"
              : p.name === "activeDrones"
              ? "Drones"
              : "Matches"}
          </span>
          <span className="history-chart__tooltip-val">{p.value}</span>
        </div>
      ))}
    </div>
  );
}

/* ─── Main Component ──────────────────────────────────────────── */

function HistoryDashboard() {
  const [logs, setLogs] = useState<HistoryLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function fetchHistory() {
      try {
        setLoading(true);
        const data = await apiRequest<HistoryLogAPI[]>("/history");
        if (cancelled) return;
        // Map API fields to display fields
        const mapped: HistoryLog[] = data.map((row) => {
          const dt = new Date(row.timestamp);
          const hh = String(dt.getHours()).padStart(2, "0");
          const mm = String(dt.getMinutes()).padStart(2, "0");
          return {
            id: String(row.id),
            timestamp: `${hh}:${mm}`,
            totalPersons: row.avg_people_count,
            peakPersons: row.peak_people_count,
            activeDrones: row.active_drones_count,
            crossCameraMatches: row.total_reid_matches,
          };
        });
        setLogs(mapped);
        setError(null);
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed to load history");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    fetchHistory();
    return () => { cancelled = true; };
  }, []);

  if (loading) {
    return (
      <div className="history-page">
        <header className="history-header">
          <div className="history-header__left">
            <Activity size={14} className="history-header__icon" />
            <div>
              <h1 className="history-title">Mission History</h1>
              <p className="history-subtitle">Loading session data...</p>
            </div>
          </div>
        </header>
      </div>
    );
  }

  if (!loading && logs.length === 0) {
    return (
      <div className="history-page">
        <header className="history-header">
          <div className="history-header__left">
            <Activity size={14} className="history-header__icon" />
            <div>
              <h1 className="history-title">Mission History</h1>
              <p className="history-subtitle">No history records yet</p>
            </div>
          </div>
        </header>
        <section className="history-section">
          <div className="history-section-label">
            {error ? `Error: ${error}` : "Start a surveillance session to see analytics here."}
          </div>
        </section>
      </div>
    );
  }

  /* Derived statistics */
  const avgPersons = Math.round(
    logs.reduce((s, l) => s + l.totalPersons, 0) / logs.length
  );
  const peakPersons = Math.max(...logs.map((l) => l.peakPersons));
  const peakDrones  = Math.max(...logs.map((l) => l.activeDrones));
  const totalMatches = logs.reduce((s, l) => s + l.crossCameraMatches, 0);

  const peakPersonsTime = logs.find((l) => l.peakPersons === peakPersons)?.timestamp ?? "—";

  const statCards: StatCardConfig[] = [
    {
      label: "Avg Persons Detected",
      value: avgPersons,
      desc: "Mean tracked persons per minute across the session",
      modifier: "green",
      Icon: Users,
    },
    {
      label: "Peak Persons",
      value: `${peakPersons} @ ${peakPersonsTime}`,
      desc: "Highest simultaneous tracked persons",
      modifier: "cyan",
      Icon: TrendingUp,
    },
    {
      label: "Peak Drones Active",
      value: peakDrones,
      desc: "Maximum concurrent camera nodes online",
      modifier: "orange",
      Icon: Radio,
    },
    {
      label: "Total Re-ID Matches",
      value: totalMatches,
      desc: "Cross-camera identity matches over the session",
      modifier: "purple",
      Icon: GitMerge,
    },
  ];

  return (
    <div className="history-page">

      {/* ── Page header ─────────────────────────────────────────── */}
      <header className="history-header">
        <div className="history-header__left">
          <Activity size={14} className="history-header__icon" />
          <div>
            <h1 className="history-title">Mission History</h1>
            <p className="history-subtitle">Session Analytics &amp; Tracking Log</p>
          </div>
        </div>
        <div className="history-header__meta">
          <Clock size={11} />
          <span>
            {logs[0]?.timestamp} – {logs[logs.length - 1]?.timestamp}
          </span>
          <span className="history-header__sep">·</span>
          <span>{logs.length} entries</span>
        </div>
      </header>

      {/* ── Stat cards ──────────────────────────────────────────── */}
      <section className="history-section">
        <div className="history-section-label">Session Summary</div>
        <div className="history-stat-grid">
          {statCards.map((card) => (
            <div
              key={card.label}
              className={`history-stat-card history-stat-card--${card.modifier}`}
            >
              <div className="history-stat-card__header">
                <span className="history-stat-card__label">{card.label}</span>
                <card.Icon size={13} className="history-stat-card__icon" />
              </div>
              <div className="history-stat-card__value">{card.value}</div>
              <div className="history-stat-card__desc">{card.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Line Chart ──────────────────────────────────────────── */}
      <section className="history-section">
        <div className="history-section-label">Tracked Persons Over Time</div>
        <div className="history-chart">
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={logs} margin={{ top: 8, right: 24, left: -12, bottom: 0 }}>
              {/* Subtle grid */}
              <CartesianGrid
                strokeDasharray="3 6"
                stroke="rgba(0,255,136,0.06)"
                vertical={false}
              />

              {/* Axes */}
              <XAxis
                dataKey="timestamp"
                tick={{ fill: "rgba(255,255,255,0.2)", fontSize: 10, fontFamily: "var(--font-mono)" }}
                tickLine={false}
                axisLine={{ stroke: "rgba(0,255,136,0.08)" }}
                interval={4}
              />
              <YAxis
                tick={{ fill: "rgba(255,255,255,0.2)", fontSize: 10, fontFamily: "var(--font-mono)" }}
                tickLine={false}
                axisLine={false}
                width={32}
              />

              {/* Average reference line */}
              <ReferenceLine
                y={avgPersons}
                stroke="rgba(0,212,255,0.25)"
                strokeDasharray="4 4"
                label={{
                  value: `avg ${avgPersons}`,
                  fill: "rgba(0,212,255,0.45)",
                  fontSize: 10,
                  fontFamily: "var(--font-mono)",
                  position: "insideTopRight",
                }}
              />

              <Tooltip content={<ChartTooltip />} />

              {/* Avg Persons — primary neon green line */}
              <Line
                type="monotone"
                dataKey="totalPersons"
                stroke="#00ff88"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, fill: "#00ff88", stroke: "#0d0d0d", strokeWidth: 2 }}
              />

              {/* Peak Persons — orange envelope */}
              <Line
                type="monotone"
                dataKey="peakPersons"
                stroke="#ff9500"
                strokeWidth={1.5}
                strokeDasharray="4 2"
                dot={false}
                activeDot={{ r: 3, fill: "#ff9500", stroke: "#0d0d0d", strokeWidth: 2 }}
              />

              {/* Active Drones — cyan secondary */}
              <Line
                type="monotone"
                dataKey="activeDrones"
                stroke="#00d4ff"
                strokeWidth={1.5}
                strokeDasharray="5 3"
                dot={false}
                activeDot={{ r: 3, fill: "#00d4ff", stroke: "#0d0d0d", strokeWidth: 2 }}
              />

              {/* Cross-camera matches — purple tertiary */}
              <Line
                type="monotone"
                dataKey="crossCameraMatches"
                stroke="#b968ff"
                strokeWidth={1.5}
                strokeDasharray="2 4"
                dot={false}
                activeDot={{ r: 3, fill: "#b968ff", stroke: "#0d0d0d", strokeWidth: 2 }}
              />
            </LineChart>
          </ResponsiveContainer>

          {/* Chart legend */}
          <div className="history-chart__legend">
            <span className="history-chart__legend-item history-chart__legend-item--green">
              Avg Persons
            </span>
            <span className="history-chart__legend-item history-chart__legend-item--orange">
              Peak Persons
            </span>
            <span className="history-chart__legend-item history-chart__legend-item--cyan">
              Active Drones
            </span>
            <span className="history-chart__legend-item history-chart__legend-item--purple">
              Re-ID Matches
            </span>
          </div>
        </div>
      </section>

      {/* ── Data Table ──────────────────────────────────────────── */}
      <section className="history-section">
        <div className="history-section-label">Raw Log</div>
        <div className="history-table-wrap">
          <table className="history-table">
            <thead>
              <tr>
                <th className="history-table__th history-table__th--id">#</th>
                <th className="history-table__th">
                  <Clock size={10} style={{ marginRight: 5, verticalAlign: "middle" }} />
                  Timestamp
                </th>
                <th className="history-table__th">
                  <Users size={10} style={{ marginRight: 5, verticalAlign: "middle" }} />
                  Tracked Persons
                </th>
                <th className="history-table__th">
                  <Radio size={10} style={{ marginRight: 5, verticalAlign: "middle" }} />
                  Active Drones
                </th>
                <th className="history-table__th">
                  <GitMerge size={10} style={{ marginRight: 5, verticalAlign: "middle" }} />
                  Matches Found
                </th>
              </tr>
            </thead>
            <tbody>
              {logs.map((log, idx) => {
                // Highlight peak-persons row
                const isPeak = log.totalPersons === peakPersons;
                return (
                  <tr
                    key={log.id}
                    className={`history-table__row${isPeak ? " history-table__row--peak" : ""}`}
                  >
                    <td className="history-table__td history-table__td--id">{idx + 1}</td>
                    <td className="history-table__td history-table__td--time">{log.timestamp}</td>
                    <td className="history-table__td history-table__td--persons">{log.totalPersons}</td>
                    <td className="history-table__td history-table__td--drones">{log.activeDrones}</td>
                    <td className="history-table__td history-table__td--matches">{log.crossCameraMatches}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

export default HistoryDashboard;
