import type { LucideIcon } from "lucide-react";

interface MetricWidgetProps {
  icon: LucideIcon;
  label: string;
  value: string | number;
  description: string;
  color?: "green" | "orange" | "red" | "purple";
}

const COLOR_MAP: Record<string, string> = {
  green: "#00ff88",
  orange: "rgba(255, 136, 68, 1)",
  red: "#ff4444",
  purple: "rgba(185, 104, 255, 1)",
};

export function MetricWidget({ icon, label, value, description, color }: MetricWidgetProps) {
  const Icon = icon;
  const valueColor = color ? COLOR_MAP[color] : "rgba(255, 255, 255, 0.85)";

  return (
    <div className={`stat-card${color ? ` stat-card--${color}` : ""}`}>
      <div className="stat-card__header">
        <span className="stat-card__label">{label}</span>
        <Icon size={13} />
      </div>
      <div className="stat-card__value" style={{ color: valueColor }}>
        {value}
      </div>
      <div className="stat-card__desc">{description}</div>
    </div>
  );
}
