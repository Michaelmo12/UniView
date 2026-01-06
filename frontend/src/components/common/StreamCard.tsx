import "./StreamCard.css";
import { Button } from "../common";

interface StreamCardProps {
  id: number;
  title: string;
  detections: number;
  isActive: boolean;
  color?: 'blue' | 'green' | 'orange' | 'purple' | 'yellow' | 'pink';
  onViewStream?: () => void;
}

export default function StreamCard({
  title,
  detections,
  isActive,
  color = 'blue',
  onViewStream
}: StreamCardProps) {
  return (
    <div className={`stream-card neon-box ${color}`}>
      {/* Status Header */}
      <div className="stream-card-header">
        <span className={`status-label ${detections > 0 ? 'tracking' : 'idle'}`}>
          {detections > 0 ? `${detections} TARGETS` : 'NO TARGETS'}
        </span>
        <span className={`stream-status ${isActive ? 'active' : 'inactive'}`}>
          {isActive ? 'ACTIVE' : 'INACTIVE'}
        </span>
      </div>

      {/* Camera Title */}
      <h3 className="stream-card-title">{title}</h3>

      {/* Stream Info */}
      <div className="stream-card-info">
        <div className="info-row">
          <span>Resolution:</span>
          <span className="info-value">1920x1080</span>
        </div>
        <div className="info-row">
          <span>FPS:</span>
          <span className="info-value">30</span>
        </div>
        <div className="info-row">
          <span>Status:</span>
          <span className={`info-value ${isActive ? 'active' : 'inactive'}`}>
            {isActive ? 'Active' : 'Inactive'}
          </span>
        </div>
      </div>

      {/* View Button */}
      <Button
        variant="outline"
        onClick={onViewStream}
        className="stream-card-button"
      >
        View Stream
      </Button>
    </div>
  );
}
