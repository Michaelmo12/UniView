import "./Card.css";

interface CardProps {
  title?: string;
  subtitle?: string;
  children: React.ReactNode;
  maxWidth?: string;
}

export default function Card({ title, subtitle, children, maxWidth = '500px' }: CardProps) {
  return (
    <div className="card-wrapper">
      <div className="card" style={{ maxWidth }}>
        {(title || subtitle) && (
          <div className="card-header">
            {title && <h2>{title}</h2>}
            {subtitle && <p>{subtitle}</p>}
          </div>
        )}
        
        <div className="card-content">
          {children}
        </div>
      </div>
    </div>
  );
}
