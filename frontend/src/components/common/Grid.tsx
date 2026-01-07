import "./Grid.css";

interface GridProps {
  children: React.ReactNode;
  columns?: number;
  minWidth?: string;
  gap?: string;
}

export default function Grid({ 
  children, 
  columns, 
  minWidth = '300px',
  gap = '2rem' 
}: GridProps) {
  const gridStyle = {
    gap,
    gridTemplateColumns: columns 
      ? `repeat(${columns}, 1fr)` 
      : `repeat(auto-fit, minmax(${minWidth}, 1fr))`
  };

  return (
    <div className="grid-container" style={gridStyle}>
      {children}
    </div>
  );
}
