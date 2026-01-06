import "./InfoBox.css";

interface InfoBoxProps {
  children: React.ReactNode;
  variant?: 'info' | 'warning' | 'success';
}

export default function InfoBox({ children, variant = 'info' }: InfoBoxProps) {
  return (
    <div className={`info-box info-box-${variant}`}>
      {children}
    </div>
  );
}
