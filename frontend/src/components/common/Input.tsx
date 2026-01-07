import "./Input.css";

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label: string;
  id: string;
  error?: string;
}

export default function Input({ label, id, error, ...props }: InputProps) {
  return (
    <div className="input-group">
      <label htmlFor={id} className="input-label">{label}</label>
      <input id={id} className="input-field" {...props} />
      {error && <span className="input-error">{error}</span>}
    </div>
  );
}
