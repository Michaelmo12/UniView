import "./Checkbox.css";

interface CheckboxProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label: string | React.ReactNode;
}

export default function Checkbox({ label, ...props }: CheckboxProps) {
  return (
    <label className="checkbox-wrapper">
      <input type="checkbox" className="checkbox-input" {...props} />
      <span className="checkbox-label">{label}</span>
    </label>
  );
}
