import { NavLink } from "react-router-dom";
import "./Navigation.css";
import { useTheme } from "../../../context/ThemeContext";

function Navigation() {
  const { isDark, toggleTheme } = useTheme();

  return (
    <nav className={`navbar ${isDark ? "dark" : ""}`}>
      <div className="nav-container">
        <div className="nav-logo">UniView</div>

        <div className="nav-links">
          <NavLink to="/" className="nav-link">
            Home
          </NavLink>
          <NavLink to="/login" className="nav-link">
            Login
          </NavLink>
          <NavLink to="/signin" className="nav-link">
            Sign In
          </NavLink>
        </div>

        <button className="theme-toggle" onClick={toggleTheme}>
          {isDark ? "☀️" : "🌙"}
        </button>
      </div>
    </nav>
  );
}

export default Navigation;
