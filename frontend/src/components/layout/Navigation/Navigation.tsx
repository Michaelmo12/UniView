import { NavLink } from "react-router-dom";
import "./Navigation.css";

function Navigation() {
  return (
    <nav className="navbar">
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
      </div>
    </nav>
  );
}

export default Navigation;
