import { NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "../../../context/AuthContext";
import "./Navigation.css";

function Navigation() {
  const { isAuthenticated, isAdmin, logout, user } = useAuth();
  const navigate = useNavigate();

  const handleLogout = async () => {
    await logout();
    navigate("/login");
  };

  return (
    <nav className="navbar">
      <div className="nav-container">
        <div className="nav-logo">UniView</div>

        <div className="nav-links">
          {isAuthenticated ? (
            <>
              <NavLink to="/" className="nav-link">
                Home
              </NavLink>
              {isAdmin && (
                <NavLink to="/admin/add-user" className="nav-link">
                  Add User
                </NavLink>
              )}
              <span className="nav-user">{user?.email}</span>
              <button onClick={handleLogout} className="nav-link nav-logout">
                Logout
              </button>
            </>
          ) : (
            <NavLink to="/login" className="nav-link">
              Login
            </NavLink>
          )}
        </div>
      </div>
    </nav>
  );
}

export default Navigation;
