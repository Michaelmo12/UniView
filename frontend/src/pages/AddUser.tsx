import { useState } from "react";
import { Input, Button } from "../components/common";
import { authAPI } from "../services/api/auth";
import "./AddUser.css";

function AddUser() {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [role, setRole] = useState<'user' | 'admin'>("user");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess("");

    if (password !== confirmPassword) {
      setError("Passwords do not match!");
      return;
    }

    if (password.length < 8) {
      setError("Password must be at least 8 characters long!");
      return;
    }

    setLoading(true);

    try {
      await authAPI.createUser({
        full_name: fullName,
        email,
        password,
        role,
      });

      setSuccess(`User ${email} created successfully!`);
      setFullName("");
      setEmail("");
      setPassword("");
      setConfirmPassword("");
      setRole("user");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create user. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="adduser-page">
      <div className="adduser-header">
        <h1 className="adduser-title">Add User</h1>
        <p className="adduser-subtitle">Create a new user account — Admin only</p>
      </div>

      <div className="adduser-card">
        <form onSubmit={handleSubmit} className="form-container">
          <Input
            id="fullName"
            label="Full Name"
            type="text"
            placeholder="John Doe"
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
            required
          />

          <Input
            id="email"
            label="Email Address"
            type="email"
            placeholder="user@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />

          <Input
            id="password"
            label="Password"
            type="password"
            placeholder="••••••••"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />

          <Input
            id="confirmPassword"
            label="Confirm Password"
            type="password"
            placeholder="••••••••"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            required
          />

          <div>
            <span className="role-label">User Role</span>
            <div className="role-toggle">
              <button
                type="button"
                className={`role-btn ${role === 'user' ? 'role-btn--active' : ''}`}
                onClick={() => setRole('user')}
              >
                User
              </button>
              <button
                type="button"
                className={`role-btn role-btn--admin ${role === 'admin' ? 'role-btn--active' : ''}`}
                onClick={() => setRole('admin')}
              >
                Admin
              </button>
            </div>
          </div>

          {error && <div className="adduser-error">{error}</div>}
          {success && <div className="adduser-success">{success}</div>}

          <Button type="submit" isLoading={loading}>
            {loading ? "Creating User..." : "Create User"}
          </Button>
        </form>
      </div>
    </div>
  );
}

export default AddUser;
