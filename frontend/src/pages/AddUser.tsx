import { useState } from "react";
import { Card, Input, Button, InfoBox } from "../components/common";
import { authAPI } from "../services/api/auth";

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
    <div className="page-content">
      <Card
        title="Add New User"
        subtitle="Create a new user account (Admin Only)"
        maxWidth="500px"
      >
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

          <div className="input-group">
            <label htmlFor="role" className="input-label">User Role</label>
            <select
              id="role"
              className="input-field"
              value={role}
              onChange={(e) => setRole(e.target.value as 'user' | 'admin')}
              required
            >
              <option value="user">User</option>
              <option value="admin">Admin</option>
            </select>
          </div>

          {error && <div className="error-message">{error}</div>}
          {success && (
            <InfoBox variant="success">
              {success}
            </InfoBox>
          )}

          <Button type="submit" isLoading={loading}>
            {loading ? "Creating User..." : "Create User"}
          </Button>
        </form>
      </Card>
    </div>
  );
}

export default AddUser;
