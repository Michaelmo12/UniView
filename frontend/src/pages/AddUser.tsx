import { useState } from "react";
import { Card, Input, Button } from "../components/common";

function AddUser() {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [role, setRole] = useState("user");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Admin creates user - Logic will be added later
    console.log("Admin creating user:", {
      fullName,
      email,
      password,
      confirmPassword,
      role,
    });
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
              onChange={(e) => setRole(e.target.value)}
              required
            >
              <option value="user">User</option>
              <option value="admin">Admin</option>
            </select>
          </div>

          <Button type="submit">Create User</Button>
        </form>
      </Card>
    </div>
  );
}

export default AddUser;
