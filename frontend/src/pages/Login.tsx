import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { AuthCard, Input, Button, Checkbox } from "../components/common";
import { useAuth } from "../context/AuthContext";
import "./Login.css";

function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [rememberMe, setRememberMe] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const { login, isAuthenticated } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (isAuthenticated) {
      navigate("/");
    }
  }, [isAuthenticated, navigate]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      await login(email, password, rememberMe);
      navigate("/");
    } catch (err: any) {
      setError(err.message || "Login failed. Please check your credentials.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-content">
      <AuthCard
        title="Welcome Back"
        subtitle="Sign in to your account"
      >
      <form onSubmit={handleSubmit} className="form-container">
        <Input
          id="email"
          label="Email Address"
          type="email"
          placeholder="you@example.com"
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

        <div className="login-form-row">
          <Checkbox
            label="Remember me"
            checked={rememberMe}
            onChange={(e) => setRememberMe(e.target.checked)}
          />
          <a href="#" className="login-forgot-link">
            Forgot password?
          </a>
        </div>

        {error && <div className="error-message">{error}</div>}

        <Button type="submit" loading={loading}>
          {loading ? "Signing In..." : "Sign In"}
        </Button>
      </form>
    </AuthCard>
    </div>
  );
}

export default Login;
