import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { Input, Button, Checkbox } from "../components/common";
import { useAuth } from "../context/AuthContext";
import lionSvg from "../assets/lion.svg";
import watermarkTile from "../assets/watermark-tile.svg?url";
import "./Login.css";

function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [rememberMe, setRememberMe] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [errorVisible, setErrorVisible] = useState(false);
  const [booted, setBooted] = useState(false);
  const fadeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const { login, isAuthenticated } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (isAuthenticated) navigate("/");
  }, [isAuthenticated, navigate]);

  useEffect(() => {
    const t = setTimeout(() => setBooted(true), 400);
    return () => clearTimeout(t);
  }, []);

  // When error is set, show the alert. When cleared, fade it out then hide.
  useEffect(() => {
    if (fadeTimerRef.current) clearTimeout(fadeTimerRef.current);
    if (error) {
      setErrorVisible(true);
    } else {
      // Trigger CSS fade-out, then remove from DOM after animation completes
      fadeTimerRef.current = setTimeout(() => setErrorVisible(false), 300);
    }
  }, [error]);

  const [leaving, setLeaving] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await login(email, password, rememberMe);
      // Trigger fade-out, then navigate after animation completes
      setLeaving(true);
      setTimeout(() => navigate("/"), 600);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Authentication failed.");
      setLoading(false);
    }
  };

  const clearError = () => setError("");

  return (
    <div className={`login-page ${leaving ? "login-page--leaving" : ""}`}>
      <div className="login-bg">
        <div className="login-bg__grid" />
        <div className="login-bg__scan" />
        <div className="login-bg__vignette" />
      </div>

      <div className="login-watermark-tile" style={{ backgroundImage: `url(${watermarkTile})` }} aria-hidden="true" />

      <img src={lionSvg} className="login-emblem" aria-hidden="true" alt="" />

      {errorVisible && (
        <div className={`login-alert ${!error ? "login-alert--fading" : ""}`}>
          <span className="login-alert__icon">!</span>
          {error}
        </div>
      )}

      <div className={`login-card ${booted ? "login-card--visible" : ""}`}>
        <div className="login-card__corner login-card__corner--tl" />
        <div className="login-card__corner login-card__corner--tr" />
        <div className="login-card__corner login-card__corner--bl" />
        <div className="login-card__corner login-card__corner--br" />

        <div className="login-card__header">
          <div className="login-card__system">UNIVIEW // SURVEILLANCE NETWORK</div>
          <h1 className="login-card__title">SYSTEM ACCESS</h1>
          <div className="login-card__status">
            <span className="login-card__status-dot" />
            AUTHENTICATION REQUIRED
          </div>
        </div>

        <div className="login-card__divider" />

        <form onSubmit={handleSubmit} className="login-card__form">
          <Input
            id="email"
            label="Operator ID"
            type="email"
            placeholder="operator@domain.com"
            value={email}
            onChange={(e) => { setEmail(e.target.value); clearError(); }}
            required
          />
          <Input
            id="password"
            label="Access Code"
            type="password"
            placeholder="••••••••"
            value={password}
            onChange={(e) => { setPassword(e.target.value); clearError(); }}
            required
          />

          <div className="login-form-row">
            <Checkbox
              label="Keep session active"
              checked={rememberMe}
              onChange={(e) => setRememberMe(e.target.checked)}
            />
          </div>

          <Button type="submit" isLoading={loading}>
            {loading ? "Authenticating..." : "Authenticate"}
          </Button>
        </form>

        <div className="login-card__footer">
          UNIVIEW · RESTRICTED ACCESS · AUTHORIZED PERSONNEL ONLY
        </div>
      </div>
    </div>
  );
}

export default Login;
