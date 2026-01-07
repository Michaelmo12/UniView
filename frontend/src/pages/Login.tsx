import { useState } from "react";
import { AuthCard, Input, Button, Checkbox } from "../components/common";

function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [rememberMe, setRememberMe] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Logic will be added later
    console.log("Login attempt:", { email, password, rememberMe });
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

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '13px' }}>
          <Checkbox
            label="Remember me"
            checked={rememberMe}
            onChange={(e) => setRememberMe(e.target.checked)}
          />
          <a href="#" style={{ color: 'var(--neon-pink)' }}>
            Forgot password?
          </a>
        </div>

        <Button type="submit">Sign In</Button>
      </form>
    </AuthCard>
    </div>
  );
}

export default Login;
