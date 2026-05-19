import { BrowserRouter, useLocation } from "react-router-dom";
import AppRoutes from "./routes/index.tsx";
import Navigation from "./components/layout/Navigation";
import Footer from "./components/layout/Footer";
import "./styles/globals.css";

function Layout() {
  const location = useLocation();
  const isLogin = location.pathname === "/login";

  if (isLogin) {
    return <AppRoutes />;
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
      <Navigation />
      <main style={{ flex: 1 }}>
        <AppRoutes />
      </main>
      <Footer />
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Layout />
    </BrowserRouter>
  );
}

export default App;
