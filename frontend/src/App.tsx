import { BrowserRouter } from "react-router-dom";
import { useEffect } from "react";
import AppRoutes from "./routes/index.tsx";
import Navigation from "./components/layout/Navigation";
import Footer from "./components/layout/Footer";
import { ThemeProvider, useTheme } from "./context/ThemeContext";
import "./styles/globals.css";

function AppContent() {
  const { isDark } = useTheme();

  useEffect(() => {
    if (isDark) {
      document.body.classList.add("dark");
    } else {
      document.body.classList.remove("dark");
    }
  }, [isDark]);

  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
      <BrowserRouter>
        <Navigation />
          <AppRoutes />
        <Footer />
      </BrowserRouter>
    </div>
  );
}

function App() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  );
}

export default App;
