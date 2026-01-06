import { BrowserRouter } from "react-router-dom";
import AppRoutes from "./routes/index.tsx";
import Navigation from "./components/layout/Navigation";
import Footer from "./components/layout/Footer";
import "./styles/globals.css";

function App() {
  return (
    <BrowserRouter>
      <div style={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
        <Navigation />
        <main style={{ flex: 1 }}>
          <AppRoutes />
        </main>
        <Footer />
      </div>
    </BrowserRouter>
  );
}

export default App;
