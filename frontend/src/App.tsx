import { BrowserRouter } from "react-router-dom";
import AppRoutes from "./routes/index.tsx";
import Navigation from "./components/layout/Navigation";
import Footer from "./components/layout/Footer";
import "./styles/globals.css";

function App() {
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

export default App;
