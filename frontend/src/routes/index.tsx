import { Routes, Route, Navigate } from "react-router-dom";
import { ProtectedRoute } from "../components/common";
import Home from "../pages/Home.tsx";
import Login from "../pages/Login.tsx";
import AddUser from "../pages/AddUser.tsx";
import Statistics from "../pages/Statistics.tsx";
import HistoryDashboard from "../pages/HistoryDashboard.tsx";

function AppRoutes() {
  return (
    <Routes>
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <Home />
          </ProtectedRoute>
        }
      />
      <Route path="/login" element={<Login />} />
      <Route
        path="/statistics"
        element={
          <ProtectedRoute>
            <Statistics />
          </ProtectedRoute>
        }
      />
      <Route
        path="/history"
        element={
          <ProtectedRoute>
            <HistoryDashboard />
          </ProtectedRoute>
        }
      />
      <Route
        path="/admin/add-user"
        element={
          <ProtectedRoute adminOnly>
            <AddUser />
          </ProtectedRoute>
        }
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default AppRoutes;
