import { Routes, Route } from "react-router-dom";
import { ProtectedRoute } from "../components/common";
import Home from "../pages/Home.tsx";
import Login from "../pages/Login.tsx";
import AddUser from "../pages/AddUser.tsx";
import Statistics from "../pages/Statistics.tsx";

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
        path="/admin/add-user"
        element={
          <ProtectedRoute adminOnly>
            <AddUser />
          </ProtectedRoute>
        }
      />
    </Routes>
  );
}

export default AppRoutes;
