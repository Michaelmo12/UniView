import { Routes, Route } from "react-router-dom";
import { ProtectedRoute } from "../components/common";
import Home from "../pages/Home.tsx";
import Login from "../pages/Login.tsx";
import AddUser from "../pages/AddUser.tsx";
import Surveillance from "../pages/Surveillance.tsx";

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
        path="/admin/add-user"
        element={
          <ProtectedRoute adminOnly>
            <AddUser />
          </ProtectedRoute>
        }
      />
      <Route
        path="/surveillance"
        element={
          <ProtectedRoute>
            <Surveillance />
          </ProtectedRoute>
        }
      />
    </Routes>
  );
}

export default AppRoutes;
