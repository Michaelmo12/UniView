import { Routes, Route } from "react-router-dom";
import { ProtectedRoute } from "../components/common";
import Home from "../pages/Home.tsx";
import Login from "../pages/Login.tsx";
import AddUser from "../pages/AddUser.tsx";

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
    </Routes>
  );
}

export default AppRoutes;
