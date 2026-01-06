import { Routes, Route } from "react-router-dom";
import Home from "../pages/Home.tsx";
import Login from "../pages/Login.tsx";
import AddUser from "../pages/AddUser.tsx";

function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/login" element={<Login />} />
      <Route path="/admin/add-user" element={<AddUser />} />
    </Routes>
  );
}

export default AppRoutes;
