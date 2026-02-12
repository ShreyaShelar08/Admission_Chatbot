import React, { useState, useEffect } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import Login from "./components/Login";
import Signup from "./components/Signup";
import Dashboard from "./components/Dashboard";

function App() {
  const [user, setUser] = useState(null);
  const [isCheckingAuth, setIsCheckingAuth] = useState(true);

  useEffect(() => {
    // Check if user is logged in
    const currentUser = localStorage.getItem("currentUser");
    if (currentUser) {
      try {
        setUser(JSON.parse(currentUser));
      } catch (error) {
        console.error("Error parsing user data:", error);
        localStorage.removeItem("currentUser");
      }
    }
    setIsCheckingAuth(false);
  }, []);

  // Show loading while checking authentication
  if (isCheckingAuth) {
    return <div>Loading...</div>;
  }

  return (
    <Routes>
      <Route
        path="/"
        element={user ? <Navigate to="/dashboard" replace /> : <Login setUser={setUser} />}
      />
      <Route path="/signup" element={<Signup setUser={setUser} />} />
      <Route
        path="/dashboard"
        element={
          user ? (
            <Dashboard user={user} setUser={setUser} />
          ) : (
            <Navigate to="/" replace />
          )
        }
      />
    </Routes>
  );
}

export default App;