// src/App.jsx
import React from "react";
import './App.css';
import WebcamComponent from "./Webcam";
// Optional: QR code to open on phone
import QRCode from "react-qr-code";

function App() {
  const ngrokURL = "https://labored-margeret-evincible.ngrok-free.dev";

  return (
    <div className="App p-4 max-w-md mx-auto">
      <h1 className="text-3xl font-bold text-center mb-6">Webcam Demo</h1>
      <WebcamComponent />
      <div className="mt-8 text-center">
        <p className="mb-2">Scan this QR to open on your phone:</p>
        <QRCode value={ngrokURL} size={128} />
      </div>
    </div>
  );
}

export default App;
