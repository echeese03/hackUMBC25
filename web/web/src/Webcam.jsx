import React, { useRef } from "react";
import axios from "axios";
import Webcam from "react-webcam";

function WebcamComponent() {
  const webcamRef = useRef(null);

  // Automatically determine backend URL
  const getBackendURL = () => {
    // If running on localhost (laptop)
    if (window.location.hostname === "localhost") {
      return "http://localhost:3000/upload";
    } else {
      // Running on phone via ngrok or LAN
      // Replace with your laptop's IP (or ngrok URL)
      return "http://192.168.X.Y:3000/upload"; // <-- Replace X.Y with laptop IP
    }
  };

  const capture = async () => {
    if (!webcamRef.current) return;

    const imageSrc = webcamRef.current.getScreenshot();

    if (!imageSrc) return alert("Failed to capture image.");

    try {
      const backendURL = getBackendURL();

      const response = await axios.post(backendURL, { image: imageSrc });
      console.log("Saved image to backend:", response.data);
      alert(`Image saved: ${response.data.file}`);
    } catch (err) {
      console.error("Error uploading image:", err);
      alert("Failed to save image to backend.");
    }
  };

  return (
    <div className="flex flex-col items-center justify-center mt-5">
      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        videoConstraints={{ facingMode: "user" }}
        className="rounded-lg shadow-md"
      />
      <button
        onClick={capture}
        className="mt-4 px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
      >
        Capture Frame
      </button>
    </div>
  );
}

export default WebcamComponent;
