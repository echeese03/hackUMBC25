import React, { useRef } from "react";
import axios from "axios";
import Webcam from "react-webcam";

function WebcamComponent() {
  const webcamRef = useRef(null);

  const getBackendURL = () => {
    const hostname = window.location.hostname;
    if (hostname === "localhost") return "http://localhost:3000/upload";
    return "https://labored-margeret-evincible.ngrok-free.dev/upload";
  };

  const capture = async () => {
    if (!webcamRef.current) return;

    // Small timeout to let mobile camera stabilize
    setTimeout(async () => {
      const imageSrc = webcamRef.current.getScreenshot();
      if (!imageSrc) {
        alert("Failed to capture image. Try again.");
        return;
      }

      try {
        const backendURL = getBackendURL();
        const response = await axios.post(backendURL, { image: imageSrc });
        alert(`Saved image as: ${response.data.file}`);
      } catch (err) {
        console.error("Error uploading image:", err);
        alert("Failed to save image to backend.");
      }
    }, 100); // 100ms delay for mobile stabilization
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
