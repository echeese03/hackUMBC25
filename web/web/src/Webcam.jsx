import React, { useRef, useState } from "react";
import axios from "axios";
import Webcam from "react-webcam";

function WebcamComponent() {
  const webcamRef = useRef(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [lastCapture, setLastCapture] = useState(null);

  const capture = async () => {
    if (!webcamRef.current) {
      alert("Webcam not ready. Please allow camera access.");
      return;
    }

    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) {
      alert("Failed to capture image. Please try again.");
      return;
    }

    setIsCapturing(true);
    
    try {
      const response = await axios.post('/upload', { 
        image: imageSrc 
      }, {
        timeout: 10000 // 10 second timeout
      });
      
      console.log("✅ Image saved:", response.data);
      setLastCapture(response.data.file);
      
      alert(`✅ Image saved successfully!\nFilename: ${response.data.file}`);
      
    } catch (err) {
      console.error("❌ Upload error:", err);
      
      let errorMessage = "Failed to save image";
      if (err.response?.data?.error) {
        errorMessage = err.response.data.error;
      } else if (err.code === 'NETWORK_ERROR') {
        errorMessage = "Network error. Check your connection.";
      } else if (err.code === 'TIMEOUT') {
        errorMessage = "Upload timeout. Server may be unavailable.";
      }
      
      alert(`❌ ${errorMessage}`);
    } finally {
      setIsCapturing(false);
    }
  };

  const videoConstraints = {
    facingMode: "user",
    width: { ideal: 1280 },
    height: { ideal: 720 }
  };

  return (
    <div className="flex flex-col items-center justify-center mt-5">
      <div className="relative">
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          videoConstraints={videoConstraints}
          className="rounded-lg shadow-md max-w-full h-auto"
          screenshotQuality={0.8}
        />
        {isCapturing && (
          <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center rounded-lg">
            <div className="text-white font-bold">Saving...</div>
          </div>
        )}
      </div>
      
      <button
        onClick={capture}
        disabled={isCapturing}
        className={`mt-4 px-6 py-2 text-white rounded transition ${
          isCapturing 
            ? 'bg-gray-400 cursor-not-allowed' 
            : 'bg-blue-600 hover:bg-blue-700'
        }`}
      >
        {isCapturing ? 'Capturing...' : 'Capture Frame'}
      </button>

      {lastCapture && (
        <div className="mt-2 text-sm text-green-600">
          Last capture: {lastCapture}
        </div>
      )}
    </div>
  );
}

export default WebcamComponent;