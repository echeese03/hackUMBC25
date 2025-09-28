import React, { useRef, useState } from "react";
import axios from "axios";
import Webcam from "react-webcam";

function WebcamComponent() {
  const webcamRef = useRef(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [classification, setClassification] = useState(null);
  const [blurb, setBlurb] = useState(""); // State for the blurb

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
      const response = await axios.post('http://localhost:8000/classify/', { 
        image: imageSrc
      }, {
        timeout: 20000 // 10 second timeout
      });
      
      console.log("✅ Classification result:", response.data);
      
      if (response.data.success) {
        // Set the classification and blurb from response
        setClassification(response.data.classification);
        setBlurb(response.data.blurb);
        print(response.data.blurb);
        alert(`✅ Item classified as: ${response.data.classification.toUpperCase()}`);
      } else {
        alert(`❌ Classification failed: ${response.data.error}`);
      }
      
    } catch (err) {
      console.error("❌ Classification error:", err);
      
      let errorMessage = "Failed to classify image";
      if (err.response?.data?.error) {
        errorMessage = err.response.data.error;
      } else if (err.code === 'ECONNABORTED') {
        errorMessage = "Request timeout. Server may be unavailable.";
      } else if (err.message === 'Network Error') {
        errorMessage = "Network error. Is the server running on port 8000?";
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
            <div className="text-white font-bold">Classifying...</div>
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
        {isCapturing ? 'Classifying...' : 'Capture & Classify'}
      </button>
      {classification && (
        <div className="mt-4 p-4 bg-green-100 rounded-lg">
          <h3 className="font-bold text-lg">Classification Result:</h3>
          <p className="text-2xl text-green-700 capitalize font-bold">{classification}</p>
          {blurb && ( // Display the blurb if it's available
            <p className="text-sm text-gray-600 mt-1">
              Info: {blurb}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export default WebcamComponent;