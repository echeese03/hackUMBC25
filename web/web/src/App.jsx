import React, { useState, useEffect } from "react";
import "./App.css";
import WebcamComponent from "./Webcam";
import QRCode from "react-qr-code";

function App() {
  const [currentUrl, setCurrentUrl] = useState("");
  const [isOnline, setIsOnline] = useState(true);

  // Get the current URL for QR code
  useEffect(() => {
    setCurrentUrl(window.location.origin);
    
    // Check if backend is reachable
    checkBackendStatus();
  }, []);

  const checkBackendStatus = async () => {
    try {
      const response = await fetch('/api/health');
      if (response.ok) {
        setIsOnline(true);
      } else {
        setIsOnline(false);
      }
    } catch (error) {
      console.error('Backend connection failed:', error);
      setIsOnline(false);
    }
  };

  const handleRefresh = () => {
    window.location.reload();
  };

  const handleTestUpload = async () => {
    try {
      // Create a test image
      const canvas = document.createElement('canvas');
      canvas.width = 100;
      canvas.height = 100;
      const ctx = canvas.getContext('2d');
      
      // Draw a simple test image
      ctx.fillStyle = '#4F46E5';
      ctx.fillRect(0, 0, 100, 100);
      ctx.fillStyle = 'white';
      ctx.font = '20px Arial';
      ctx.fillText('Test', 20, 50);
      
      const testImage = canvas.toDataURL('image/jpeg');
      
      const response = await fetch('/classify/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: testImage })
      });
      
      const result = await response.json();
      
      if (result.success) {
        alert(`✅ Test upload successful!\nFile: ${result.file}`);
      } else {
        alert(`❌ Test upload failed: ${result.error}`);
      }
    } catch (error) {
      alert(`❌ Test upload error: ${error.message}`);
    }
  };

  return (
    <div className="App p-4 max-w-md mx-auto">
      <h1 className="text-3xl font-bold text-center mb-2">Webcam Demo</h1>
      
      {/* Status indicator */}
      <div className={`text-center mb-4 p-2 rounded ${
        isOnline ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
      }`}>
        Status: {isOnline ? '✅ Online' : '❌ Offline'}
        {!isOnline && (
          <button 
            onClick={checkBackendStatus}
            className="ml-2 px-2 py-1 bg-blue-500 text-white rounded text-sm"
          >
            Retry
          </button>
        )}
      </div>

      {/* Connection info */}
      <div className="bg-gray-100 p-3 rounded mb-4">
        <p className="text-sm break-all">
          <strong>Server:</strong> {currentUrl}
        </p>
        <button 
          onClick={handleTestUpload}
          className="mt-2 px-3 py-1 bg-green-500 text-white rounded text-sm"
        >
          Test Upload
        </button>
        <button 
          onClick={handleRefresh}
          className="mt-2 ml-2 px-3 py-1 bg-gray-500 text-white rounded text-sm"
        >
          Refresh
        </button>
      </div>

      <WebcamComponent />
      
      <div className="mt-8 text-center">
        <p className="mb-2 font-medium">Scan to open on other devices:</p>
        <div className="flex justify-center">
          <QRCode value={currentUrl} size={128} />
        </div>
        <p className="mt-2 text-sm text-gray-600 break-all">
          {currentUrl}
        </p>
        
        <div className="mt-4 text-xs text-gray-500">
          <p>Make sure all devices are on the same network</p>
          <p>Or use ngrok for external access</p>
        </div>
      </div>

      {/* Instructions */}
      <div className="mt-6 p-3 bg-blue-50 rounded">
        <h3 className="font-bold mb-2">How to use:</h3>
        <ol className="text-sm list-decimal list-inside space-y-1">
          <li>Allow camera access when prompted</li>
          <li>Click "Capture Frame" to take a photo</li>
          <li>Image saves automatically to server</li>
          <li>Scan QR code to open on other devices</li>
        </ol>
      </div>
    </div>
  );
}

export default App;