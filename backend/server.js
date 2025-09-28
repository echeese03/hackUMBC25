const express = require("express");
const cors = require("cors");
const fs = require("fs").promises;
const path = require("path");
const axios = require("axios");

const app = express();
const PORT = 3000;
const PYTHON_API_URL = "http://localhost:8000/classify/";
const UPLOADS_DIR = path.join(process.cwd(), "uploads");

app.use(cors());
app.use(express.json({ limit: "10mb" }));

// Ensure uploads directory exists
(async () => {
  try {
    await fs.mkdir(UPLOADS_DIR, { recursive: true });
  } catch (err) {
    console.error("Failed to create uploads directory:", err.message);
    process.exit(1);
  }
})();

// POST /upload
app.post("/upload", async (req, res) => {
  const { image } = req.body;

  if (!image) {
    return res.status(400).json({ error: "No image provided" });
  }

  const base64Data = image.replace(/^data:image\/jpeg;base64,/, "");
  const fileName = `frame_${Date.now()}.jpeg`;
  const filePath = path.join(UPLOADS_DIR, fileName);

  try {
    // Save image to disk
    await fs.writeFile(filePath, base64Data, "base64");

    // Send path to Python service
    const pythonResponse = await axios.post(PYTHON_API_URL, {
      image_path: filePath
    });

    console.log("Python classification response:", pythonResponse.data);

    // Send result back to client
    res.json({
      file: fileName,
      classification: pythonResponse.data
    });

  } catch (err) {
    console.error("Error handling upload:", err.message);

    if (err.response) {
      console.error("Python service error:", err.response.status, err.response.data);
    } else if (err.request) {
      console.error("No response from Python service");
    }

    res.status(500).json({ error: "Failed to process image" });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Backend running at http://localhost:${PORT}`);
});
