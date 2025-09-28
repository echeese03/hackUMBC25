const express = require("express");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const axios = require('axios');

const app = express();
app.use(cors()); // Allow all origins (or restrict to your ngrok / IP)
app.use(express.json({ limit: "10mb" }));

const UPLOADS_DIR = path.join(process.cwd(), "uploads");
if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR);

app.post("/upload", async (req, res) => {
  const { image } = req.body;
  const base64Data = image.replace(/^data:image\/jpeg;base64,/, "");
  const fileName = `frame_${Date.now()}.jpeg`;
  const filePath = path.join(UPLOADS_DIR, fileName);

  fs.writeFile(filePath, base64Data, "base64", (err) => {
    if (err) {
      console.error(err);
      return res.status(500).json({ error: "Failed to save image" });
    }
    res.json({ file: fileName });
  });

  // Call Python FastAPI service with the saved image path
  const pythonResponse = await axios.post("http://localhost:8000/classify/", {
    image_path: filePath
  });

  console.log("Python classification response:", pythonResponse.data);

  res.json({ classification: pythonResponse.data });
});

app.listen(3000, () => console.log("Backend running at http://localhost:3000"));
