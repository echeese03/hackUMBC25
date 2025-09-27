import express from "express";
import cors from "cors";
import fs from "fs";
import path from "path";

const app = express();
app.use(cors()); // Allow all origins (or restrict to your ngrok / IP)
app.use(express.json({ limit: "10mb" }));

const UPLOADS_DIR = path.join(process.cwd(), "uploads");
if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR);

app.post("/upload", (req, res) => {
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
});

app.listen(3000, () => console.log("Backend running at http://localhost:3000"));
