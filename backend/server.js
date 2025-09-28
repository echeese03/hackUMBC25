const express = require('express');
const fs = require('fs');
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

// Enable CORS for all origins
app.use(cors());
app.use(express.json({ limit: '100mb' }));

// Create uploads directory
const UPLOADS_DIR = path.join(__dirname, 'uploads');
if (!fs.existsSync(UPLOADS_DIR)) {
    fs.mkdirSync(UPLOADS_DIR, { recursive: true });
    console.log('📁 Created uploads directory');
}

// Upload endpoint
app.post('/upload', (req, res) => {
    console.log('📸 Upload endpoint hit');
    
    const { image } = req.body;
    
    if (!image) {
        return res.status(400).json({ error: 'No image provided' });
    }

    try {
        const base64Data = image.replace(/^data:image\/jpeg;base64,/, "");
        const fileName = `frame_${Date.now()}.jpeg`;
        const filePath = path.join(UPLOADS_DIR, fileName);

        fs.writeFile(filePath, base64Data, "base64", (err) => {
            if (err) {
                console.error('Error saving image:', err);
                return res.status(500).json({ error: 'Failed to save image' });
            }
            console.log('✅ Image saved:', fileName);
            res.json({ 
                success: true, 
                file: fileName,
                message: 'Image saved successfully!'
            });
        });
    } catch (error) {
        console.error('Upload error:', error);
        res.status(500).json({ error: 'Server error' });
    }
});

// Serve static files from the React build
const ReactBuildPath = path.join(__dirname, 'dist');
if (fs.existsSync(ReactBuildPath)) {
    app.use(express.static(ReactBuildPath));
    console.log('📁 Serving React app from:', ReactBuildPath);
} else {
    console.log('⚠️  React build not found at:', ReactBuildPath);
}

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({ status: 'OK', message: 'Server is running' });
});

// FIXED: Use regex for catch-all route instead of '*'
app.get(/^\/(?!api).*/, (req, res) => {
    if (fs.existsSync(ReactBuildPath)) {
        res.sendFile(path.join(ReactBuildPath, 'index.html'));
    } else {
        res.send(`
            <!DOCTYPE html>
            <html>
            <head><title>Webcam Server</title></head>
            <body>
                <h1>Webcam Server is Running! ✅</h1>
                <p>Upload endpoint: POST /upload</p>
                <p>React app not built yet. Run: npm run build in frontend</p>
            </body>
            </html>
        `);
    }
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(`🚀 Server running on http://localhost:${PORT}`);
    console.log(`💾 Uploads directory: ${UPLOADS_DIR}`);
    console.log('✅ Ready for image uploads!');
});