const express = require("express");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const axios = require("axios");

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
app.post('/upload', async (req, res) => {
    console.log('📸 Upload endpoint hit');
    
    const { image } = req.body;
    
    if (!image) {
        return res.status(400).json({ error: 'No image provided' });
    }

    try {
        const base64Data = image.replace(/^data:image\/jpeg;base64,/, "");
        const fileName = `frame_${Date.now()}.jpeg`;
        const filePath = path.join(UPLOADS_DIR, fileName);

        // Save the image first
        await fs.promises.writeFile(filePath, base64Data, "base64");
        console.log('✅ Image saved:', fileName);

        // Then call Python FastAPI service
        try {
            const pythonResponse = await axios.post("http://localhost:8000/classify/", {
                image_path: filePath
            }, {
                timeout: 10000
            });
            
            console.log("Python classification response:", pythonResponse.data);
            
            res.json({ 
                success: true, 
                file: fileName,
                classification: pythonResponse.data,
                message: 'Image saved and classified successfully!'
            });
            
        } catch (pythonError) {
            console.error('❌ Python service error:', pythonError.message);
            res.json({ 
                success: true, 
                file: fileName,
                classification: null,
                error: 'Classification service unavailable',
                message: 'Image saved but classification failed'
            });
        }

    } catch (error) {
        console.error('❌ Upload error:', error);
        res.status(500).json({ 
            error: 'Server error during image processing',
            details: error.message 
        });
    }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({ status: 'OK', message: 'Server is running' });
});

// Serve static files from React build (if it exists)
const ReactBuildPath = path.join(__dirname, 'dist');
if (fs.existsSync(ReactBuildPath)) {
    app.use(express.static(ReactBuildPath));
    console.log('📁 Serving React app from:', ReactBuildPath);
    
    // Simple root route for React app
    app.get('/', (req, res) => {
        res.sendFile(path.join(ReactBuildPath, 'index.html'));
    });
} else {
    console.log('⚠️  React build not found at:', ReactBuildPath);
    
    // Basic root route when no React build exists
    app.get('/', (req, res) => {
        res.send(`
            <!DOCTYPE html>
            <html>
            <head>
                <title>Webcam Server</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    h1 { color: green; }
                    .container { max-width: 800px; margin: 0 auto; }
                    .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Webcam Server is Running! ✅</h1>
                    <p>Server is ready to receive image uploads.</p>
                    
                    <div class="endpoint">
                        <strong>Upload Endpoint:</strong> POST /upload
                    </div>
                    
                    <div class="endpoint">
                        <strong>Health Check:</strong> GET /api/health
                    </div>
                    
                    <p><em>React app not built yet. Run: npm run build in frontend directory</em></p>
                </div>
            </body>
            </html>
        `);
    });
}

// Start server
app.listen(PORT, () => {
    console.log(`🚀 Server running on http://localhost:${PORT}`);
    console.log(`💾 Uploads directory: ${UPLOADS_DIR}`);
    console.log('✅ Ready for image uploads!');
    console.log(`📊 Health check: http://localhost:${PORT}/api/health`);
});
