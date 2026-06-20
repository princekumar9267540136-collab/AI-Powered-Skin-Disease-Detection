// server.js
const express = require('express');
const fileUpload = require('express-fileupload');
const { exec } = require('child_process');
const path = require('path');
const fs = require('fs');
// Development helpers
const cors = require('cors');

// Function to generate labels.json from DATASET folder
function generateLabelsFromDataset() {
    const datasetPath = path.join(__dirname, 'DATASET');
    const labelsPath = path.join(__dirname, 'backend', 'labels.json');

    if (!fs.existsSync(datasetPath)) {
        console.log('DATASET folder not found, skipping labels generation');
        return;
    }

    if (fs.existsSync(labelsPath)) {
        console.log('labels.json already exists, skipping generation');
        return;
    }

    try {
        const classes = fs.readdirSync(datasetPath)
            .filter(item => fs.statSync(path.join(datasetPath, item)).isDirectory())
            .sort(); // Sort alphabetically to match training order

        if (classes.length === 0) {
            console.log('No class folders found in DATASET');
            return;
        }

        fs.writeFileSync(labelsPath, JSON.stringify(classes, null, 2));
        console.log(`Generated labels.json with ${classes.length} classes:`, classes);
    } catch (error) {
        console.error('Error generating labels.json:', error);
    }
}

// Generate labels on startup
generateLabelsFromDataset();

const app = express();
// Use port from environment for deployment, fall back to 5000 for local development
const PORT = process.env.PORT || 5000;
const PYTHON_COMMAND = process.env.PYTHON_PATH || process.env.PYTHON || 'python';

// Allow CORS in development so the frontend (served from the same server or another origin)
// can call the /predict endpoint. Remove or restrict this in production.
app.use(cors());

// Parse JSON and URL-encoded request bodies
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Enable file uploads and create parent folders automatically
app.use(fileUpload({ createParentPath: true, limits: { fileSize: 10 * 1024 * 1024 }, abortOnLimit: true }));

// Set a permissive Content-Security-Policy for local development so DevTools and
// fetch/connect requests to localhost are not blocked. Tighten this for production.
app.use((req, res, next) => {
    res.setHeader("Content-Security-Policy",
        "default-src 'self' 'unsafe-inline' data:; connect-src 'self' http://localhost:5000; img-src 'self' data: http://localhost:5000;");
    next();
});

app.use(express.static('public'));
app.use(fileUpload());

// Simple health check to verify server reachability from the browser
app.get('/ping', (req, res) => {
    res.json({ ok: true });
});

// Route for detection
app.post('/upload', (req, res) => {
    if (!req.files || !req.files.image) {
        return res.send('No file uploaded');
    }

    let imageFile = req.files.image;
    let uploadPath = path.join(__dirname, 'public', imageFile.name);

    imageFile.mv(uploadPath, (err) => {
        if (err) return res.status(500).send(err);

        const predictScript = path.join(__dirname, 'backend', 'predict.py');
        exec(`python -u "${predictScript}" "${uploadPath}"`, { maxBuffer: 2 * 1024 * 1024 }, (error, stdout, stderr) => {
            if (error) {
                console.error(`Prediction error: ${stderr}`);
                return res.status(500).send(stderr);
            }
            res.send(`<h2>Prediction: ${stdout}</h2><a href="/detect.html">Try another image</a>`);
        });
    });
});

// JSON /predict endpoint used by the frontend (expects form field name 'skinImage')
app.post('/predict', (req, res) => {
    try {
        // Check if any files were uploaded
        if (!req.files || Object.keys(req.files).length === 0) {
            return res.status(400).json({ error: 'No files were uploaded.' });
        }

        // Get the image file from the 'skinImage' field
        const imageFile = req.files.skinImage;
        if (!imageFile) {
            return res.status(400).json({ error: 'Missing skinImage field in upload.' });
        }

        // Create a safe filename from the original name
        const safeFileName = imageFile.name.replace(/[^a-zA-Z0-9.-]/g, '_');
        
        // Save to a unique filename to avoid conflicts
        const timestamp = Date.now();
        const uploadPath = path.join(__dirname, 'uploads', `${timestamp}_${safeFileName}`);

        // Create uploads directory if it doesn't exist
        const uploadsDir = path.join(__dirname, 'uploads');
        if (!require('fs').existsSync(uploadsDir)) {
            require('fs').mkdirSync(uploadsDir);
        }

        imageFile.mv(uploadPath, (err) => {
            if (err) {
                console.error('File save error:', err);
                return res.status(500).json({ error: 'Failed to save uploaded file.' });
            }

            const predictScript = path.join(__dirname, 'backend', 'predict.py');
            console.log('Debug: Predict script path:', predictScript);
            console.log('Debug: Image upload path:', uploadPath);
            console.log('Debug: Current directory:', __dirname);

            const execEnv = Object.assign({}, process.env, { PYTHONIOENCODING: 'utf-8' });
            exec(`python -u "${predictScript}" "${uploadPath}"`, { maxBuffer: 2 * 1024 * 1024, env: execEnv }, (error, stdout, stderr) => {
                console.log('Debug: Python stdout:', stdout);
                if (stderr) console.log('Debug: Python stderr:', stderr);

                // Clean up the uploaded file after prediction
                try {
                    if (fs.existsSync(uploadPath)) {
                        fs.unlinkSync(uploadPath);
                    }
                } catch (unlinkErr) {
                    console.error('Failed to clean up uploaded file:', unlinkErr);
                }

                if (error) {
                    console.error('Python script error:', error);
                    console.error('Error details from stderr:', stderr);

                    try {
                        const parsed = JSON.parse(stdout || '{}');
                        if (parsed && parsed.error) {
                            return res.status(500).json(parsed);
                        }
                    } catch (e) {
                        // ignore parse error
                    }

                    return res.status(500).json({
                        error: 'Model prediction failed',
                        details: stderr || error.message
                    });
                }

                try {
                    const parsed = JSON.parse(String(stdout || '{}'));
                    if (parsed.error) {
                        return res.status(500).json(parsed);
                    }
                    if (parsed.prediction && typeof parsed.confidence !== 'undefined') {
                        return res.json(parsed);
                    }
                    if (parsed.prediction) {
                        return res.json({ prediction: parsed.prediction, confidence: parsed.confidence || 1.0 });
                    }
                    return res.status(500).json({ error: 'Model returned unexpected result', raw: parsed });
                } catch (parseError) {
                    console.error('Failed to parse JSON output from Python:', parseError);
                    return res.status(500).json({ error: 'Failed to parse prediction result' });
                }
            });
        });
    } catch (ex) {
        console.error('Unexpected error in /predict:', ex);
        return res.status(500).json({ error: 'Unexpected server error.' });
    }
});

// Chatbot endpoint - simple local responses
app.post('/chat', (req, res) => {
    try {
        const { message } = req.body;
        
        if (!message) {
            return res.status(400).json({ error: 'No message provided' });
        }

        // Simple chatbot responses for skin health questions
        const lowerMessage = message.toLowerCase();
        let response = '';

        // Keyword-based responses for common skin questions
        if (lowerMessage.includes('acne') || lowerMessage.includes('pimple')) {
            response = 'Acne is a common skin condition caused by clogged pores with oil and dead skin cells. Tips: Keep skin clean, avoid touching your face, use appropriate skincare products, stay hydrated, and consider consulting a dermatologist for persistent cases.';
        } else if (lowerMessage.includes('dry skin') || lowerMessage.includes('dryness')) {
            response = 'Dry skin can be managed by: using gentle cleansers, moisturizing daily, avoiding hot water, staying hydrated, and using products with hyaluronic acid or glycerin. If severe, consult a dermatologist.';
        } else if (lowerMessage.includes('eczema')) {
            response = 'Eczema is a chronic inflammatory skin condition. Management includes: moisturizing regularly, avoiding irritants, using mild soaps, managing stress, and seeing a dermatologist for treatment options.';
        } else if (lowerMessage.includes('psoriasis')) {
            response = 'Psoriasis is an autoimmune condition causing skin lesions. Treatments include topical creams, phototherapy, and oral medications. Consult a dermatologist for proper diagnosis and treatment plan.';
        } else if (lowerMessage.includes('fungal') || lowerMessage.includes('ringworm')) {
            response = 'Fungal infections require antifungal treatment. Common options include topical antifungal creams, oral medications, and maintaining proper hygiene. See a dermatologist for diagnosis and treatment.';
        } else if (lowerMessage.includes('sun') || lowerMessage.includes('uv') || lowerMessage.includes('sunscreen')) {
            response = 'Sun protection is essential: Use SPF 30+ sunscreen daily, reapply every 2 hours, wear protective clothing, avoid peak sun hours (10am-4pm), and seek shade. This helps prevent skin damage and cancer.';
        } else if (lowerMessage.includes('age') || lowerMessage.includes('wrinkle') || lowerMessage.includes('anti')) {
            response = 'Anti-aging tips: Use retinoids, vitamin C serums, moisturizers, sunscreen daily, stay hydrated, sleep well, reduce stress, and exercise. Professional treatments like microdermabrasion or chemical peels can help.';
        } else if (lowerMessage.includes('sensitive')) {
            response = 'For sensitive skin: Use fragrance-free, hypoallergenic products, avoid harsh soaps, patch test new products, use moisturizer regularly, and minimize use of active ingredients. Consult a dermatologist if you have persistent issues.';
        } else {
            response = 'I can help you with skin care information. Ask me about common conditions like acne, eczema, psoriasis, dry skin, fungal infections, sun protection, or anti-aging tips. For serious concerns, always consult a dermatologist.';
        }

        res.json({ response });
    } catch (ex) {
        console.error('Chatbot error:', ex);
        return res.status(500).json({ error: 'Chatbot service error' });
    }
});

// Add error handling for server startup
const server = app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
}).on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
        console.error(`Error: Port ${PORT} is already in use. Try a different port.`);
    } else {
        console.error('Server failed to start:', err);
    }
});
