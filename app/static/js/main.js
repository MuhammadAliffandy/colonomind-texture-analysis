document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnSpinner = document.getElementById('btnSpinner');
    const btnText = analyzeBtn.querySelector('span');
    
    // Result Elements
    const resultCard = document.getElementById('resultCard');
    const mayoBadge = document.getElementById('mayoBadge');
    const mayoScoreText = document.getElementById('mayoScoreText');
    const mayoLabelText = document.getElementById('mayoLabelText');
    const confidenceText = document.getElementById('confidenceText');
    const confidenceFill = document.getElementById('confidenceFill');
    
    // Metric Elements
    const metrics = {
        contrast: document.getElementById('valContrast'),
        homogeneity: document.getElementById('valHomogeneity'),
        glcmEnergy: document.getElementById('valGlcmEnergy'),
        correlation: document.getElementById('valCorrelation'),
        dwtLL: document.getElementById('valDwtLL'),
        dwtHH: document.getElementById('valDwtHH')
    };

    let currentFile = null;

    // --- Drag and Drop Handlers ---
    
    uploadZone.addEventListener('click', () => fileInput.click());

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        // Validate file type
        const validTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff'];
        if (!validTypes.includes(file.type)) {
            alert('Please upload a valid image file (JPEG, PNG, BMP, TIFF).');
            return;
        }

        currentFile = file;
        
        // Show Preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove('hidden');
            analyzeBtn.disabled = false;
            
            // Hide previous results
            resultCard.classList.add('hidden');
            clearMetrics();
        };
        reader.readAsDataURL(file);
    }

    // --- API Integration ---

    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI Loading State
        analyzeBtn.disabled = true;
        btnText.textContent = 'Analyzing...';
        btnSpinner.classList.remove('hidden');
        resultCard.classList.add('hidden');

        const formData = new FormData();
        formData.append('image', currentFile);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Server error occurred');
            }

            renderResults(data);

        } catch (error) {
            console.error('Inference Error:', error);
            alert(`Analysis Failed: ${error.message}`);
        } finally {
            // Restore UI State
            analyzeBtn.disabled = false;
            btnText.textContent = 'Analyze Image';
            btnSpinner.classList.add('hidden');
        }
    });

    // --- Result Rendering ---

    function renderResults(data) {
        // 1. Update Mayo Badge
        const score = data.mayo_score;
        mayoScoreText.textContent = `Mayo ${score}`;
        
        // Extract just the label word (e.g., "Mayo 2 — Moderate" -> "Moderate")
        const labelParts = data.mayo_label.split('—');
        mayoLabelText.textContent = labelParts.length > 1 ? labelParts[1].trim() : data.mayo_label;

        // Apply Color Theme
        mayoBadge.className = `mayo-badge mayo-${score}`;
        confidenceFill.className = `progress-fill mayo-${score}`;

        // 2. Update Confidence Bar
        const confPercent = (data.confidence * 100).toFixed(1);
        confidenceText.textContent = `${confPercent}%`;
        
        // Animate width
        setTimeout(() => {
            confidenceFill.style.width = `${confPercent}%`;
        }, 100);

        resultCard.classList.remove('hidden');

        // 3. Update Metrics Grid
        const m = data.texture_metrics;
        metrics.contrast.textContent = m.glcm_contrast_mean.toFixed(2);
        metrics.homogeneity.textContent = m.glcm_homogeneity_mean.toFixed(4);
        metrics.glcmEnergy.textContent = m.glcm_energy_mean.toFixed(4);
        metrics.correlation.textContent = m.glcm_correlation_mean.toFixed(4);
        metrics.dwtLL.textContent = formatLargeNumber(m.dwt_ll_energy);
        metrics.dwtHH.textContent = m.dwt_hh_variance.toFixed(2);

        // 4. Render 3D UMAP
        if (data.umap_embedding && data.umap_embedding.length >= 3) {
            renderUmap3D(data.umap_embedding, score);
        }
    }

    function clearMetrics() {
        Object.values(metrics).forEach(el => el.textContent = '--');
        document.getElementById('plotlyContainer').innerHTML = '<div class="placeholder-text">Upload an image to visualize its embedding in the clinical feature space.</div>';
    }

    function formatLargeNumber(num) {
        if (num > 10000) return (num / 1000).toFixed(1) + 'k';
        return num.toFixed(2);
    }

    // --- Plotly 3D Scatter ---

    function renderUmap3D(coords, mayoScore) {
        const [u1, u2, u3] = coords;
        
        // Colors mapping to CSS variables
        const colors = ['#10b981', '#fbbf24', '#f97316', '#ef4444'];
        const markerColor = colors[mayoScore] || '#38bdf8';

        const trace = {
            x: [u1],
            y: [u2],
            z: [u3],
            mode: 'markers',
            marker: {
                size: 12,
                color: markerColor,
                line: {
                    color: '#ffffff',
                    width: 2
                },
                opacity: 0.9
            },
            type: 'scatter3d',
            name: `Mayo ${mayoScore}`,
            hoverinfo: 'name+x+y+z'
        };

        const layout = {
            margin: { l: 0, r: 0, b: 0, t: 0 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            scene: {
                xaxis: { title: 'UMAP 1', backgroundcolor: 'rgba(0,0,0,0)', gridcolor: '#334155', showbackground: false },
                yaxis: { title: 'UMAP 2', backgroundcolor: 'rgba(0,0,0,0)', gridcolor: '#334155', showbackground: false },
                zaxis: { title: 'UMAP 3', backgroundcolor: 'rgba(0,0,0,0)', gridcolor: '#334155', showbackground: false },
                camera: {
                    eye: { x: 1.5, y: 1.5, z: 1.2 }
                }
            }
        };

        const config = {
            displayModeBar: false,
            responsive: true
        };

        // Clear placeholder and render
        document.getElementById('plotlyContainer').innerHTML = '';
        Plotly.newPlot('plotlyContainer', [trace], layout, config);
    }
});
