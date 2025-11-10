// static/script.js - Complete JavaScript Fix
console.log("🚀 Script loaded successfully!");

// Global variables
let currentResultsFile = null;

// File upload form handler
document.getElementById('uploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    console.log("📁 Form submitted");
    
    const formData = new FormData(this);
    const submitBtn = this.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    
    // Show loading state
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    submitBtn.disabled = true;
    
    try {
        console.log("🔄 Sending analysis request...");
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
        
        console.log("📨 Response received:", response.status);
        const result = await response.json();
        console.log("📊 Analysis result:", result);
        
        if (result.success) {
            displayResults(result);
        } else {
            showError(result.error || 'Unknown error occurred');
        }
    } catch (error) {
        console.error("❌ Network error:", error);
        showError('Network error: ' + error.message);
    } finally {
        // Reset button
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
    }
});

// Demo data analysis
async function useDemoData() {
    console.log("🎮 Using demo data");
    const button = document.querySelector('.btn-warning');
    const originalText = button.innerHTML;
    
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing Demo...';
    button.disabled = true;
    
    try {
        const response = await fetch('/demo_analysis');
        const result = await response.json();
        
        if (result.success) {
            displayResults(result);
        } else {
            showError(result.error);
        }
    } catch (error) {
        showError('Demo analysis failed: ' + error.message);
    } finally {
        button.innerHTML = originalText;
        button.disabled = false;
    }
}

// Display analysis results
function displayResults(result) {
    console.log("🎯 Displaying results:", result);
    
    // Update statistics
    const statsHtml = `
        <div class="stat-card">
            <div class="stat-number">${result.statistics.total_predictions.toLocaleString()}</div>
            <div class="stat-label">Total Predictions</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">${result.statistics.unique_videos}</div>
            <div class="stat-label">Unique Videos</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">${result.statistics.behaviors_detected}</div>
            <div class="stat-label">Behaviors Detected</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">${result.statistics.average_confidence}</div>
            <div class="stat-label">Avg Confidence</div>
        </div>
    `;
    document.getElementById('resultsStats').innerHTML = statsHtml;
    
    // Show main plot
    if (result.visualizations && result.visualizations.main_plot) {
        document.getElementById('plotContainer').innerHTML = 
            `<img src="${result.visualizations.main_plot}" style="width: 100%; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">`;
    } else {
        document.getElementById('plotContainer').innerHTML = 
            '<div class="alert alert-error">No visualization available</div>';
    }
    
    // Show sample predictions table
    if (result.sample_predictions && result.sample_predictions.length > 0) {
        let tableHtml = `
            <div class="table-responsive">
                <table class="table" style="width: 100%; border-collapse: collapse; margin-top: 20px;">
                    <thead>
                        <tr style="background: #f8f9fa;">
                            <th style="padding: 12px; border: 1px solid #dee2e6;">Video ID</th>
                            <th style="padding: 12px; border: 1px solid #dee2e6;">Frame</th>
                            <th style="padding: 12px; border: 1px solid #dee2e6;">Behavior</th>
                            <th style="padding: 12px; border: 1px solid #dee2e6;">Behavior Name</th>
                            <th style="padding: 12px; border: 1px solid #dee2e6;">Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        result.sample_predictions.forEach(pred => {
            tableHtml += `
                <tr>
                    <td style="padding: 10px; border: 1px solid #dee2e6;">${pred.video_id}</td>
                    <td style="padding: 10px; border: 1px solid #dee2e6;">${pred.frame}</td>
                    <td style="padding: 10px; border: 1px solid #dee2e6;">${pred.behavior}</td>
                    <td style="padding: 10px; border: 1px solid #dee2e6;">${pred.behavior_name}</td>
                    <td style="padding: 10px; border: 1px solid #dee2e6;">${(pred.confidence * 100).toFixed(1)}%</td>
                </tr>
            `;
        });
        
        tableHtml += `</tbody></table></div>`;
        document.getElementById('sampleTable').innerHTML = tableHtml;
    }
    
    // Store download info
    currentResultsFile = result.results_file;
    
    // Show results section
    document.getElementById('resultSection').style.display = 'block';
    document.getElementById('errorAlert').style.display = 'none';
    
    // Scroll to results
    document.getElementById('resultSection').scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
    });
    
    console.log("✅ Results displayed successfully");
}

// Download sample data
function downloadSample() {
    console.log("📥 Downloading sample data");
    window.open('/sample_data', '_blank');
}

// Download results
function downloadResults() {
    if (currentResultsFile) {
        const filename = currentResultsFile.split('/').pop();
        console.log("💾 Downloading results:", filename);
        window.open('/download/' + filename, '_blank');
    } else {
        showError('No results available to download');
    }
}

// Download project files
function downloadFile(category, filename) {
    console.log(`📁 Downloading ${category}/${filename}`);
    window.open(`/project_files/${category}/${filename}`, '_blank');
}

// Show error message
function showError(message) {
    console.error("❌ Error:", message);
    document.getElementById('errorAlert').innerHTML = `
        <strong><i class="fas fa-exclamation-triangle"></i> Error:</strong> ${message}
    `;
    document.getElementById('errorAlert').style.display = 'block';
    document.getElementById('resultSection').style.display = 'none';
    
    // Scroll to error
    document.getElementById('errorAlert').scrollIntoView({ 
        behavior: 'smooth',
        block: 'center'
    });
}

// Drag and drop functionality
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.querySelector('input[type="file"]');

if (uploadArea && fileInput) {
    // Highlight drop area when file is dragged over
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        uploadArea.classList.add('dragover');
    }
    
    function unhighlight() {
        uploadArea.classList.remove('dragover');
    }
    
    // Handle dropped files
    uploadArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
        
        // Trigger form submission
        if (files.length > 0) {
            document.getElementById('uploadForm').dispatchEvent(new Event('submit'));
        }
    }
}

console.log("✅ All JavaScript functions loaded successfully!");