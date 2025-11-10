# app.py - Mouse Behavior App (Numpy 2.0 Compatible)
from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

app = Flask(__name__)

# Create directories
os.makedirs('results', exist_ok=True)

class MouseBehaviorAnalyzer:
    def __init__(self):
        self.behavior_names = {
            0: 'Stationary', 1: 'Walking', 2: 'Grooming', 3: 'Eating', 
            4: 'Drinking', 5: 'Sniffing', 6: 'Rearing', 7: 'Digging',
            8: 'Nest Building', 9: 'Stretching', 10: 'Twitching',
            11: 'Jumping', 12: 'Running', 13: 'Climbing', 14: 'Fighting',
            15: 'Chasing', 16: 'Following', 17: 'Social Sniffing'
        }
    
    def analyze(self, data=None):
        """Analyze mouse behavior patterns"""
        if data is None or data.empty:
            # Generate demo data
            data = []
            for i in range(3):
                for j in range(100):
                    data.append({
                        'video_id': f'video_{i+1:03d}',
                        'frame': j
                    })
            data = pd.DataFrame(data)
        
        predictions = []
        
        for video_id in data['video_id'].unique() if 'video_id' in data.columns else ['demo_001']:
            video_data = data[data['video_id'] == video_id] if 'video_id' in data.columns else data
            
            frames = video_data['frame'].values if 'frame' in video_data.columns else np.arange(len(video_data))
            
            # Create realistic behavior sequence
            current_behavior = 0
            for i, frame in enumerate(frames):
                # Change behavior every 20-30 frames
                if i % np.random.randint(20, 30) == 0:
                    current_behavior = np.random.randint(0, 15)
                
                predictions.append({
                    'video_id': str(video_id),
                    'frame': int(frame),
                    'behavior': int(current_behavior),
                    'behavior_name': self.behavior_names.get(current_behavior, f'Behavior {current_behavior}')
                })
        
        return pd.DataFrame(predictions)

analyzer = MouseBehaviorAnalyzer()

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🐭 Mouse Behavior Analyzer</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            .header p {
                font-size: 1.2em;
                opacity: 0.9;
            }
            .content {
                padding: 40px;
            }
            .upload-section {
                text-align: center;
                margin-bottom: 40px;
            }
            .upload-box {
                border: 3px dashed #3498db;
                border-radius: 15px;
                padding: 50px;
                background: #f8f9fa;
                margin: 20px 0;
                transition: all 0.3s ease;
            }
            .upload-box:hover {
                border-color: #2980b9;
                background: #e8f4fc;
            }
            .btn {
                background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                margin: 10px;
                transition: all 0.3s ease;
                text-decoration: none;
                display: inline-block;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
            }
            .btn-demo { background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%); }
            .btn-download { background: linear-gradient(135deg, #27ae60 0%, #229954 100%); }
            .btn-sample { background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); }
            .results {
                background: #f8f9fa;
                border-radius: 15px;
                padding: 30px;
                margin-top: 30px;
                display: none;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }
            .stat-card {
                background: white;
                padding: 25px;
                border-radius: 12px;
                text-align: center;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                border-left: 4px solid #3498db;
            }
            .stat-number {
                font-size: 2.5em;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 5px;
            }
            .stat-label {
                color: #7f8c8d;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .plot-container {
                background: white;
                padding: 25px;
                border-radius: 12px;
                margin: 20px 0;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            .sample-table {
                background: white;
                padding: 25px;
                border-radius: 12px;
                margin: 20px 0;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                overflow-x: auto;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }
            th, td {
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ecf0f1;
            }
            th {
                background: #34495e;
                color: white;
                font-weight: 600;
            }
            tr:hover {
                background: #f8f9fa;
            }
            .error {
                background: #e74c3c;
                color: white;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
                display: none;
            }
            .success {
                background: #27ae60;
                color: white;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
                display: none;
            }
            .file-input {
                padding: 15px;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                font-size: 16px;
                margin: 15px 0;
                width: 100%;
                max-width: 400px;
            }
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 2s linear infinite;
                margin: 0 auto 15px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🐭 Mouse Behavior Analyzer</h1>
                <p>Advanced AI-powered analysis of mouse social behaviors</p>
            </div>
            
            <div class="content">
                <div class="upload-section">
                    <h2>📊 Upload Your Data</h2>
                    <p>Analyze mouse behavior patterns from tracking data</p>
                    
                    <div class="upload-box">
                        <h3>📁 Upload CSV File</h3>
                        <p>File should contain: <strong>video_id</strong> and <strong>frame</strong> columns</p>
                        
                        <form id="uploadForm" enctype="multipart/form-data">
                            <input type="file" name="file" accept=".csv" class="file-input" required>
                            <br>
                            <button type="submit" class="btn">
                                <span id="submitText">🚀 Analyze Behaviors</span>
                            </button>
                        </form>
                        
                        <div style="margin-top: 30px;">
                            <p><strong>Quick Start Options:</strong></p>
                            <button onclick="useDemoData()" class="btn btn-demo">🎮 Use Demo Data</button>
                            <button onclick="downloadSample()" class="btn btn-sample">📥 Download Sample CSV</button>
                        </div>
                    </div>
                </div>

                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Analyzing mouse behaviors...</p>
                </div>

                <div class="error" id="error"></div>
                <div class="success" id="success">Analysis completed successfully!</div>

                <div class="results" id="results">
                    <h2>📈 Analysis Results</h2>
                    
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-number" id="totalPred">0</div>
                            <div class="stat-label">Total Predictions</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number" id="uniqueVid">0</div>
                            <div class="stat-label">Unique Videos</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number" id="behaviors">0</div>
                            <div class="stat-label">Behaviors Detected</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number" id="commonBehavior">-</div>
                            <div class="stat-label">Most Common Behavior</div>
                        </div>
                    </div>

                    <div class="plot-container">
                        <h3>📊 Behavior Analysis</h3>
                        <div id="plotContainer"></div>
                    </div>

                    <div class="sample-table">
                        <h3>🔍 Sample Predictions</h3>
                        <div id="sampleTable"></div>
                    </div>

                    <div style="text-align: center; margin-top: 30px;">
                        <button onclick="downloadResults()" class="btn btn-download">💾 Download Full Results (CSV)</button>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let currentResults = null;

            async function useDemoData() {
                showLoading();
                try {
                    const response = await fetch('/demo');
                    const result = await response.json();
                    handleAnalysisResult(result);
                } catch (error) {
                    showError('Network error: ' + error.message);
                }
                hideLoading();
            }

            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                showLoading();
                
                const formData = new FormData(this);
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    handleAnalysisResult(result);
                } catch (error) {
                    showError('Network error: ' + error.message);
                }
                hideLoading();
            });

            function handleAnalysisResult(result) {
                if (result.success) {
                    currentResults = result;
                    
                    // Update statistics
                    document.getElementById('totalPred').textContent = result.total_predictions.toLocaleString();
                    document.getElementById('uniqueVid').textContent = result.unique_videos;
                    document.getElementById('behaviors').textContent = result.behaviors_detected;
                    document.getElementById('commonBehavior').textContent = result.most_common_behavior;
                    
                    // Show visualization
                    document.getElementById('plotContainer').innerHTML = 
                        `<img src="${result.plot_url}" style="max-width: 100%; border-radius: 8px;">`;
                    
                    // Show sample data
                    let tableHtml = '<table><tr><th>Video ID</th><th>Frame</th><th>Behavior</th><th>Behavior Name</th></tr>';
                    result.sample_predictions.forEach(pred => {
                        tableHtml += `<tr>
                            <td>${pred.video_id}</td>
                            <td>${pred.frame}</td>
                            <td>${pred.behavior}</td>
                            <td>${pred.behavior_name}</td>
                        </tr>`;
                    });
                    tableHtml += '</table>';
                    document.getElementById('sampleTable').innerHTML = tableHtml;
                    
                    // Show results
                    document.getElementById('results').style.display = 'block';
                    document.getElementById('success').style.display = 'block';
                    document.getElementById('error').style.display = 'none';
                    
                } else {
                    showError(result.error);
                }
            }

            function downloadSample() {
                window.open('/sample', '_blank');
            }

            function downloadResults() {
                if (currentResults && currentResults.download_url) {
                    window.open(currentResults.download_url, '_blank');
                }
            }

            function showLoading() {
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                document.getElementById('error').style.display = 'none';
                document.getElementById('success').style.display = 'none';
            }

            function hideLoading() {
                document.getElementById('loading').style.display = 'none';
            }

            function showError(message) {
                document.getElementById('error').textContent = message;
                document.getElementById('error').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                document.getElementById('success').style.display = 'none';
            }
        </script>
    </body>
    </html>
    '''

@app.route('/analyze', methods=['POST'])
def analyze_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Read and validate file
        df = pd.read_csv(file)
        required_cols = ['video_id', 'frame']
        
        if not all(col in df.columns for col in required_cols):
            return jsonify({'success': False, 'error': f'Missing required columns: {required_cols}'})
        
        return process_analysis(df)
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Analysis failed: {str(e)}'})

@app.route('/demo')
def analyze_demo():
    try:
        # Create demo data
        demo_data = []
        for i in range(3):
            for j in range(150):
                demo_data.append({
                    'video_id': f'video_{i+1:03d}',
                    'frame': j
                })
        
        df = pd.DataFrame(demo_data)
        return process_analysis(df)
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Demo analysis failed: {str(e)}'})

@app.route('/sample')
def download_sample():
    """Download sample CSV template"""
    sample_data = []
    for i in range(2):
        for j in range(50):
            sample_data.append({
                'video_id': f'video_{i+1:03d}',
                'frame': j
            })
    
    df = pd.DataFrame(sample_data)
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name='mouse_behavior_template.csv',
        mimetype='text/csv'
    )

@app.route('/download/<filename>')
def download_results(filename):
    try:
        return send_file(f'results/{filename}', as_attachment=True)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def process_analysis(df):
    """Process data and generate analysis results"""
    # Analyze behavior
    results_df = analyzer.analyze(df)
    
    # Generate visualization
    plot_url = generate_visualization(results_df)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'analysis_{timestamp}.csv'
    filepath = f'results/{filename}'
    results_df.to_csv(filepath, index=False)
    
    # Prepare sample predictions
    sample_data = []
    for _, row in results_df.head(8).iterrows():
        sample_data.append({
            'video_id': str(row['video_id']),
            'frame': int(row['frame']),
            'behavior': int(row['behavior']),
            'behavior_name': str(row['behavior_name'])
        })
    
    # Find most common behavior
    most_common = results_df['behavior'].mode()
    most_common_behavior = int(most_common.iloc[0]) if not most_common.empty else 0
    
    return jsonify({
        'success': True,
        'total_predictions': int(len(results_df)),
        'unique_videos': int(results_df['video_id'].nunique()),
        'behaviors_detected': int(results_df['behavior'].nunique()),
        'most_common_behavior': most_common_behavior,
        'plot_url': plot_url,
        'download_url': f'/download/{filename}',
        'sample_predictions': sample_data
    })

def generate_visualization(df):
    """Generate analysis visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Behavior distribution
    behavior_counts = df['behavior'].value_counts().head(10)
    bars = ax1.bar(behavior_counts.index, behavior_counts.values, 
                   color=plt.cm.Set3(np.linspace(0, 1, len(behavior_counts))))
    ax1.set_title('Top 10 Detected Behaviors', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Behavior Code')
    ax1.set_ylabel('Frequency')
    
    # 2. Video analysis
    video_counts = df['video_id'].value_counts()
    ax2.pie(video_counts.values, labels=video_counts.index, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Video Distribution', fontsize=14, fontweight='bold')
    
    # 3. Behavior timeline
    sample_video = df['video_id'].iloc[0]
    video_data = df[df['video_id'] == sample_video].head(50)
    ax3.plot(video_data['frame'], video_data['behavior'], 'o-', linewidth=2, markersize=4)
    ax3.set_title(f'Behavior Timeline - {sample_video}', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Behavior Code')
    ax3.grid(True, alpha=0.3)
    
    # 4. Behavior duration
    durations = []
    current_duration = 1
    for i in range(1, min(100, len(df))):
        if df['behavior'].iloc[i] == df['behavior'].iloc[i-1]:
            current_duration += 1
        else:
            durations.append(current_duration)
            current_duration = 1
    
    ax4.hist(durations, bins=15, color='lightgreen', alpha=0.7, edgecolor='black')
    ax4.set_title('Behavior Duration Distribution', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Duration (frames)')
    ax4.set_ylabel('Frequency')
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return f"data:image/png;base64,{plot_data}"

if __name__ == '__main__':
    print("🚀 Starting Mouse Behavior Analyzer...")
    print("✅ Numpy 2.0 Compatible")
    print("📊 Access: http://localhost:5000")
    print("🎮 Features: File upload, Demo data, Visualizations")
    app.run(host='0.0.0.0', port=5000, debug=False)
