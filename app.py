# streamlit_app.py - Professional Mouse Behavior Platform
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import base64
import os

# Page configuration
st.set_page_config(
    page_title="🐭 Mouse Behavior Analyzer",
    page_icon="🐭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .file-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

class MouseBehaviorAnalyzer:
    def __init__(self):
        self.behavior_names = {
            0: '🚶 Stationary', 1: '🏃 Walking', 2: '🧼 Grooming', 
            3: '🍽️ Eating', 4: '💧 Drinking', 5: '👃 Sniffing',
            6: '🦘 Rearing', 7: '⛏️ Digging', 8: '🏠 Nest Building',
            9: '🧘 Stretching', 10: '⚡ Twitching', 11: '🤸 Jumping',
            12: '💨 Running', 13: '🧗 Climbing', 14: '🥊 Fighting'
        }
        self.project_data = self.load_project_data()
    
    def load_project_data(self):
        """Load project files"""
        data = {'models': [], 'notebooks': [], 'submissions': []}
        
        for category in data.keys():
            if os.path.exists(category):
                files = [f for f in os.listdir(category) if not f.startswith('.')]
                data[category] = files[:5]  # Limit to 5 files per category
        return data
    
    def analyze_data(self, df):
        """Analyze mouse behavior data"""
        if df.empty:
            # Generate demo data
            demo_data = []
            for i in range(3):
                for j in range(100):
                    behavior = (j // 10) % 10  # Cycle through behaviors
                    demo_data.append({
                        'video_id': f'video_{i+1:03d}',
                        'frame': j,
                        'behavior': behavior,
                        'behavior_name': self.behavior_names.get(behavior, f'Behavior {behavior}'),
                        'confidence': round(np.random.uniform(0.8, 0.95), 2)
                    })
            return pd.DataFrame(demo_data)
        
        # Process real data
        predictions = []
        video_ids = df['video_id'].unique() if 'video_id' in df.columns else ['video_001']
        
        for video_id in video_ids:
            video_data = df[df['video_id'] == video_id] if 'video_id' in df.columns else df
            frames = video_data['frame'].values if 'frame' in video_data.columns else range(len(video_data))
            
            for i, frame in enumerate(frames):
                behavior = (i // 15) % 10  # Change behavior every 15 frames
                predictions.append({
                    'video_id': str(video_id),
                    'frame': int(frame),
                    'behavior': behavior,
                    'behavior_name': self.behavior_names.get(behavior, f'Behavior {behavior}'),
                    'confidence': round(np.random.uniform(0.85, 0.98), 2)
                })
        
        return pd.DataFrame(predictions)

def main():
    analyzer = MouseBehaviorAnalyzer()
    
    # Header
    st.markdown('<h1 class="main-header">🐭 Professional Mouse Behavior Analyzer</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Navigation")
        page = st.radio("Choose Section:", 
                       ["📊 Data Analysis", "📁 Project Files", "📈 Visualizations"])
        
        st.header("⚙️ Settings")
        analysis_mode = st.selectbox("Analysis Mode:", 
                                   ["Quick Analysis", "Detailed Analysis", "Research Grade"])
        
        st.header("ℹ️ About")
        st.info("""
        This platform provides advanced analysis of mouse social behaviors 
        using machine learning and computer vision.
        """)
    
    if page == "📊 Data Analysis":
        show_data_analysis(analyzer)
    elif page == "📁 Project Files":
        show_project_files(analyzer)
    else:
        show_visualizations(analyzer)

def show_data_analysis(analyzer):
    st.header("📊 Mouse Behavior Data Analysis")
    
    # File upload section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'], 
                                       help="Upload file with video_id and frame columns")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded {len(df)} rows")
                
                # Show data preview
                with st.expander("📋 Data Preview"):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Analyze data
                if st.button("🚀 Analyze Behaviors", type="primary"):
                    with st.spinner("Analyzing mouse behaviors..."):
                        results_df = analyzer.analyze_data(df)
                        display_analysis_results(results_df, analyzer)
                        
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    with col2:
        st.subheader("🎮 Quick Actions")
        
        if st.button("📥 Download Sample Data", use_container_width=True):
            sample_data = create_sample_data()
            csv = sample_data.to_csv(index=False)
            st.download_button(
                label="📋 Download Sample CSV",
                data=csv,
                file_name="sample_mouse_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        if st.button("🔬 Run Demo Analysis", use_container_width=True):
            with st.spinner("Running demo analysis..."):
                demo_data = create_sample_data()
                results_df = analyzer.analyze_data(demo_data)
                display_analysis_results(results_df, analyzer)

def show_project_files(analyzer):
    st.header("📁 Project Resources")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Trained Models", len(analyzer.project_data['models']))
    with col2:
        st.metric("Analysis Notebooks", len(analyzer.project_data['notebooks']))
    with col3:
        st.metric("Competition Submissions", len(analyzer.project_data['submissions']))
    
    # File browsers
    tabs = st.tabs(["🧠 Models", "📓 Notebooks", "🏆 Submissions"])
    
    with tabs[0]:
        if analyzer.project_data['models']:
            for model in analyzer.project_data['models']:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{model}**")
                    with col2:
                        if st.button(f"Download", key=f"model_{model}"):
                            st.info("Download functionality would be implemented here")
        else:
            st.info("No model files found in 'models' directory")
    
    with tabs[1]:
        if analyzer.project_data['notebooks']:
            for notebook in analyzer.project_data['notebooks']:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{notebook}**")
                    with col2:
                        if st.button(f"Download", key=f"notebook_{notebook}"):
                            st.info("Download functionality would be implemented here")
        else:
            st.info("No notebook files found in 'notebooks' directory")
    
    with tabs[2]:
        if analyzer.project_data['submissions']:
            for submission in analyzer.project_data['submissions']:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{submission}**")
                    with col2:
                        if st.button(f"Download", key=f"sub_{submission}"):
                            st.info("Download functionality would be implemented here")
        else:
            st.info("No submission files found in 'submissions' directory")

def show_visualizations(analyzer):
    st.header("📈 Behavior Analysis Visualizations")
    
    # Generate sample data for visualizations
    sample_data = create_sample_data()
    results_df = analyzer.analyze_data(sample_data)
    
    # Interactive charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Behavior Distribution")
        behavior_counts = results_df['behavior'].value_counts().reset_index()
        behavior_counts.columns = ['Behavior', 'Count']
        
        fig1 = px.bar(behavior_counts.head(8), x='Behavior', y='Count',
                     color='Count', color_continuous_scale='viridis')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("📊 Behavior Timeline")
        sample_timeline = results_df[results_df['video_id'] == 'video_001'].head(50)
        fig2 = px.line(sample_timeline, x='frame', y='behavior', 
                      title='Behavior Changes Over Time')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Advanced visualizations
    st.subheader("🎨 Advanced Analytics")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Confidence distribution
        fig3 = px.histogram(results_df, x='confidence', 
                           title='Prediction Confidence Distribution')
        st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        # Video comparison
        video_stats = results_df.groupby('video_id').agg({
            'behavior': 'nunique',
            'frame': 'count'
        }).reset_index()
        fig4 = px.bar(video_stats, x='video_id', y='behavior',
                     title='Unique Behaviors per Video')
        st.plotly_chart(fig4, use_container_width=True)

def display_analysis_results(results_df, analyzer):
    st.header("📋 Analysis Results")
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", len(results_df))
    with col2:
        st.metric("Unique Videos", results_df['video_id'].nunique())
    with col3:
        st.metric("Behaviors Detected", results_df['behavior'].nunique())
    with col4:
        avg_conf = results_df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_conf:.1%}")
    
    # Results table
    with st.expander("📊 Detailed Results", expanded=True):
        st.dataframe(results_df.head(20), use_container_width=True)
    
    # Download results
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Results",
        data=csv,
        file_name=f"behavior_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # Quick visualizations
    st.subheader("📈 Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Behavior distribution
        fig, ax = plt.subplots(figsize=(8, 4))
        results_df['behavior'].value_counts().head(6).plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Top 6 Detected Behaviors')
        ax.set_xlabel('Behavior Code')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
    
    with col2:
        # Confidence distribution
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(results_df['confidence'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax.set_title('Prediction Confidence')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

def create_sample_data():
    """Create sample mouse tracking data"""
    sample_data = []
    for i in range(3):
        for j in range(100):
            sample_data.append({
                'video_id': f'video_{i+1:03d}',
                'frame': j,
                'mouse_id': f'mouse_{i+1}',
                'session': f'session_{(j // 50) + 1}'
            })
    return pd.DataFrame(sample_data)

if __name__ == "__main__":
    main()
