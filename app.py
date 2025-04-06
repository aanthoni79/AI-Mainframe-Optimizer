import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

from src.data_processor import DataProcessor
from src.forecast_engine import ForecastEngine
from src.visualization import (
    create_msu_usage_chart,
    create_top_jobs_chart,
    create_job_class_chart,
    create_heatmap,
    create_forecast_chart,
    create_optimization_impact_chart,
    simulate_optimized_schedule
)

# Set page configuration
st.set_page_config(
    page_title="AI Mainframe Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .savings-card {
        background-color: #e8f5e9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = None
if 'forecast_engine' not in st.session_state:
    st.session_state.forecast_engine = None
if 'optimization_suggestions' not in st.session_state:
    st.session_state.optimization_suggestions = {}
if 'total_savings' not in st.session_state:
    st.session_state.total_savings = 0

# Main header
st.markdown('<h1 class="main-header">AI Mainframe Optimizer</h1>', unsafe_allow_html=True)
st.markdown('Leverage AI to analyze mainframe job data, detect expensive jobs, and generate optimization suggestions.')

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/server.png", width=80)
    st.markdown("## Upload Data")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload SMF/CSV job data", type=["csv"])
    
    # Use sample data option
    use_sample = st.checkbox("Use sample data", value=True if not uploaded_file else False)
    
    # Load data button
    load_data = st.button("Load Data")
    
    if load_data:
        with st.spinner("Processing data..."):
            if uploaded_file is not None:
                # Save uploaded file temporarily
                temp_file_path = os.path.join("data", "temp_upload.csv")
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Initialize data processor with uploaded file
                st.session_state.data_processor = DataProcessor(temp_file_path)
                st.success("Data loaded successfully!")
            
            elif use_sample:
                # Use sample data
                sample_path = os.path.join("data", "enhanced_mock_smf_data.csv")
                st.session_state.data_processor = DataProcessor(sample_path)
                st.success("Sample data loaded successfully!")
            
            else:
                st.error("Please upload a file or use sample data.")
    
    # Sidebar info
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This application analyzes mainframe job data to identify optimization opportunities and provide recommendations for reducing MSU usage and costs.
    
    **Features:**
    - MSU usage analysis
    - Peak detection
    - Job optimization recommendations
    - MSU forecasting
    - Savings calculator
    """)

# Main content
if st.session_state.data_processor is not None:
    # Create tabs
    tabs = st.tabs([
        "📊 Dashboard", 
        "🔍 Job Analysis", 
        "⏱️ Peak Detection", 
        "🔮 Forecasting", 
        "💰 Optimization"
    ])
    
    # Dashboard tab
    with tabs[0]:
        st.markdown('<h2 class="sub-header">MSU Usage Dashboard</h2>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            total_msu = st.session_state.data_processor.df['MSU Used'].sum()
            st.metric("Total MSU Consumption", f"{total_msu:,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            avg_msu_per_job = st.session_state.data_processor.df['MSU Used'].mean()
            st.metric("Average MSU per Job", f"{avg_msu_per_job:,.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            total_jobs = len(st.session_state.data_processor.df)
            st.metric("Total Jobs", f"{total_jobs:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # MSU usage over time
        st.subheader("MSU Usage Over Time")
        hourly_msu = st.session_state.data_processor.get_hourly_msu_usage()
        msu_chart = create_msu_usage_chart(hourly_msu)
        st.plotly_chart(msu_chart, use_container_width=True)
        
        # Top jobs and job class distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top MSU-Consuming Jobs")
            top_jobs = st.session_state.data_processor.get_top_msu_jobs(n=10)
            top_jobs_chart = create_top_jobs_chart(top_jobs)
            st.plotly_chart(top_jobs_chart, use_container_width=True)
        
        with col2:
            st.subheader("MSU Usage by Job Class")
            job_class_data = st.session_state.data_processor.get_job_class_distribution()
            job_class_chart = create_job_class_chart(job_class_data)
            st.plotly_chart(job_class_chart, use_container_width=True)
        
        # MSU usage heatmap
        st.subheader("MSU Usage Heatmap")
        heatmap = create_heatmap(hourly_msu)
        st.plotly_chart(heatmap, use_container_width=True)
    
    # Job Analysis tab
    with tabs[1]:
        st.markdown('<h2 class="sub-header">Job Analysis</h2>', unsafe_allow_html=True)
        
        # Job selector
        job_names = sorted(st.session_state.data_processor.df['Job Name'].unique())
        selected_job = st.selectbox("Select a job to analyze:", job_names)
        
        if selected_job:
            # Get job details
            job_details = st.session_state.data_processor.get_job_details(selected_job)
            
            # Job summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                total_job_msu = job_details['MSU Used'].sum()
                st.metric("Total MSU Consumption", f"{total_job_msu:,.0f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                avg_duration = job_details['Duration (min)'].mean()
                st.metric("Average Duration (min)", f"{avg_duration:,.1f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                job_count = len(job_details)
                st.metric("Job Executions", f"{job_count:,}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Job execution timeline
            st.subheader("Job Execution Timeline")
            
            # Create timeline chart
            fig = px.scatter(
                job_details,
                x='Start Time',
                y='MSU Used',
                size='Duration (min)',
                color='Job Class',
                hover_data=['CPU Time (s)', 'IO Count', 'Storage Used (MB)'],
                title=f"Execution Timeline for {selected_job}"
            )
            
            fig.update_layout(
                xaxis_title='Start Time',
                yaxis_title='MSU Usage',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Job details table
            st.subheader("Job Execution Details")
            
            # Format the table
            display_df = job_details.copy()
            display_df['Start Time'] = display_df['Start Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df['End Time'] = display_df['End Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df = display_df.sort_values('Start Time', ascending=False)
            
            st.dataframe(
                display_df[[
                    'Start Time', 'Duration (min)', 'MSU Used', 'CPU Time (s)', 
                    'IO Count', 'Enqueue', 'Storage Used (MB)', 'Job Class'
                ]],
                use_container_width=True
            )
    
    # Peak Detection tab
    with tabs[2]:
        st.markdown('<h2 class="sub-header">Peak MSU Detection</h2>', unsafe_allow_html=True)
        
        # Threshold selector
        threshold_percentile = st.slider(
            "Peak Detection Threshold (Percentile):",
            min_value=80,
            max_value=99,
            value=90,
            step=1,
            help="Set the percentile threshold for peak detection. Higher values detect only the most extreme peaks."
        )
        
        # Detect peaks
        peak_windows = st.session_state.data_processor.detect_peak_windows(threshold_percentile)
        
        # Display peak windows
        st.subheader(f"Peak MSU Usage Windows (Above {threshold_percentile}th Percentile)")
        
        if not peak_windows.empty:
            # Create peak windows chart
            fig = px.bar(
                peak_windows,
                x='Time Window',
                y='MSU Usage',
                title=f"Peak MSU Usage Windows (Above {threshold_percentile}th Percentile)",
                color='MSU Usage',
                color_continuous_scale='Reds'
            )
            
            fig.update_layout(
                xaxis_title='Time Window',
                yaxis_title='MSU Usage',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display peak windows table
            st.dataframe(peak_windows, use_container_width=True)
            
            # Jobs running during peak windows
            st.subheader("Jobs Running During Peak Windows")
            
            # Identify jobs running during peak hours
            optimization_candidates = st.session_state.data_processor.identify_optimization_candidates(n=10)
            
            if not optimization_candidates.empty:
                # Format the table
                display_df = optimization_candidates.copy()
                display_df['Start Time'] = display_df['Start Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                display_df['End Time'] = display_df['End Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                st.dataframe(
                    display_df[[
                        'Job Name', 'Start Time', 'Duration (min)', 'MSU Used', 
                        'Job Class', 'CPU Time (s)', 'IO Count'
                    ]],
                    use_container_width=True
                )
            else:
                st.info("No jobs found running during peak windows.")
        else:
            st.info("No peak windows detected with the current threshold.")
    
    # Forecasting tab
    with tabs[3]:
        st.markdown('<h2 class="sub-header">MSU Usage Forecasting</h2>', unsafe_allow_html=True)
        
        # Initialize forecast engine if not already done
        if st.session_state.forecast_engine is None:
            with st.spinner("Initializing forecast engine..."):
                st.session_state.forecast_engine = ForecastEngine(st.session_state.data_processor)
        
        # Forecast parameters
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_periods = st.slider(
                "Forecast Periods (Hours):",
                min_value=12,
                max_value=72,
                value=24,
                step=12,
                help="Number of hours to forecast into the future."
            )
        
        with col2:
            generate_forecast = st.button("Generate Forecast")
        
        if generate_forecast:
            with st.spinner("Generating forecast..."):
                # Train model and generate forecast
                forecast = st.session_state.forecast_engine.train_model(periods=forecast_periods)
                
                # Get forecast data for visualization
                forecast_data = st.session_state.forecast_engine.get_forecast_data()
                
                if forecast_data is not None:
                    # Create forecast chart
                    st.subheader("MSU Usage Forecast")
                    forecast_chart = create_forecast_chart(forecast_data)
                    st.plotly_chart(forecast_chart, use_container_width=True)
                    
                    # Identify future peaks
                    future_peaks = st.session_state.forecast_engine.identify_future_peaks()
                    
                    if future_peaks is not None and not future_peaks.empty:
                        st.subheader("Predicted Peak MSU Usage Windows")
                        st.dataframe(future_peaks, use_container_width=True)
                    else:
                        st.info("No significant future peaks predicted.")
                else:
                    st.error("Failed to generate forecast. Please try again.")
    
    # Optimization tab
    with tabs[4]:
        st.markdown('<h2 class="sub-header">MSU Optimization Recommendations</h2>', unsafe_allow_html=True)
        
        # Identify optimization candidates
        if not st.session_state.optimization_suggestions:
            optimization_candidates = st.session_state.data_processor.identify_optimization_candidates(n=5)
            
            if not optimization_candidates.empty:
                # Generate optimization suggestions
                for _, job in optimization_candidates.iterrows():
                    job_name = job['Job Name']
                    suggestions = st.session_state.data_processor.suggest_job_rescheduling(job_name)
                    if suggestions:
                        st.session_state.optimization_suggestions[job_name] = suggestions
                
                # Calculate total savings
                st.session_state.total_savings = st.session_state.data_processor.calculate_total_savings(
                    st.session_state.optimization_suggestions
                )
        
        # Display optimization suggestions
        if st.session_state.optimization_suggestions:
            # Display total savings
            st.markdown('<div class="savings-card">', unsafe_allow_html=True)
            st.metric(
                "Estimated MSU Savings",
                f"{st.session_state.total_savings:.2f}%",
                delta=f"{st.session_state.total_savings:.2f}%"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Display job rescheduling recommendations
            st.subheader("Job Rescheduling Recommendations")
            
            for job_name, suggestions in st.session_state.optimization_suggestions.items():
                if suggestions:
                    st.markdown(f"### {job_name}")
                    
                    # Create a DataFrame for display
                    suggestions_df = pd.DataFrame(suggestions)
                    suggestions_df['Estimated Savings'] = suggestions_df['Estimated Savings'].apply(
                        lambda x: f"{x:.2f}%"
                    )
                    
                    st.dataframe(suggestions_df, use_container_width=True)
            
            # Simulate optimized schedule
            st.subheader("Optimization Impact Visualization")
            
            if st.button("Simulate Optimized Schedule"):
                with st.spinner("Simulating optimized schedule..."):
                    before_data, after_data = simulate_optimized_schedule(
                        st.session_state.data_processor,
                        st.session_state.optimization_suggestions
                    )
                    
                    # Create optimization impact chart
                    impact_chart = create_optimization_impact_chart(before_data, after_data)
                    st.plotly_chart(impact_chart, use_container_width=True)
                    
                    # Calculate peak reduction
                    before_peak = before_data['MSU Usage'].max()
                    after_peak = after_data['MSU Usage'].max()
                    peak_reduction = (before_peak - after_peak) / before_peak * 100
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Peak MSU Reduction",
                            f"{peak_reduction:.2f}%",
                            delta=f"{peak_reduction:.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "Total MSU Savings",
                            f"{st.session_state.total_savings:.2f}%",
                            delta=f"{st.session_state.total_savings:.2f}%"
                        )
        else:
            st.info("No optimization suggestions available. Please analyze the data first.")
else:
    # Display welcome message when no data is loaded
    st.info("👈 Please upload data or use the sample data to get started.")
    
    # Display sample screenshots
    st.markdown("### Sample Dashboard Preview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("https://img.icons8.com/color/452/dashboard-layout.png", width=300)
        st.caption("MSU Usage Dashboard")
    
    with col2:
        st.image("https://img.icons8.com/color/452/combo-chart--v1.png", width=300)
        st.caption("Optimization Recommendations")
