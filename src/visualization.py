import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from src.data_processor import DataProcessor
from src.forecast_engine import ForecastEngine

def create_msu_usage_chart(hourly_msu):
    """
    Create a chart showing MSU usage over time.
    
    Args:
        hourly_msu (pandas.DataFrame): Hourly MSU usage data.
        
    Returns:
        plotly.graph_objects.Figure: MSU usage chart.
    """
    # Convert Date and Hour to datetime for proper time series plotting
    hourly_msu['Timestamp'] = pd.to_datetime(hourly_msu['Date'] + ' ' + hourly_msu['Hour'].astype(str) + ':00:00')
    
    # Create figure
    fig = px.line(
        hourly_msu, 
        x='Timestamp', 
        y='MSU Usage',
        title='MSU Usage Over Time',
        labels={'MSU Usage': 'MSU Consumption', 'Timestamp': 'Time'}
    )
    
    # Add threshold line for peak detection (90th percentile)
    threshold = hourly_msu['MSU Usage'].quantile(0.9)
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="Peak Threshold (90th percentile)",
        annotation_position="top right"
    )
    
    # Improve layout
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='MSU Usage',
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_top_jobs_chart(top_jobs):
    """
    Create a chart showing top MSU-consuming jobs.
    
    Args:
        top_jobs (pandas.DataFrame): Top jobs by MSU usage.
        
    Returns:
        plotly.graph_objects.Figure: Top jobs chart.
    """
    # Create figure
    fig = px.bar(
        top_jobs, 
        x='Job Name', 
        y='MSU Used',
        title='Top MSU-Consuming Jobs',
        color='MSU Used',
        color_continuous_scale='Viridis'
    )
    
    # Improve layout
    fig.update_layout(
        xaxis_title='Job Name',
        yaxis_title='Total MSU Usage',
        coloraxis_showscale=False,
        height=500
    )
    
    return fig

def create_job_class_chart(job_class_data):
    """
    Create a chart showing job distribution by class.
    
    Args:
        job_class_data (pandas.DataFrame): Job distribution by class.
        
    Returns:
        plotly.graph_objects.Figure: Job class distribution chart.
    """
    # Create figure
    fig = px.pie(
        job_class_data, 
        names='Job Class', 
        values='sum',
        title='MSU Usage by Job Class',
        hover_data=['count']
    )
    
    # Add custom hovertemplate
    fig.update_traces(
        hovertemplate='<b>Job Class:</b> %{label}<br><b>MSU Usage:</b> %{value:.2f}<br><b>Job Count:</b> %{customdata[0]}'
    )
    
    # Improve layout
    fig.update_layout(height=500)
    
    return fig

def create_heatmap(hourly_msu):
    """
    Create a heatmap showing MSU usage by hour and date.
    
    Args:
        hourly_msu (pandas.DataFrame): Hourly MSU usage data.
        
    Returns:
        plotly.graph_objects.Figure: MSU usage heatmap.
    """
    # Pivot data for heatmap
    pivot_data = hourly_msu.pivot(index='Hour', columns='Date', values='MSU Usage')
    
    # Create figure
    fig = px.imshow(
        pivot_data,
        labels=dict(x="Date", y="Hour of Day", color="MSU Usage"),
        x=pivot_data.columns,
        y=pivot_data.index,
        title="MSU Usage Heatmap by Hour and Date",
        color_continuous_scale='Viridis'
    )
    
    # Improve layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Hour of Day',
        height=600
    )
    
    return fig

def create_forecast_chart(forecast_data):
    """
    Create a chart showing MSU usage forecast.
    
    Args:
        forecast_data (pandas.DataFrame): Forecast data.
        
    Returns:
        plotly.graph_objects.Figure: Forecast chart.
    """
    # Create figure
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(
        go.Scatter(
            x=forecast_data['Timestamp'],
            y=forecast_data['Actual MSU'],
            mode='markers',
            name='Actual MSU',
            marker=dict(color='blue', size=8)
        )
    )
    
    # Add predicted values
    fig.add_trace(
        go.Scatter(
            x=forecast_data['Timestamp'],
            y=forecast_data['Predicted MSU'],
            mode='lines',
            name='Predicted MSU',
            line=dict(color='green', width=2)
        )
    )
    
    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=forecast_data['Timestamp'].tolist() + forecast_data['Timestamp'].tolist()[::-1],
            y=forecast_data['Upper Bound'].tolist() + forecast_data['Lower Bound'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0,176,246,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        )
    )
    
    # Improve layout
    fig.update_layout(
        title='MSU Usage Forecast',
        xaxis_title='Time',
        yaxis_title='MSU Usage',
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_optimization_impact_chart(before_data, after_data):
    """
    Create a chart showing the impact of optimization.
    
    Args:
        before_data (pandas.DataFrame): MSU usage before optimization.
        after_data (pandas.DataFrame): MSU usage after optimization.
        
    Returns:
        plotly.graph_objects.Figure: Optimization impact chart.
    """
    # Create figure
    fig = go.Figure()
    
    # Add before optimization
    fig.add_trace(
        go.Scatter(
            x=before_data['Timestamp'],
            y=before_data['MSU Usage'],
            mode='lines',
            name='Before Optimization',
            line=dict(color='red', width=2)
        )
    )
    
    # Add after optimization
    fig.add_trace(
        go.Scatter(
            x=after_data['Timestamp'],
            y=after_data['MSU Usage'],
            mode='lines',
            name='After Optimization',
            line=dict(color='green', width=2)
        )
    )
    
    # Improve layout
    fig.update_layout(
        title='MSU Usage Before and After Optimization',
        xaxis_title='Time',
        yaxis_title='MSU Usage',
        hovermode='x unified',
        height=500
    )
    
    return fig

def simulate_optimized_schedule(data_processor, optimization_suggestions):
    """
    Simulate MSU usage after applying optimization suggestions.
    
    Args:
        data_processor (DataProcessor): Instance of DataProcessor.
        optimization_suggestions (dict): Dictionary of job names and their rescheduling suggestions.
        
    Returns:
        tuple: (before_data, after_data) for visualization.
    """
    # Get original hourly MSU usage
    before_data = data_processor.get_hourly_msu_usage()
    before_data['Timestamp'] = pd.to_datetime(before_data['Date'] + ' ' + before_data['Hour'].astype(str) + ':00:00')
    
    # Create a copy of the original data for simulation
    df_optimized = data_processor.df.copy()
    
    # Apply optimization suggestions
    for job_name, suggestions in optimization_suggestions.items():
        if suggestions:
            # Get job details
            job_details = data_processor.get_job_details(job_name)
            
            if not job_details.empty:
                # Get best suggestion
                best_suggestion = max(suggestions, key=lambda x: x['Estimated Savings'])
                
                # Parse new start time
                new_date = best_suggestion['Date']
                new_hour = int(best_suggestion['Start Time'].split(':')[0])
                
                # Update job start times in the optimized dataframe
                for idx in job_details.index:
                    # Calculate new start time
                    original_start = df_optimized.loc[idx, 'Start Time']
                    original_minutes = original_start.minute
                    
                    new_start_time = pd.to_datetime(f"{new_date} {new_hour:02d}:{original_minutes:02d}:00")
                    
                    # Update start time
                    df_optimized.loc[idx, 'Start Time'] = new_start_time
                    
                    # Update end time
                    duration = df_optimized.loc[idx, 'Duration (min)']
                    df_optimized.loc[idx, 'End Time'] = new_start_time + pd.to_timedelta(duration, unit='m')
    
    # Create a new DataProcessor with the optimized data
    optimized_processor = DataProcessor(df=df_optimized)
    
    # Get optimized hourly MSU usage
    after_data = optimized_processor.get_hourly_msu_usage()
    after_data['Timestamp'] = pd.to_datetime(after_data['Date'] + ' ' + after_data['Hour'].astype(str) + ':00:00')
    
    return before_data, after_data
