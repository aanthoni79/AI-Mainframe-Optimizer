import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def format_msu(msu_value):
    """
    Format MSU values for display.
    
    Args:
        msu_value (float): MSU value to format.
        
    Returns:
        str: Formatted MSU value.
    """
    return f"{msu_value:,.2f}"

def format_percentage(value):
    """
    Format percentage values for display.
    
    Args:
        value (float): Percentage value to format.
        
    Returns:
        str: Formatted percentage value.
    """
    return f"{value:.2f}%"

def format_datetime(dt):
    """
    Format datetime for display.
    
    Args:
        dt (datetime): Datetime to format.
        
    Returns:
        str: Formatted datetime.
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def generate_summary(data_processor):
    """
    Generate a natural language summary of the data.
    
    Args:
        data_processor (DataProcessor): Instance of DataProcessor.
        
    Returns:
        str: Natural language summary.
    """
    # Get key metrics
    total_msu = data_processor.df['MSU Used'].sum()
    total_jobs = len(data_processor.df)
    unique_jobs = data_processor.df['Job Name'].nunique()
    avg_msu_per_job = data_processor.df['MSU Used'].mean()
    
    # Get top jobs
    top_jobs = data_processor.get_top_msu_jobs(n=3)
    top_job_names = top_jobs['Job Name'].tolist()
    top_job_msu = top_jobs['MSU Used'].tolist()
    
    # Get peak windows
    peak_windows = data_processor.detect_peak_windows()
    peak_count = len(peak_windows)
    
    # Generate summary
    summary = f"""
    ## Performance Summary
    
    The analysis of {total_jobs} job executions shows a total MSU consumption of {format_msu(total_msu)} MSU.
    
    ### Key Findings:
    
    - {unique_jobs} unique jobs were executed with an average MSU usage of {format_msu(avg_msu_per_job)} per job.
    - The top MSU-consuming job is {top_job_names[0]} with {format_msu(top_job_msu[0])} MSU.
    - {peak_count} peak usage windows were identified where MSU consumption exceeded the 90th percentile threshold.
    
    ### Top MSU Consumers:
    
    1. {top_job_names[0]}: {format_msu(top_job_msu[0])} MSU
    2. {top_job_names[1]}: {format_msu(top_job_msu[1])} MSU
    3. {top_job_names[2]}: {format_msu(top_job_msu[2])} MSU
    
    These top 3 jobs account for {format_percentage(sum(top_job_msu) / total_msu * 100)} of total MSU consumption.
    """
    
    return summary

def generate_optimization_summary(data_processor, optimization_suggestions, total_savings):
    """
    Generate a natural language summary of optimization recommendations.
    
    Args:
        data_processor (DataProcessor): Instance of DataProcessor.
        optimization_suggestions (dict): Dictionary of job names and their rescheduling suggestions.
        total_savings (float): Total estimated MSU savings percentage.
        
    Returns:
        str: Natural language summary of optimization recommendations.
    """
    # Generate summary
    summary = f"""
    ## Optimization Recommendations
    
    Based on the analysis of job execution patterns and MSU usage, we recommend the following optimizations:
    
    ### Overall Impact:
    
    - Estimated total MSU savings: {format_percentage(total_savings)}
    - Number of jobs recommended for rescheduling: {len(optimization_suggestions)}
    
    ### Specific Recommendations:
    
    """
    
    # Add job-specific recommendations
    for job_name, suggestions in optimization_suggestions.items():
        if suggestions:
            # Get job details
            job_details = data_processor.get_job_details(job_name)
            total_job_msu = job_details['MSU Used'].sum()
            
            # Get best suggestion
            best_suggestion = max(suggestions, key=lambda x: x['Estimated Savings'])
            
            summary += f"""
    #### {job_name}:
    - Current MSU usage: {format_msu(total_job_msu)}
    - Recommended action: Reschedule to {best_suggestion['Date']} at {best_suggestion['Start Time']}
    - Estimated savings: {format_percentage(best_suggestion['Estimated Savings'])}
    - Current environment MSU: {format_msu(best_suggestion['Current MSU Usage'])}
            """
    
    return summary
