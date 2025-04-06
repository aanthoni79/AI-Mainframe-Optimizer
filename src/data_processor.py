import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataProcessor:
    """
    Class for processing and analyzing mainframe job data.
    """
    
    def __init__(self, data_path=None, df=None):
        """
        Initialize the DataProcessor with either a path to a CSV file or a pandas DataFrame.
        
        Args:
            data_path (str, optional): Path to the CSV file containing job data.
            df (pandas.DataFrame, optional): DataFrame containing job data.
        """
        if df is not None:
            self.df = df
        elif data_path is not None:
            self.df = self.load_data(data_path)
        else:
            self.df = None
    
    def load_data(self, data_path):
        """
        Load data from a CSV file.
        
        Args:
            data_path (str): Path to the CSV file.
            
        Returns:
            pandas.DataFrame: Loaded and preprocessed data.
        """
        df = pd.read_csv(data_path)
        return self.preprocess_data(df)
    
    def preprocess_data(self, df):
        """
        Preprocess the data for analysis.
        
        Args:
            df (pandas.DataFrame): Raw data.
            
        Returns:
            pandas.DataFrame: Preprocessed data.
        """
        # Convert Start Time to datetime
        df['Start Time'] = pd.to_datetime(df['Start Time'])
        
        # Calculate End Time
        df['End Time'] = df['Start Time'] + pd.to_timedelta(df['Duration (min)'], unit='m')
        
        # Extract date and hour for aggregation
        df['Date'] = df['Start Time'].dt.date
        df['Hour'] = df['Start Time'].dt.hour
        
        # Calculate MSU per minute
        df['MSU per Minute'] = df['MSU Used'] / df['Duration (min)']
        
        return df
    
    def get_top_msu_jobs(self, n=10):
        """
        Get the top N jobs by MSU usage.
        
        Args:
            n (int, optional): Number of top jobs to return. Defaults to 10.
            
        Returns:
            pandas.DataFrame: Top N jobs by MSU usage.
        """
        job_msu = self.df.groupby('Job Name')['MSU Used'].sum().reset_index()
        return job_msu.sort_values('MSU Used', ascending=False).head(n)
    
    def get_hourly_msu_usage(self):
        """
        Calculate MSU usage by hour.
        
        Returns:
            pandas.DataFrame: MSU usage aggregated by hour.
        """
        # Create a time series with all hours
        all_dates = self.df['Date'].unique()
        all_hours = range(24)
        
        # Create a DataFrame with all date-hour combinations
        date_hour_combinations = [(date, hour) for date in all_dates for hour in all_hours]
        hourly_df = pd.DataFrame(date_hour_combinations, columns=['Date', 'Hour'])
        
        # Function to calculate MSU usage for a specific hour
        def calculate_msu_for_hour(date, hour):
            hour_start = datetime.combine(date, datetime.min.time()) + timedelta(hours=hour)
            hour_end = hour_start + timedelta(hours=1)
            
            # Filter jobs that were running during this hour
            jobs_in_hour = self.df[
                ((self.df['Start Time'] < hour_end) & (self.df['End Time'] > hour_start))
            ]
            
            if jobs_in_hour.empty:
                return 0
            
            # Calculate the MSU contribution for each job during this hour
            total_msu = 0
            for _, job in jobs_in_hour.iterrows():
                job_start = max(job['Start Time'], hour_start)
                job_end = min(job['End Time'], hour_end)
                
                # Calculate the fraction of the hour that the job was running
                fraction_of_hour = (job_end - job_start).total_seconds() / 3600
                
                # Calculate MSU contribution (assuming linear distribution of MSU over job duration)
                msu_contribution = job['MSU per Minute'] * fraction_of_hour * 60
                total_msu += msu_contribution
            
            return total_msu
        
        # Calculate MSU for each hour
        hourly_df['MSU Usage'] = hourly_df.apply(
            lambda row: calculate_msu_for_hour(row['Date'], row['Hour']), 
            axis=1
        )
        
        # Convert Date to string for easier grouping
        hourly_df['Date'] = hourly_df['Date'].astype(str)
        
        return hourly_df
    
    def detect_peak_windows(self, threshold_percentile=90):
        """
        Detect time windows with peak MSU usage.
        
        Args:
            threshold_percentile (int, optional): Percentile threshold for peak detection. Defaults to 90.
            
        Returns:
            pandas.DataFrame: Time windows with peak MSU usage.
        """
        hourly_msu = self.get_hourly_msu_usage()
        
        # Calculate threshold based on percentile
        threshold = hourly_msu['MSU Usage'].quantile(threshold_percentile/100)
        
        # Identify peak hours
        peak_hours = hourly_msu[hourly_msu['MSU Usage'] >= threshold].copy()
        
        # Sort by date and hour
        peak_hours = peak_hours.sort_values(['Date', 'Hour'])
        
        # Format for display
        peak_hours['Time Window'] = peak_hours.apply(
            lambda row: f"{row['Date']} {row['Hour']:02d}:00-{row['Hour']+1:02d}:00", 
            axis=1
        )
        
        return peak_hours[['Time Window', 'MSU Usage']]
    
    def get_job_class_distribution(self):
        """
        Get the distribution of jobs by job class.
        
        Returns:
            pandas.DataFrame: Job distribution by class.
        """
        return self.df.groupby('Job Class')['MSU Used'].agg(['sum', 'count']).reset_index()
    
    def get_job_details(self, job_name):
        """
        Get detailed information for a specific job.
        
        Args:
            job_name (str): Name of the job.
            
        Returns:
            pandas.DataFrame: Detailed information for the specified job.
        """
        return self.df[self.df['Job Name'] == job_name].copy()
    
    def identify_optimization_candidates(self, n=5):
        """
        Identify jobs that are good candidates for optimization.
        
        Args:
            n (int, optional): Number of candidates to return. Defaults to 5.
            
        Returns:
            pandas.DataFrame: Jobs that are good candidates for optimization.
        """
        # Get hourly MSU usage
        hourly_msu = self.get_hourly_msu_usage()
        
        # Identify peak hours (top 10% of MSU usage)
        threshold = hourly_msu['MSU Usage'].quantile(0.9)
        peak_hours = hourly_msu[hourly_msu['MSU Usage'] >= threshold]
        
        # Convert peak hours to datetime for comparison
        peak_datetime_ranges = []
        for _, row in peak_hours.iterrows():
            date_obj = datetime.strptime(row['Date'], '%Y-%m-%d').date()
            hour_start = datetime.combine(date_obj, datetime.min.time()) + timedelta(hours=row['Hour'])
            hour_end = hour_start + timedelta(hours=1)
            peak_datetime_ranges.append((hour_start, hour_end))
        
        # Function to check if a job runs during peak hours
        def runs_during_peak(job):
            for peak_start, peak_end in peak_datetime_ranges:
                if (job['Start Time'] < peak_end) and (job['End Time'] > peak_start):
                    return True
            return False
        
        # Identify jobs running during peak hours
        self.df['Runs During Peak'] = self.df.apply(runs_during_peak, axis=1)
        peak_jobs = self.df[self.df['Runs During Peak']].copy()
        
        # Sort by MSU usage to find the most expensive jobs during peak hours
        peak_jobs = peak_jobs.sort_values('MSU Used', ascending=False)
        
        # Return top N candidates
        return peak_jobs.head(n)
    
    def suggest_job_rescheduling(self, job_name):
        """
        Suggest alternative times for running a specific job.
        
        Args:
            job_name (str): Name of the job to reschedule.
            
        Returns:
            list: List of suggested time windows for rescheduling.
        """
        # Get hourly MSU usage
        hourly_msu = self.get_hourly_msu_usage()
        
        # Get job details
        job_details = self.get_job_details(job_name)
        
        if job_details.empty:
            return []
        
        # Get average duration of the job
        avg_duration = job_details['Duration (min)'].mean()
        
        # Find low-usage hours (bottom 30% of MSU usage)
        threshold = hourly_msu['MSU Usage'].quantile(0.3)
        low_usage_hours = hourly_msu[hourly_msu['MSU Usage'] <= threshold]
        
        # Sort by MSU usage (ascending)
        low_usage_hours = low_usage_hours.sort_values('MSU Usage')
        
        # Format suggestions
        suggestions = []
        for _, row in low_usage_hours.head(3).iterrows():
            date_obj = datetime.strptime(row['Date'], '%Y-%m-%d').date()
            hour_start = datetime.combine(date_obj, datetime.min.time()) + timedelta(hours=row['Hour'])
            suggestion = {
                'Date': row['Date'],
                'Start Time': f"{row['Hour']:02d}:00",
                'Current MSU Usage': row['MSU Usage'],
                'Estimated Savings': self._estimate_savings(job_name, hour_start)
            }
            suggestions.append(suggestion)
        
        return suggestions
    
    def _estimate_savings(self, job_name, new_start_time):
        """
        Estimate MSU savings from rescheduling a job.
        
        Args:
            job_name (str): Name of the job to reschedule.
            new_start_time (datetime): Proposed new start time.
            
        Returns:
            float: Estimated MSU savings.
        """
        # Get job details
        job_details = self.get_job_details(job_name)
        
        if job_details.empty:
            return 0
        
        # Get average MSU usage and duration
        avg_msu = job_details['MSU Used'].mean()
        avg_duration = job_details['Duration (min)'].mean()
        
        # Get hourly MSU usage
        hourly_msu = self.get_hourly_msu_usage()
        
        # Calculate current average MSU environment during job execution
        current_avg_msu_env = 0
        count = 0
        
        for _, job in job_details.iterrows():
            job_hours = []
            current_hour = job['Start Time'].replace(minute=0, second=0, microsecond=0)
            end_hour = job['End Time'].replace(minute=0, second=0, microsecond=0)
            
            while current_hour <= end_hour:
                job_hours.append((current_hour.date(), current_hour.hour))
                current_hour += timedelta(hours=1)
            
            for date, hour in job_hours:
                date_str = str(date)
                hour_msu = hourly_msu[(hourly_msu['Date'] == date_str) & (hourly_msu['Hour'] == hour)]
                if not hour_msu.empty:
                    current_avg_msu_env += hour_msu.iloc[0]['MSU Usage']
                    count += 1
        
        if count > 0:
            current_avg_msu_env /= count
        
        # Calculate proposed average MSU environment
        proposed_avg_msu_env = 0
        count = 0
        
        proposed_hours = []
        current_hour = new_start_time
        end_hour = new_start_time + timedelta(minutes=avg_duration)
        
        while current_hour <= end_hour:
            proposed_hours.append((current_hour.date(), current_hour.hour))
            current_hour += timedelta(hours=1)
        
        for date, hour in proposed_hours:
            date_str = str(date)
            hour_msu = hourly_msu[(hourly_msu['Date'] == date_str) & (hourly_msu['Hour'] == hour)]
            if not hour_msu.empty:
                proposed_avg_msu_env += hour_msu.iloc[0]['MSU Usage']
                count += 1
        
        if count > 0:
            proposed_avg_msu_env /= count
        
        # Estimate savings (assuming MSU cost is proportional to peak usage)
        if current_avg_msu_env > 0:
            savings_percentage = (current_avg_msu_env - proposed_avg_msu_env) / current_avg_msu_env * 100
            return max(0, savings_percentage)
        
        return 0
    
    def calculate_total_savings(self, optimization_suggestions):
        """
        Calculate total estimated MSU savings from all optimization suggestions.
        
        Args:
            optimization_suggestions (dict): Dictionary of job names and their rescheduling suggestions.
            
        Returns:
            float: Total estimated MSU savings percentage.
        """
        total_msu = self.df['MSU Used'].sum()
        saved_msu = 0
        
        for job_name, suggestions in optimization_suggestions.items():
            if suggestions:
                # Get job details
                job_details = self.get_job_details(job_name)
                
                if not job_details.empty:
                    # Get total MSU for this job
                    job_msu = job_details['MSU Used'].sum()
                    
                    # Get best suggestion
                    best_suggestion = max(suggestions, key=lambda x: x['Estimated Savings'])
                    
                    # Calculate saved MSU
                    saved_msu += job_msu * (best_suggestion['Estimated Savings'] / 100)
        
        if total_msu > 0:
            return (saved_msu / total_msu) * 100
        
        return 0
