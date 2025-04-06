import pandas as pd
import numpy as np
from prophet import Prophet

class ForecastEngine:
    """
    Class for forecasting future MSU usage using Prophet.
    """
    
    def __init__(self, data_processor):
        """
        Initialize the ForecastEngine with a DataProcessor instance.
        
        Args:
            data_processor (DataProcessor): Instance of DataProcessor containing job data.
        """
        self.data_processor = data_processor
        self.model = None
        self.forecast = None
    
    def prepare_forecast_data(self):
        """
        Prepare data for forecasting.
        
        Returns:
            pandas.DataFrame: Data prepared for Prophet forecasting.
        """
        # Get hourly MSU usage
        hourly_msu = self.data_processor.get_hourly_msu_usage()
        
        # Convert to Prophet format (ds, y)
        forecast_data = pd.DataFrame()
        forecast_data['ds'] = pd.to_datetime(hourly_msu['Date'] + ' ' + hourly_msu['Hour'].astype(str) + ':00:00')
        forecast_data['y'] = hourly_msu['MSU Usage']
        
        return forecast_data
    
    def train_model(self, periods=24, frequency='H'):
        """
        Train the Prophet model and generate forecast.
        
        Args:
            periods (int, optional): Number of periods to forecast. Defaults to 24.
            frequency (str, optional): Frequency of forecast. Defaults to 'H' (hourly).
            
        Returns:
            pandas.DataFrame: Forecast results.
        """
        # Prepare data
        data = self.prepare_forecast_data()
        
        # Initialize and train model
        self.model = Prophet(
            daily_seasonality=True,
            yearly_seasonality=False,
            weekly_seasonality=True
        )
        self.model.fit(data)
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=frequency)
        
        # Generate forecast
        self.forecast = self.model.predict(future)
        
        return self.forecast
    
    def get_forecast_components(self):
        """
        Get the components of the forecast (trend, seasonality).
        
        Returns:
            dict: Dictionary containing forecast components.
        """
        if self.model is None or self.forecast is None:
            return None
        
        return {
            'trend': self.forecast[['ds', 'trend']],
            'daily': self.forecast[['ds', 'daily']],
            'weekly': self.forecast[['ds', 'weekly']]
        }
    
    def get_forecast_data(self):
        """
        Get the forecast data in a format suitable for visualization.
        
        Returns:
            pandas.DataFrame: Forecast data for visualization.
        """
        if self.forecast is None:
            return None
        
        # Select relevant columns
        forecast_data = self.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        
        # Add actual values
        actual_data = self.prepare_forecast_data()
        forecast_data = pd.merge(forecast_data, actual_data, on='ds', how='left')
        
        # Rename columns for clarity
        forecast_data.rename(columns={
            'ds': 'Timestamp',
            'y': 'Actual MSU',
            'yhat': 'Predicted MSU',
            'yhat_lower': 'Lower Bound',
            'yhat_upper': 'Upper Bound'
        }, inplace=True)
        
        return forecast_data
    
    def identify_future_peaks(self, threshold_percentile=90):
        """
        Identify future peak MSU usage periods.
        
        Args:
            threshold_percentile (int, optional): Percentile threshold for peak detection. Defaults to 90.
            
        Returns:
            pandas.DataFrame: Future peak periods.
        """
        if self.forecast is None:
            return None
        
        # Calculate threshold based on percentile
        threshold = self.forecast['yhat'].quantile(threshold_percentile/100)
        
        # Identify future peaks (only for forecasted periods, not historical)
        actual_data = self.prepare_forecast_data()
        last_actual_date = actual_data['ds'].max()
        
        future_data = self.forecast[self.forecast['ds'] > last_actual_date].copy()
        future_peaks = future_data[future_data['yhat'] >= threshold].copy()
        
        # Format for display
        future_peaks['Time Window'] = future_peaks['ds'].dt.strftime('%Y-%m-%d %H:00-%H:59')
        future_peaks['Predicted MSU'] = future_peaks['yhat']
        
        return future_peaks[['Time Window', 'Predicted MSU']].sort_values('Predicted MSU', ascending=False)
