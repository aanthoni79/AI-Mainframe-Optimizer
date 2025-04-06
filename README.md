# AI Mainframe Optimizer

## Overview
The AI Mainframe Optimizer is a Streamlit application designed to analyze mainframe job data (SMF-like logs) to identify optimization opportunities and provide recommendations for reducing MSU usage and costs.

## Features
- **MSU Usage Analysis**: Identify top MSU-consuming jobs and time periods
- **Peak Detection**: Highlight time windows with MSU overload
- **Job Analysis**: Detailed analysis of individual job performance
- **Optimization Recommendations**: Suggest job rescheduling to reduce peak MSU usage
- **Savings Calculator**: Estimate cost savings from implementing recommendations
- **MSU Forecasting**: Predict future MSU usage patterns using Prophet

## Application Structure
- `app.py`: Main Streamlit application
- `src/`: Source code directory
  - `data_processor.py`: Data processing and analysis functions
  - `forecast_engine.py`: MSU usage forecasting using Prophet
  - `visualization.py`: Data visualization components
- `utils/`: Utility functions
  - `helpers.py`: Helper functions for formatting and summaries
- `data/`: Data directory containing sample data
- `static/`: Static assets directory

## Installation
1. Clone the repository
2. Install dependencies:
   ```
   pip install streamlit pandas numpy plotly matplotlib prophet scikit-learn
   ```

## Usage
1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```
2. Upload SMF/CSV job data or use the provided sample data
3. Explore the different tabs to analyze MSU usage and get optimization recommendations

## Deployment
The application is deployed and accessible at:
https://optizframe.streamlit.app/

## Sample Data
The application includes sample data in CSV format with the following columns:
- Job Name: Name of the mainframe job
- MSU Used: MSU consumption for the job
- Start Time: Job start timestamp
- Duration (min): Job duration in minutes
- CPU Time (s): CPU time in seconds
- IO Count: Number of I/O operations
- Enqueue: Enqueue count
- Storage Used (MB): Storage usage in MB
- Job Class: Job classification

## Optimization Methodology
The application identifies optimization opportunities by:
1. Detecting peak MSU usage windows
2. Identifying jobs running during these peak periods
3. Finding alternative time slots with lower MSU usage
4. Calculating potential savings from rescheduling

## Future Enhancements
- Integration with real-time mainframe monitoring systems
- Advanced machine learning for predictive job scheduling
- What-if simulator for testing different optimization scenarios
- Export functionality for reports and recommendations
