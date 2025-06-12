import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def generate_dashboard():
    """Generate a sample dashboard for the AI-based Grid Management System"""
    
    # Sample data for demonstration
    sample_data = {
        'Solar_Generation': 320.0,
        'Wind_Generation': 450.0,
        'Hydro_Generation': 0.0,
        'Total_Consumption': 1200.0,
        'Residential_Consumption': 500.0,
        'Commercial_Consumption': 400.0,
        'Industrial_Consumption': 300.0
    }
    
    # Sample optimization
    optimization = {
        'solar_used': 300.0,
        'wind_used': 450.0,
        'hydro_used': 0.0,
        'grid_supply': 450.0,
        'battery_charge': 20.0,
        'renewable_usage_pct': 64.2,
        'grid_dependency_pct': 35.8
    }
    
    # Format the dashboard output
    dashboard = """
AI-Powered Energy Grid Management Output
=======================================
Date: 2025-04-11
Time: 14:30
Temperature: 32.5°C
Weather: Sunny
Region: Bangalore

Predicted Renewable Energy Production (Next Hour):
  Solar Power: {:.1f} kWh
  Wind Power: {:.1f} kWh
  Hydro Power: {:.1f} kWh
  Total Renewable: {:.1f} kWh

Predicted Energy Consumption (Next Hour):
  Residential: {:.1f} kWh
  Commercial: {:.1f} kWh
  Industrial: {:.1f} kWh
  Total Consumption: {:.1f} kWh

Optimization Summary:
  From Solar: {:.1f} kWh (used)
  From Wind: {:.1f} kWh (used fully)
  From Main Grid: {:.1f} kWh (to meet demand)
  Battery Storage: Charged with {:.1f} kWh excess solar

Efficiency Report:
  Renewable Usage: {:.1f}%
  Grid Dependency: {:.1f}%
  Battery Health: Good (85% capacity)

Alerts:
  - High consumption forecasted for Industrial sector next hour.
  - Consider peak pricing adjustment for Commercial sector.
""".format(
        sample_data['Solar_Generation'],
        sample_data['Wind_Generation'],
        sample_data['Hydro_Generation'],
        sample_data['Solar_Generation'] + sample_data['Wind_Generation'] + sample_data['Hydro_Generation'],
        sample_data['Residential_Consumption'],
        sample_data['Commercial_Consumption'],
        sample_data['Industrial_Consumption'],
        sample_data['Total_Consumption'],
        optimization['solar_used'],
        optimization['wind_used'],
        optimization['grid_supply'],
        optimization['battery_charge'],
        optimization['renewable_usage_pct'],
        optimization['grid_dependency_pct']
    )
    
    return dashboard

def analyze_dataset():
    """Analyze the current dataset and suggest improvements"""
    try:
        # Try to load the dataset
        df = pd.read_csv('karnataka_energy_data_2024_final.csv', nrows=10)
        
        # Get column names
        columns = df.columns.tolist()
        
        analysis = """
Dataset Analysis
===============
Current columns in dataset: {}

Suggested Improvements for AI-based Grid Management:
1. Add separate consumption columns for different sectors:
   - Residential_Consumption
   - Commercial_Consumption
   - Industrial_Consumption

2. Add more detailed weather data:
   - Cloud_Cover_Percentage
   - Wind_Speed
   - Humidity
   - Precipitation

3. Add grid stability metrics:
   - Voltage_Fluctuation
   - Frequency_Deviation
   - Grid_Stability_Index

4. Add time-based features:
   - Is_Weekend
   - Is_Holiday
   - Season

5. Add predictive columns:
   - Predicted_Consumption
   - Predicted_Generation
   - Optimization_Strategy

These additions would significantly improve the model's ability to predict energy needs and optimize grid management.
""".format(columns)
        
        return analysis
    except Exception as e:
        return f"Error analyzing dataset: {e}"

def main():
    # Generate the dashboard
    dashboard_output = generate_dashboard()
    print(dashboard_output)
    
    # Save the dashboard to a file
    with open('energy_dashboard_output.txt', 'w') as f:
        f.write(dashboard_output)
    print("Dashboard output saved to energy_dashboard_output.txt")
    
    # Analyze the dataset
    analysis = analyze_dataset()
    print("\n" + analysis)
    
    # Save the analysis to a file
    with open('dataset_analysis.txt', 'w') as f:
        f.write(analysis)
    print("Dataset analysis saved to dataset_analysis.txt")

if __name__ == "__main__":
    main() 