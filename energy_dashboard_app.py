import pandas as pd
import numpy as np
import joblib
import os
import requests
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# OpenWeather API key - replace with your own API key
OPENWEATHER_API_KEY = "b337c39de6f64e181fd2411f6e7ac435"

class EnergyGridDashboard:
    def __init__(self, models_dir='models'):
        """Initialize the dashboard with trained models"""
        self.models_dir = models_dir
        self.models = {}
        self.scaler = None
        self.feature_cols = None
        self.historical_avg = None
        self.categorical_values = None
        
        # Load models and related data
        self.load_models()
    
    def load_models(self):
        """Load trained models and related data"""
        print("Loading trained models...")
        
        # Check if models directory exists
        if not os.path.exists(self.models_dir):
            raise FileNotFoundError(f"Models directory '{self.models_dir}' not found. Please run train_energy_models.py first.")
        
        # Load feature columns
        self.feature_cols = joblib.load(f'{self.models_dir}/feature_cols.pkl')
        
        # Load scaler
        self.scaler = joblib.load(f'{self.models_dir}/scaler.pkl')
        
        # Load historical averages
        self.historical_avg = joblib.load(f'{self.models_dir}/historical_avg.pkl')
        
        # Load categorical values
        self.categorical_values = joblib.load(f'{self.models_dir}/categorical_values.pkl')
        
        # Load all models
        target_columns = [
            'Solar_Generation', 'Wind_Generation', 'Hydro_Generation',
            'Total_Consumption', 'Residential_Consumption', 
            'Commercial_Consumption', 'Industrial_Consumption'
        ]
        
        for target in target_columns:
            model_path = f'{self.models_dir}/{target}_model.pkl'
            if os.path.exists(model_path):
                self.models[target] = joblib.load(model_path)
                print(f"  - Loaded {target} model")
    
    def get_weather_data(self, location):
        """Get real-time weather data from OpenWeather API"""
        print(f"Fetching weather data for {location}...")
        
        # Make API request
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Error fetching weather data: {response.status_code}")
            print(response.text)
            return None
        
        # Parse response
        data = response.json()
        
        # Extract relevant information
        weather_data = {
            'temperature': data['main']['temp'],
            'weather': data['weather'][0]['main'],
            'description': data['weather'][0]['description'],
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed'],
            'clouds': data.get('clouds', {}).get('all', 0)
        }
        
        return weather_data
    
    def map_weather_condition(self, api_weather):
        """Map OpenWeather API weather condition to our model's weather categories"""
        # Get the list of weather conditions our model knows
        known_conditions = self.categorical_values['weather_conditions']
        
        # Simple mapping from OpenWeather conditions to our categories
        weather_mapping = {
            'Clear': 'Sunny',
            'Clouds': 'Cloudy',
            'Rain': 'Rainy',
            'Drizzle': 'Rainy',
            'Thunderstorm': 'Rainy',
            'Snow': 'Snowy',
            'Mist': 'Foggy',
            'Fog': 'Foggy',
            'Haze': 'Foggy'
        }
        
        # Get the mapped condition
        mapped_condition = weather_mapping.get(api_weather, api_weather)
        
        # Check if the mapped condition is in our known conditions
        if mapped_condition in known_conditions:
            return mapped_condition
        
        # If not, return the most similar condition or the first one as default
        print(f"Warning: Weather condition '{api_weather}' not in training data. Using '{known_conditions[0]}' instead.")
        return known_conditions[0]
    
    def prepare_input_features(self, date_obj, hour, region, temperature, weather):
        """Prepare input features for prediction"""
        # Create a dataframe for the prediction input
        pred_df = pd.DataFrame({
            'Date': [date_obj],
            'Hour': [hour],
            'Region': [region],
            'Temperature': [temperature],
            'Weather': [weather],
            'Month': [date_obj.month],
            'DayOfWeek': [date_obj.weekday()],
            'Battery_Level': [0],  # Default value
            'Grid_Supply': [0]     # Default value
        })
        
        # Create feature engineering for time-based patterns
        pred_df['Hour_Sin'] = np.sin(2 * np.pi * pred_df['Hour']/24)
        pred_df['Hour_Cos'] = np.cos(2 * np.pi * pred_df['Hour']/24)
        pred_df['Month_Sin'] = np.sin(2 * np.pi * pred_df['Month']/12)
        pred_df['Month_Cos'] = np.cos(2 * np.pi * pred_df['Month']/12)
        
        # One-hot encode categorical variables
        pred_df = pd.get_dummies(pred_df, columns=['Region', 'Weather'])
        
        # Ensure all necessary columns are present
        missing_cols = set(self.feature_cols) - set(pred_df.columns)
        for col in missing_cols:
            pred_df[col] = 0
            
        # Ensure columns are in the same order as during training
        pred_df = pred_df[self.feature_cols]
        
        # Scale features
        X_pred = self.scaler.transform(pred_df.values)
        
        return X_pred
    
    def predict_energy_metrics(self, date_obj, hour, region, temperature, weather):
        """Predict energy metrics for the given parameters"""
        # Check if region is in our training data
        if region not in self.categorical_values['regions']:
            available_regions = ', '.join(self.categorical_values['regions'])
            print(f"Warning: Region '{region}' not in training data. Available regions: {available_regions}")
            region = self.categorical_values['regions'][0]
            print(f"Using '{region}' instead.")
        
        # Map weather condition to our categories
        weather = self.map_weather_condition(weather)
        
        # Prepare input features
        X_pred = self.prepare_input_features(date_obj, hour, region, temperature, weather)
        
        # Make predictions
        predictions = {}
        for target, model in self.models.items():
            predictions[target] = max(0, model.predict(X_pred)[0])  # Ensure non-negative values
        
        return predictions
    
    def optimize_energy_allocation(self, predictions):
        """Optimize energy allocation based on predictions"""
        # Extract predictions
        solar = predictions['Solar_Generation']
        wind = predictions['Wind_Generation']
        hydro = predictions.get('Hydro_Generation', 0)
        total_consumption = predictions['Total_Consumption']
        
        # Total renewable energy available
        total_renewable = solar + wind + hydro
        
        # Determine how much to use from each source
        if total_renewable >= total_consumption:
            # We have excess renewable energy
            solar_used = min(solar, total_consumption)
            remaining = total_consumption - solar_used
            wind_used = min(wind, remaining)
            remaining -= wind_used
            hydro_used = min(hydro, remaining)
            grid_supply = 0
            
            # Calculate excess for battery storage
            excess_renewable = total_renewable - total_consumption
            battery_charge = excess_renewable
            
            # Calculate battery health based on excess energy
            battery_health = min(100, (excess_renewable / total_renewable) * 100)
        else:
            # We need to use grid supply
            solar_used = solar
            wind_used = wind
            hydro_used = hydro
            grid_supply = total_consumption - total_renewable
            battery_charge = 0
            battery_health = 0  # No excess energy means no battery charging
        
        # Calculate efficiency metrics
        renewable_usage_pct = (solar_used + wind_used + hydro_used) / total_consumption * 100
        grid_dependency_pct = grid_supply / total_consumption * 100
        
        return {
            'solar_used': solar_used,
            'wind_used': wind_used,
            'hydro_used': hydro_used,
            'grid_supply': grid_supply,
            'battery_charge': battery_charge,
            'battery_health': battery_health,
            'renewable_usage_pct': renewable_usage_pct,
            'grid_dependency_pct': grid_dependency_pct
        }
    
    def generate_alerts(self, predictions, optimization):
        """Generate alerts based on predictions and optimization"""
        alerts = []
        
        # Check for high consumption
        if predictions['Total_Consumption'] > self.historical_avg['Total_Consumption'] * 1.2:
            alerts.append("High overall consumption forecasted (20% above average).")
        
        # Check sector-specific consumption
        if predictions['Residential_Consumption'] > self.historical_avg['Residential_Consumption'] * 1.3:
            alerts.append("High residential consumption forecasted (30% above average).")
        
        if predictions['Commercial_Consumption'] > self.historical_avg['Commercial_Consumption'] * 1.3:
            alerts.append("High commercial consumption forecasted. Consider peak pricing adjustment.")
        
        if predictions['Industrial_Consumption'] > self.historical_avg['Industrial_Consumption'] * 1.3:
            alerts.append("High industrial consumption forecasted. Consider load balancing.")
        
        # Check renewable generation
        if predictions['Solar_Generation'] < self.historical_avg['Solar_Generation'] * 0.7:
            alerts.append("Low solar generation forecasted (30% below average).")
        
        if predictions['Wind_Generation'] < self.historical_avg['Wind_Generation'] * 0.7:
            alerts.append("Low wind generation forecasted (30% below average).")
        
        # Grid dependency alert
        if optimization['grid_dependency_pct'] > 50:
            alerts.append("High grid dependency (>50%). Consider demand response measures.")
        
        return alerts
    
    def generate_dashboard(self, location):
        """Generate a dashboard output for the current time and specified location"""
        # Get current date and time
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        hour = now.hour
        
        # Get real-time weather data
        weather_data = self.get_weather_data(location)
        if not weather_data:
            print("Could not fetch weather data. Using default values.")
            temperature = 25.0
            weather = "Clear"
        else:
            temperature = weather_data['temperature']
            weather = weather_data['weather']
        
        # Make predictions
        predictions = self.predict_energy_metrics(now, hour, location, temperature, weather)
        
        # Optimize energy allocation
        optimization = self.optimize_energy_allocation(predictions)
        
        # Generate alerts
        alerts = self.generate_alerts(predictions, optimization)
        
        # Format the dashboard output without emoji characters
        dashboard = f"""
AI-Powered Energy Grid Management Output
=======================================
Date: {date_str}
Time: {hour:02d}:00
Temperature: {temperature:.1f}°C
Weather: {weather}
Region: {location}

Predicted Renewable Energy Production (Next Hour):
  Solar Power: {predictions['Solar_Generation']:.1f} kWh
  Wind Power: {predictions['Wind_Generation']:.1f} kWh
  Hydro Power: {predictions.get('Hydro_Generation', 0):.1f} kWh
  Total Renewable: {predictions['Solar_Generation'] + predictions['Wind_Generation'] + predictions.get('Hydro_Generation', 0):.1f} kWh

Predicted Energy Consumption (Next Hour):
  Residential: {predictions['Residential_Consumption']:.1f} kWh
  Commercial: {predictions['Commercial_Consumption']:.1f} kWh
  Industrial: {predictions['Industrial_Consumption']:.1f} kWh
  Total Consumption: {predictions['Total_Consumption']:.1f} kWh

Optimization Summary:
  From Solar: {optimization['solar_used']:.1f} kWh ({100 * optimization['solar_used'] / max(0.1, predictions['Solar_Generation']):.1f}% used)
  From Wind: {optimization['wind_used']:.1f} kWh ({100 * optimization['wind_used'] / max(0.1, predictions['Wind_Generation']):.1f}% used)
  From Hydro: {optimization['hydro_used']:.1f} kWh
  From Main Grid: {optimization['grid_supply']:.1f} kWh (to meet demand)
  Battery Storage: {"Charged with " + f"{optimization['battery_charge']:.1f} kWh excess" if optimization['battery_charge'] > 0 else "No charging needed"}

Efficiency Report:
  Renewable Usage: {optimization['renewable_usage_pct']:.1f}%
  Grid Dependency: {optimization['grid_dependency_pct']:.1f}%
  Battery Health: {'Good' if optimization['battery_health'] > 70 else 'Fair' if optimization['battery_health'] > 40 else 'Poor'} ({optimization['battery_health']:.1f}% capacity)

Alerts:
"""
        
        if alerts:
            for alert in alerts:
                dashboard += f"  - {alert}\n"
        else:
            dashboard += "  - No alerts at this time.\n"
        
        # Add weather details
        if weather_data:
            dashboard += f"""
Detailed Weather Information:
  Description: {weather_data['description']}
  Humidity: {weather_data['humidity']}%
  Wind Speed: {weather_data['wind_speed']} m/s
  Cloud Cover: {weather_data['clouds']}%
"""
        
        return dashboard

def main():
    print("AI-based Grid Management System for Renewable Sources")
    print("====================================================")
    
    try:
        # Initialize the dashboard with trained models
        dashboard = EnergyGridDashboard()
        
        # Get location from user
        location = input("Enter your location (e.g., Bangalore): ")
        
        # Generate and print the dashboard
        output = dashboard.generate_dashboard(location)
        print("\n" + output)
        
        # Save the dashboard to a file
        with open('energy_dashboard_output.txt', 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Dashboard output saved to energy_dashboard_output.txt")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 