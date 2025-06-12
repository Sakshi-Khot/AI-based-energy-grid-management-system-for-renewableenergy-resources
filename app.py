from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import requests
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

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
            raise FileNotFoundError(f"Models directory '{self.models_dir}' not found.")
        
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
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
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
        known_conditions = self.categorical_values['weather_conditions']
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
        
        mapped_condition = weather_mapping.get(api_weather, api_weather)
        return mapped_condition if mapped_condition in known_conditions else known_conditions[0]

    def prepare_input_features(self, date_obj, hour, region, temperature, weather):
        """Prepare input features for prediction"""
        pred_df = pd.DataFrame({
            'Date': [date_obj],
            'Hour': [hour],
            'Region': [region],
            'Temperature': [temperature],
            'Weather': [weather],
            'Month': [date_obj.month],
            'DayOfWeek': [date_obj.weekday()],
            'Battery_Level': [0],
            'Grid_Supply': [0]
        })
        
        pred_df['Hour_Sin'] = np.sin(2 * np.pi * pred_df['Hour']/24)
        pred_df['Hour_Cos'] = np.cos(2 * np.pi * pred_df['Hour']/24)
        pred_df['Month_Sin'] = np.sin(2 * np.pi * pred_df['Month']/12)
        pred_df['Month_Cos'] = np.cos(2 * np.pi * pred_df['Month']/12)
        
        pred_df = pd.get_dummies(pred_df, columns=['Region', 'Weather'])
        
        missing_cols = set(self.feature_cols) - set(pred_df.columns)
        for col in missing_cols:
            pred_df[col] = 0
            
        pred_df = pred_df[self.feature_cols]
        X_pred = self.scaler.transform(pred_df.values)
        
        return X_pred

    def predict_energy_metrics(self, date_obj, hour, region, temperature, weather):
        """Predict energy metrics for the given parameters"""
        if region not in self.categorical_values['regions']:
            region = self.categorical_values['regions'][0]
        
        weather = self.map_weather_condition(weather)
        X_pred = self.prepare_input_features(date_obj, hour, region, temperature, weather)
        
        predictions = {}
        for target, model in self.models.items():
            predictions[target] = max(0, model.predict(X_pred)[0])
        
        return predictions

    def optimize_energy_allocation(self, predictions):
        """Optimize energy allocation based on predictions"""
        solar = predictions['Solar_Generation']
        wind = predictions['Wind_Generation']
        hydro = predictions.get('Hydro_Generation', 0)
        total_consumption = predictions['Total_Consumption']
        
        total_renewable = solar + wind + hydro
        
        if total_renewable >= total_consumption:
            solar_used = min(solar, total_consumption)
            remaining = total_consumption - solar_used
            wind_used = min(wind, remaining)
            remaining -= wind_used
            hydro_used = min(hydro, remaining)
            grid_supply = 0
            excess_renewable = total_renewable - total_consumption
            battery_charge = excess_renewable
        else:
            solar_used = solar
            wind_used = wind
            hydro_used = hydro
            grid_supply = total_consumption - total_renewable
            battery_charge = 0
        
        renewable_usage_pct = (solar_used + wind_used + hydro_used) / total_consumption * 100
        grid_dependency_pct = grid_supply / total_consumption * 100
        
        return {
            'solar_used': solar_used,
            'wind_used': wind_used,
            'hydro_used': hydro_used,
            'grid_supply': grid_supply,
            'battery_charge': battery_charge,
            'renewable_usage_pct': renewable_usage_pct,
            'grid_dependency_pct': grid_dependency_pct
        }

    def generate_alerts(self, predictions, optimization):
        """Generate alerts based on predictions and optimization"""
        alerts = []
        
        if predictions['Total_Consumption'] > self.historical_avg['Total_Consumption'] * 1.2:
            alerts.append("High overall consumption forecasted (20% above average).")
        
        if predictions['Residential_Consumption'] > self.historical_avg['Residential_Consumption'] * 1.3:
            alerts.append("High residential consumption forecasted (30% above average).")
        
        if predictions['Commercial_Consumption'] > self.historical_avg['Commercial_Consumption'] * 1.3:
            alerts.append("High commercial consumption forecasted. Consider peak pricing adjustment.")
        
        if predictions['Industrial_Consumption'] > self.historical_avg['Industrial_Consumption'] * 1.3:
            alerts.append("High industrial consumption forecasted. Consider load balancing.")
        
        if predictions['Solar_Generation'] < self.historical_avg['Solar_Generation'] * 0.7:
            alerts.append("Low solar generation forecasted (30% below average).")
        
        if predictions['Wind_Generation'] < self.historical_avg['Wind_Generation'] * 0.7:
            alerts.append("Low wind generation forecasted (30% below average).")
        
        if optimization['grid_dependency_pct'] > 50:
            alerts.append("High grid dependency (>50%). Consider demand response measures.")
        
        return alerts

# Initialize the dashboard
dashboard = EnergyGridDashboard()

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html')

@app.route('/get_dashboard_data', methods=['POST'])
def get_dashboard_data():
    """Get dashboard data for the specified location"""
    location = request.json.get('location', 'Bangalore')
    
    # Get current date and time
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    hour = now.hour
    
    # Get weather data
    weather_data = dashboard.get_weather_data(location)
    if not weather_data:
        return jsonify({'error': 'Could not fetch weather data'}), 400
    
    # Make predictions
    predictions = dashboard.predict_energy_metrics(
        now, hour, location, 
        weather_data['temperature'], 
        weather_data['weather']
    )
    
    # Optimize energy allocation
    optimization = dashboard.optimize_energy_allocation(predictions)
    
    # Generate alerts
    alerts = dashboard.generate_alerts(predictions, optimization)
    
    # Prepare response data
    response_data = {
        'date': date_str,
        'time': f"{hour:02d}:00",
        'weather': {
            'temperature': weather_data['temperature'],
            'condition': weather_data['weather'],
            'description': weather_data['description'],
            'humidity': weather_data['humidity'],
            'wind_speed': weather_data['wind_speed'],
            'clouds': weather_data['clouds']
        },
        'predictions': {
            'solar': predictions['Solar_Generation'],
            'wind': predictions['Wind_Generation'],
            'hydro': predictions.get('Hydro_Generation', 0),
            'total_renewable': predictions['Solar_Generation'] + predictions['Wind_Generation'] + predictions.get('Hydro_Generation', 0),
            'residential': predictions['Residential_Consumption'],
            'commercial': predictions['Commercial_Consumption'],
            'industrial': predictions['Industrial_Consumption'],
            'total_consumption': predictions['Total_Consumption']
        },
        'optimization': optimization,
        'alerts': alerts
    }
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True) 