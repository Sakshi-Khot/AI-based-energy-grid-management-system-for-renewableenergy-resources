import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnergyGridDashboard:
    def __init__(self, data_path):
        """Initialize the dashboard with data"""
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.scaler = None
        self.feature_cols = None
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Train prediction models
        self.train_models()
    
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        print("Loading and preprocessing data...")
        self.df = pd.read_csv(self.data_path)
        
        # Convert Date to datetime
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Create day of week feature
        self.df['DayOfWeek'] = self.df['Date'].dt.dayofweek
        
        # Handle missing values if any
        if self.df.isnull().sum().any():
            print("Handling missing values...")
            # Fill numeric columns with median
            numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                self.df[col].fillna(self.df[col].median(), inplace=True)
            
            # Fill categorical columns with mode
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        
        # Check if we have consumption breakdown data
        if not all(col in self.df.columns for col in ['Residential_Consumption', 'Commercial_Consumption', 'Industrial_Consumption']):
            print("Creating consumption breakdown based on region and hour patterns...")
            # Create synthetic breakdown based on time of day and region
            self.create_consumption_breakdown()
    
    def create_consumption_breakdown(self):
        """Create synthetic breakdown of consumption data for dashboard"""
        # Residential: Higher in mornings and evenings, lower during work hours
        residential_pattern = np.array([0.7, 0.65, 0.6, 0.55, 0.5, 0.6, 0.7, 0.65, 0.5, 0.4, 0.35, 0.3, 
                                       0.3, 0.3, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.75, 0.7])
        
        # Commercial: Higher during work hours
        commercial_pattern = np.array([0.2, 0.15, 0.1, 0.1, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0,
                                      0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.2])
        
        # Industrial: More consistent with slight variations
        industrial_pattern = np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.6])
        
        # Scale patterns to sum to 1.0 at each hour
        for hour in range(24):
            total = residential_pattern[hour] + commercial_pattern[hour] + industrial_pattern[hour]
            residential_pattern[hour] /= total
            commercial_pattern[hour] /= total
            industrial_pattern[hour] /= total
        
        # Create consumption breakdown
        self.df['Residential_Consumption'] = self.df.apply(
            lambda row: row['Total_Consumption'] * residential_pattern[int(row['Hour'])], axis=1
        )
        self.df['Commercial_Consumption'] = self.df.apply(
            lambda row: row['Total_Consumption'] * commercial_pattern[int(row['Hour'])], axis=1
        )
        self.df['Industrial_Consumption'] = self.df.apply(
            lambda row: row['Total_Consumption'] * industrial_pattern[int(row['Hour'])], axis=1
        )
        
        # Add some region-based variation
        region_factors = self.df.groupby('Region')['Total_Consumption'].mean()
        region_factors = region_factors / region_factors.mean()
        
        for region, factor in region_factors.items():
            mask = self.df['Region'] == region
            if factor > 1.1:  # High consumption regions get more industrial
                self.df.loc[mask, 'Industrial_Consumption'] *= 1.2
                self.df.loc[mask, 'Commercial_Consumption'] *= 0.9
                self.df.loc[mask, 'Residential_Consumption'] *= 0.9
            elif factor < 0.9:  # Low consumption regions get more residential
                self.df.loc[mask, 'Industrial_Consumption'] *= 0.8
                self.df.loc[mask, 'Commercial_Consumption'] *= 0.9
                self.df.loc[mask, 'Residential_Consumption'] *= 1.3
        
        # Ensure the sum matches Total_Consumption
        for idx in self.df.index:
            total = (self.df.loc[idx, 'Residential_Consumption'] + 
                    self.df.loc[idx, 'Commercial_Consumption'] + 
                    self.df.loc[idx, 'Industrial_Consumption'])
            
            factor = self.df.loc[idx, 'Total_Consumption'] / total
            self.df.loc[idx, 'Residential_Consumption'] *= factor
            self.df.loc[idx, 'Commercial_Consumption'] *= factor
            self.df.loc[idx, 'Industrial_Consumption'] *= factor
    
    def prepare_features(self, df):
        """Prepare features for model training/prediction"""
        # Create feature engineering for time-based patterns
        df = df.copy()
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour']/24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour']/24)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month']/12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month']/12)
        
        # One-hot encode categorical variables
        df = pd.get_dummies(df, columns=['Region', 'Weather'])
        
        return df
    
    def train_models(self):
        """Train prediction models for various energy metrics"""
        print("Training prediction models...")
        
        # Prepare features
        df_features = self.prepare_features(self.df)
        
        # Define target columns to predict
        target_columns = [
            'Solar_Generation', 'Wind_Generation', 'Hydro_Generation',
            'Total_Consumption', 'Residential_Consumption', 
            'Commercial_Consumption', 'Industrial_Consumption'
        ]
        
        # Features to use (exclude targets and date)
        self.feature_cols = [col for col in df_features.columns if col not in target_columns + ['Date']]
        
        # Initialize scaler
        self.scaler = StandardScaler()
        X = df_features[self.feature_cols].values
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Train a model for each target
        for target in target_columns:
            if target in df_features.columns:
                y = df_features[target].values
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
                
                # Train model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                self.models[target] = model
                
                # Evaluate
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                print(f"  - {target} model trained: Train R² = {train_score:.4f}, Test R² = {test_score:.4f}")
    
    def predict_next_hour(self, date_str, hour, region, temperature, weather):
        """Predict energy metrics for the next hour"""
        # Create a dataframe for the prediction input
        date_obj = pd.to_datetime(date_str)
        pred_df = pd.DataFrame({
            'Date': [date_obj],
            'Hour': [hour],
            'Region': [region],
            'Temperature': [temperature],
            'Weather': [weather],
            'Month': [date_obj.month],
            'DayOfWeek': [date_obj.dayofweek],
            'Battery_Level': [0],  # Default value
            'Grid_Supply': [0]     # Default value
        })
        
        # Prepare features
        pred_features = self.prepare_features(pred_df)
        
        # Ensure all necessary columns are present
        missing_cols = set(self.feature_cols) - set(pred_features.columns)
        for col in missing_cols:
            pred_features[col] = 0
            
        # Ensure columns are in the same order as during training
        pred_features = pred_features[self.feature_cols]
        
        # Scale features
        X_pred = self.scaler.transform(pred_features.values)
        
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
        else:
            # We need to use grid supply
            solar_used = solar
            wind_used = wind
            hydro_used = hydro
            grid_supply = total_consumption - total_renewable
            battery_charge = 0
        
        # Calculate efficiency metrics
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
    
    def generate_alerts(self, predictions, optimization, historical_avg=None):
        """Generate alerts based on predictions and optimization"""
        alerts = []
        
        # Get historical averages if not provided
        if historical_avg is None:
            historical_avg = {
                'Total_Consumption': self.df['Total_Consumption'].mean(),
                'Residential_Consumption': self.df['Residential_Consumption'].mean(),
                'Commercial_Consumption': self.df['Commercial_Consumption'].mean(),
                'Industrial_Consumption': self.df['Industrial_Consumption'].mean(),
                'Solar_Generation': self.df['Solar_Generation'].mean(),
                'Wind_Generation': self.df['Wind_Generation'].mean()
            }
        
        # Check for high consumption
        if predictions['Total_Consumption'] > historical_avg['Total_Consumption'] * 1.2:
            alerts.append("High overall consumption forecasted (20% above average).")
        
        # Check sector-specific consumption
        if predictions['Residential_Consumption'] > historical_avg['Residential_Consumption'] * 1.3:
            alerts.append("High residential consumption forecasted (30% above average).")
        
        if predictions['Commercial_Consumption'] > historical_avg['Commercial_Consumption'] * 1.3:
            alerts.append("High commercial consumption forecasted. Consider peak pricing adjustment.")
        
        if predictions['Industrial_Consumption'] > historical_avg['Industrial_Consumption'] * 1.3:
            alerts.append("High industrial consumption forecasted. Consider load balancing.")
        
        # Check renewable generation
        if predictions['Solar_Generation'] < historical_avg['Solar_Generation'] * 0.7:
            alerts.append("Low solar generation forecasted (30% below average).")
        
        if predictions['Wind_Generation'] < historical_avg['Wind_Generation'] * 0.7:
            alerts.append("Low wind generation forecasted (30% below average).")
        
        # Grid dependency alert
        if optimization['grid_dependency_pct'] > 50:
            alerts.append("High grid dependency (>50%). Consider demand response measures.")
        
        return alerts
    
    def generate_dashboard(self, date_str, hour, region, temperature, weather, battery_health=85):
        """Generate a dashboard output for the specified parameters"""
        # Make predictions
        predictions = self.predict_next_hour(date_str, hour, region, temperature, weather)
        
        # Optimize energy allocation
        optimization = self.optimize_energy_allocation(predictions)
        
        # Generate alerts
        alerts = self.generate_alerts(predictions, optimization)
        
        # Format the dashboard output
        dashboard = f"""
AI-Powered Energy Grid Management Output
=======================================
Date: {date_str}
Time: {hour:02d}:00
Temperature: {temperature}°C
Weather: {weather}
Region: {region}

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
  Battery Health: {'Good' if battery_health > 70 else 'Fair' if battery_health > 40 else 'Poor'} ({battery_health}% capacity)

Alerts:
"""
        
        if alerts:
            for alert in alerts:
                dashboard += f"  - {alert}\n"
        else:
            dashboard += "  - No alerts at this time.\n"
        
        return dashboard

def get_user_input():
    """Get input parameters from the user"""
    print("\nEnter prediction parameters:")
    
    # Get date
    while True:
        try:
            date_str = input("Date (YYYY-MM-DD): ")
            date_obj = pd.to_datetime(date_str)
            break
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")
    
    # Get hour
    while True:
        try:
            hour = int(input("Hour (0-23): "))
            if 0 <= hour <= 23:
                break
            else:
                print("Hour must be between 0 and 23.")
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 23.")
    
    # Get region
    region = input("Region (e.g., Bangalore, Mysore): ")
    
    # Get temperature
    while True:
        try:
            temperature = float(input("Temperature (°C): "))
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Get weather
    weather = input("Weather (e.g., Sunny, Rainy, Cloudy): ")
    
    # Get battery health
    while True:
        try:
            battery_health = float(input("Battery Health (0-100%): "))
            if 0 <= battery_health <= 100:
                break
            else:
                print("Battery health must be between 0 and 100.")
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 100.")
    
    return date_str, hour, region, temperature, weather, battery_health

def main():
    print("AI-based Grid Management System for Renewable Sources")
    print("====================================================")
    
    try:
        # Initialize the dashboard with the real data
        dashboard = EnergyGridDashboard('karnataka_energy_data_2024_final.csv')
        
        while True:
            # Get user input for prediction
            date_str, hour, region, temperature, weather, battery_health = get_user_input()
            
            # Generate and print the dashboard
            output = dashboard.generate_dashboard(date_str, hour, region, temperature, weather, battery_health)
            print("\n" + output)
            
            # Save the dashboard to a file
            with open('energy_dashboard_output.txt', 'w') as f:
                f.write(output)
            print(f"Dashboard output saved to energy_dashboard_output.txt")
            
            # Ask if user wants to make another prediction
            another = input("\nGenerate another prediction? (y/n): ").lower()
            if another != 'y':
                break
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 