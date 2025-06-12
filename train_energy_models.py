import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(data_path):
    """Load and preprocess the dataset"""
    print("Loading and preprocessing data...")
    df = pd.read_csv(data_path)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Create day of week feature
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    # Handle missing values if any
    if df.isnull().sum().any():
        print("Handling missing values...")
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Check if we have consumption breakdown data
    if not all(col in df.columns for col in ['Residential_Consumption', 'Commercial_Consumption', 'Industrial_Consumption']):
        print("Creating consumption breakdown based on region and hour patterns...")
        # Create synthetic breakdown based on time of day and region
        df = create_consumption_breakdown(df)
    
    return df

def create_consumption_breakdown(df):
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
    df['Residential_Consumption'] = df.apply(
        lambda row: row['Total_Consumption'] * residential_pattern[int(row['Hour'])], axis=1
    )
    df['Commercial_Consumption'] = df.apply(
        lambda row: row['Total_Consumption'] * commercial_pattern[int(row['Hour'])], axis=1
    )
    df['Industrial_Consumption'] = df.apply(
        lambda row: row['Total_Consumption'] * industrial_pattern[int(row['Hour'])], axis=1
    )
    
    # Add some region-based variation
    region_factors = df.groupby('Region')['Total_Consumption'].mean()
    region_factors = region_factors / region_factors.mean()
    
    for region, factor in region_factors.items():
        mask = df['Region'] == region
        if factor > 1.1:  # High consumption regions get more industrial
            df.loc[mask, 'Industrial_Consumption'] *= 1.2
            df.loc[mask, 'Commercial_Consumption'] *= 0.9
            df.loc[mask, 'Residential_Consumption'] *= 0.9
        elif factor < 0.9:  # Low consumption regions get more residential
            df.loc[mask, 'Industrial_Consumption'] *= 0.8
            df.loc[mask, 'Commercial_Consumption'] *= 0.9
            df.loc[mask, 'Residential_Consumption'] *= 1.3
    
    # Ensure the sum matches Total_Consumption
    for idx in df.index:
        total = (df.loc[idx, 'Residential_Consumption'] + 
                df.loc[idx, 'Commercial_Consumption'] + 
                df.loc[idx, 'Industrial_Consumption'])
        
        factor = df.loc[idx, 'Total_Consumption'] / total
        df.loc[idx, 'Residential_Consumption'] *= factor
        df.loc[idx, 'Commercial_Consumption'] *= factor
        df.loc[idx, 'Industrial_Consumption'] *= factor
    
    return df

def prepare_features(df):
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

def train_models(df):
    """Train prediction models for various energy metrics"""
    print("Training prediction models...")
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Prepare features
    df_features = prepare_features(df)
    
    # Define target columns to predict
    target_columns = [
        'Solar_Generation', 'Wind_Generation', 'Hydro_Generation',
        'Total_Consumption', 'Residential_Consumption', 
        'Commercial_Consumption', 'Industrial_Consumption'
    ]
    
    # Features to use (exclude targets and date)
    feature_cols = [col for col in df_features.columns if col not in target_columns + ['Date']]
    
    # Save feature columns for later use
    joblib.dump(feature_cols, 'models/feature_cols.pkl')
    
    # Save unique regions and weather conditions for later use
    regions = df['Region'].unique().tolist()
    weather_conditions = df['Weather'].unique().tolist()
    joblib.dump({'regions': regions, 'weather_conditions': weather_conditions}, 'models/categorical_values.pkl')
    
    # Initialize scaler
    scaler = StandardScaler()
    X = df_features[feature_cols].values
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    
    # Save the scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Save historical averages
    historical_avg = {
        'Total_Consumption': df['Total_Consumption'].mean(),
        'Residential_Consumption': df['Residential_Consumption'].mean(),
        'Commercial_Consumption': df['Commercial_Consumption'].mean(),
        'Industrial_Consumption': df['Industrial_Consumption'].mean(),
        'Solar_Generation': df['Solar_Generation'].mean(),
        'Wind_Generation': df['Wind_Generation'].mean(),
        'Hydro_Generation': df['Hydro_Generation'].mean() if 'Hydro_Generation' in df.columns else 0
    }
    joblib.dump(historical_avg, 'models/historical_avg.pkl')
    
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
            
            # Save model
            joblib.dump(model, f'models/{target}_model.pkl')
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            print(f"  - {target} model trained: Train R² = {train_score:.4f}, Test R² = {test_score:.4f}")
    
    print("\nAll models trained and saved to 'models' directory.")
    return feature_cols, scaler

def main():
    print("Energy Grid Management System - Model Training")
    print("=============================================")
    
    # Load and preprocess data
    df = load_and_preprocess_data('karnataka_energy_data_2024_final.csv')
    
    # Train models
    feature_cols, scaler = train_models(df)
    
    print("\nTraining complete. Models are ready for use in the dashboard application.")

if __name__ == "__main__":
    main() 