import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class EnergyConsumptionPredictor:
    def __init__(self, data_path=None):
        """Initialize the energy consumption predictor"""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'Total_Consumption'
        
        if data_path and os.path.exists(data_path):
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """Load and preprocess the energy consumption data"""
        print(f"Loading data from {data_path}...")
        self.data = pd.read_csv(data_path)
        
        # Convert date column to datetime if it exists
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            
            # Extract time-based features
            self.data['Hour'] = self.data['Date'].dt.hour
            self.data['DayOfWeek'] = self.data['Date'].dt.dayofweek
            self.data['Month'] = self.data['Date'].dt.month
            
            # Create cyclical features
            self.data['Hour_Sin'] = np.sin(2 * np.pi * self.data['Hour']/24)
            self.data['Hour_Cos'] = np.cos(2 * np.pi * self.data['Hour']/24)
            self.data['Month_Sin'] = np.sin(2 * np.pi * self.data['Month']/12)
            self.data['Month_Cos'] = np.cos(2 * np.pi * self.data['Month']/12)
    
    def preprocess_data(self):
        """Preprocess the data for model training"""
        print("Preprocessing data...")
        
        # Select features and target
        feature_cols = [col for col in self.data.columns 
                       if col not in ['Date', self.target_column]]
        self.feature_columns = feature_cols
        
        X = self.data[feature_cols]
        y = self.data[self.target_column]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
    
    def train_model(self):
        """Train the energy consumption prediction model"""
        print("Training model...")
        
        # Initialize and train Random Forest model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate model
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        
        train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
        train_r2 = r2_score(self.y_train, train_pred)
        test_r2 = r2_score(self.y_test, test_pred)
        
        print(f"\nModel Performance:")
        print(f"Training RMSE: {train_rmse:.2f}")
        print(f"Testing RMSE: {test_rmse:.2f}")
        print(f"Training R²: {train_r2:.2f}")
        print(f"Testing R²: {test_r2:.2f}")
    
    def save_model(self, model_dir='models'):
        """Save the trained model and scaler"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Save model
        joblib.dump(self.model, f'{model_dir}/energy_consumption_model.pkl')
        # Save scaler
        joblib.dump(self.scaler, f'{model_dir}/scaler.pkl')
        # Save feature columns
        joblib.dump(self.feature_columns, f'{model_dir}/feature_columns.pkl')
        
        print(f"Model saved in {model_dir}/")
    
    def load_model(self, model_dir='models'):
        """Load a trained model"""
        model_path = f'{model_dir}/energy_consumption_model.pkl'
        scaler_path = f'{model_dir}/scaler.pkl'
        features_path = f'{model_dir}/feature_columns.pkl'
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
            raise FileNotFoundError("Model files not found. Please train the model first.")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_columns = joblib.load(features_path)
        
        print("Model loaded successfully")
    
    def predict(self, input_data):
        """Make predictions for new data"""
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")
        
        # Ensure input data has all required features
        if isinstance(input_data, pd.DataFrame):
            input_data = input_data[self.feature_columns]
        else:
            input_data = pd.DataFrame(input_data, columns=self.feature_columns)
        
        # Scale features
        input_scaled = self.scaler.transform(input_data)
        
        # Make predictions
        predictions = self.model.predict(input_scaled)
        
        return predictions
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")
        
        # Get feature importance
        importance = self.model.feature_importances_
        features = self.feature_columns
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance for Energy Consumption Prediction')
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, y_true, y_pred, title='Actual vs Predicted Energy Consumption'):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Energy Consumption')
        plt.ylabel('Predicted Energy Consumption')
        plt.title(title)
        plt.tight_layout()
        plt.show()

def main():
    # Example usage
    predictor = EnergyConsumptionPredictor()
    
    # Load data
    data_path = 'karnataka_energy_data_2024_final.csv'
    if os.path.exists(data_path):
        predictor.load_data(data_path)
        
        # Preprocess data
        predictor.preprocess_data()
        
        # Train model
        predictor.train_model()
        
        # Save model
        predictor.save_model()
        
        # Plot feature importance
        predictor.plot_feature_importance()
        
        # Plot predictions on test set
        test_pred = predictor.model.predict(predictor.X_test)
        predictor.plot_predictions(predictor.y_test, test_pred)
    else:
        print(f"Data file {data_path} not found.")

if __name__ == "__main__":
    main() 