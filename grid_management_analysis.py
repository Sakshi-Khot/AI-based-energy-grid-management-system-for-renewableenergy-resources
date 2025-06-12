import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset
    """
    print("Loading and preprocessing data...")
    df = pd.read_csv(file_path)
    
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
    
    return df

def perform_eda(df):
    """
    Perform Exploratory Data Analysis
    """
    print("\nPerforming Exploratory Data Analysis...")
    
    # Display basic information
    print("\nDataset Info:")
    print(df.info())
    
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Create visualizations
    plt.figure(figsize=(20, 15))
    
    # Plot correlation heatmap
    plt.subplot(2, 2, 1)
    corr = df.select_dtypes(include=['float64', 'int64']).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    
    # Plot renewable energy generation by region
    plt.subplot(2, 2, 2)
    renewable_by_region = df.groupby('Region')[['Solar_Generation', 'Wind_Generation', 'Hydro_Generation']].mean()
    renewable_by_region.plot(kind='bar', ax=plt.gca())
    plt.title('Average Renewable Energy Generation by Region')
    plt.ylabel('Energy (MW)')
    plt.xticks(rotation=45)
    
    # Plot total consumption vs time of day
    plt.subplot(2, 2, 3)
    hour_consumption = df.groupby('Hour')['Total_Consumption'].mean()
    hour_consumption.plot(kind='line', marker='o')
    plt.title('Average Energy Consumption by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Energy Consumption (MW)')
    plt.grid(True)
    
    # Plot renewable vs grid supply
    plt.subplot(2, 2, 4)
    df['Renewable_Generation'] = df['Solar_Generation'] + df['Wind_Generation'] + df['Hydro_Generation']
    df['Supply_Type'] = pd.cut(
        df['Renewable_Generation'] / (df['Renewable_Generation'] + df['Grid_Supply']),
        bins=[0, 0.25, 0.5, 0.75, 1],
        labels=['0-25%', '25-50%', '50-75%', '75-100%']
    )
    supply_counts = df['Supply_Type'].value_counts()
    supply_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Renewable Energy Contribution to Total Supply')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('eda_plots.png')
    plt.close()
    
    # Additional renewable energy specific plots
    plt.figure(figsize=(20, 10))
    
    # Plot solar generation by hour
    plt.subplot(1, 3, 1)
    solar_by_hour = df.groupby('Hour')['Solar_Generation'].mean()
    solar_by_hour.plot(kind='line', marker='o', color='orange')
    plt.title('Average Solar Generation by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Solar Generation (MW)')
    plt.grid(True)
    
    # Plot wind generation by hour
    plt.subplot(1, 3, 2)
    wind_by_hour = df.groupby('Hour')['Wind_Generation'].mean()
    wind_by_hour.plot(kind='line', marker='o', color='blue')
    plt.title('Average Wind Generation by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Wind Generation (MW)')
    plt.grid(True)
    
    # Plot weather impact on renewable generation
    plt.subplot(1, 3, 3)
    weather_impact = df.groupby('Weather')[['Solar_Generation', 'Wind_Generation']].mean()
    weather_impact.plot(kind='bar')
    plt.title('Weather Impact on Renewable Generation')
    plt.xlabel('Weather Condition')
    plt.ylabel('Average Generation (MW)')
    plt.xticks(rotation=45)
    
    # Save the renewable energy plots
    plt.tight_layout()
    plt.savefig('renewable_energy_plots.png')
    plt.close()

def prepare_data_for_training(df, target_column):
    """
    Prepare data for model training
    """
    print("\nPreparing data for training...")
    
    # Create feature engineering for time-based patterns
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour']/24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour']/24)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month']/12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month']/12)
    
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['Region', 'Weather'], drop_first=True)
    
    # Drop non-feature columns
    df = df.drop(columns=['Date'])
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Testing set shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns

def train_model(X_train, y_train):
    """
    Train the model
    """
    print("\nTraining the model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    """
    Evaluate the model performance
    """
    print("\nEvaluating model performance...")
    
    # Training performance
    y_train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Testing performance
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("Training Performance:")
    print(f"  Mean Squared Error: {train_mse:.4f}")
    print(f"  Mean Absolute Error: {train_mae:.4f}")
    print(f"  R2 Score: {train_r2:.4f}")
    
    print("\nTesting Performance:")
    print(f"  Mean Squared Error: {test_mse:.4f}")
    print(f"  Mean Absolute Error: {test_mae:.4f}")
    print(f"  R2 Score: {test_r2:.4f}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(12, 10))
    
    # Training data
    plt.subplot(2, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Training: Actual vs Predicted Values')
    
    # Testing data
    plt.subplot(2, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Testing: Actual vs Predicted Values')
    
    # Residuals
    plt.subplot(2, 2, 3)
    residuals = y_test - y_test_pred
    plt.scatter(y_test_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    # Feature importance
    plt.subplot(2, 2, 4)
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]  # Top 10 features
    plt.barh(range(10), importances[indices])
    plt.yticks(range(10), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Important Features')
    
    # Save the evaluation plots
    plt.tight_layout()
    plt.savefig('model_evaluation.png')
    plt.close()

def main():
    # Load and preprocess data
    df = load_and_preprocess_data('karnataka_energy_data_2024_final.csv')
    
    # Perform EDA
    perform_eda(df)
    
    # Set target column for energy consumption prediction
    target_column = 'Total_Consumption'
    
    # Prepare data for training
    X_train, X_test, y_train, y_test, feature_names = prepare_data_for_training(df, target_column)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_train, X_test, y_train, y_test, feature_names)
    
    print("\nAnalysis complete! Check the generated plots for visualizations.")

if __name__ == "__main__":
    main() 