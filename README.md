# Energy Consumption Prediction System

This project implements a machine learning-based system for predicting energy consumption patterns. It uses historical energy data to train a Random Forest model that can predict future energy consumption based on various features.

## Features

- Data preprocessing and feature engineering
- Time-based feature extraction (hour, day of week, month)
- Cyclical feature encoding for time variables
- Random Forest model for energy consumption prediction
- Model performance evaluation (RMSE and R² metrics)
- Feature importance visualization
- Actual vs. Predicted values plotting
- Model persistence (save/load functionality)

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd energy-consumption-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data:
   - The system expects a CSV file with energy consumption data
   - Required columns: Date, Total_Consumption, and other relevant features
   - Date column should be in a format that pandas can parse

2. Run the prediction system:
```python
from energy_consumption_predictor import EnergyConsumptionPredictor

# Initialize predictor
predictor = EnergyConsumptionPredictor()

# Load and preprocess data
predictor.load_data('your_data.csv')
predictor.preprocess_data()

# Train model
predictor.train_model()

# Save model
predictor.save_model()

# Make predictions
predictions = predictor.predict(new_data)

# Visualize results
predictor.plot_feature_importance()
predictor.plot_predictions(actual_values, predictions)
```

## Model Details

The system uses a Random Forest Regressor with the following default parameters:
- n_estimators: 100
- max_depth: 10
- random_state: 42

Features are preprocessed using StandardScaler for normalization.

## Output

The system provides:
1. Model performance metrics (RMSE and R²)
2. Feature importance visualization
3. Actual vs. Predicted values plot
4. Saved model files in the 'models' directory:
   - energy_consumption_model.pkl
   - scaler.pkl
   - feature_columns.pkl

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 