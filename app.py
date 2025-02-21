import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import glob
import os
from datetime import timedelta
import matplotlib.pyplot as plt
import pickle
import joblib

class StockVolatilityForecaster:
    def __init__(self, data_path):
        self.data_path = data_path
        self.sequence_length = 60
        self.feature_scaler = MinMaxScaler()
        self.volatility_scaler = MinMaxScaler()
        self.model = None
        self.companies = None
        self.data = None
        
    def save_model(self, model_dir="saved_model"):
        """Save the trained model and scalers"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Save the Keras model with .keras extension
        model_path = os.path.join(model_dir, 'volatility_model.keras')
        self.model.save(model_path)
        
        # Save the scalers and data
        joblib.dump(self.feature_scaler, os.path.join(model_dir, 'feature_scaler.pkl'))
        joblib.dump(self.volatility_scaler, os.path.join(model_dir, 'volatility_scaler.pkl'))
        
        # Save the companies list and preprocessed data
        with open(os.path.join(model_dir, 'companies.pkl'), 'wb') as f:
            pickle.dump(self.companies, f)
        
        # Save preprocessed data
        self.data.to_pickle(os.path.join(model_dir, 'preprocessed_data.pkl'))
            
        print(f"Model and data saved to {model_dir}")
    
    def load_model(self, model_dir="saved_model"):
        """Load the trained model and scalers"""
        try:
            # Load the Keras model
            model_path = os.path.join(model_dir, 'volatility_model.keras')
            self.model = tf.keras.models.load_model(model_path)
            
            # Load the scalers
            self.feature_scaler = joblib.load(os.path.join(model_dir, 'feature_scaler.pkl'))
            self.volatility_scaler = joblib.load(os.path.join(model_dir, 'volatility_scaler.pkl'))
            
            # Load the companies list and preprocessed data
            with open(os.path.join(model_dir, 'companies.pkl'), 'rb') as f:
                self.companies = pickle.load(f)
            
            # Load preprocessed data
            self.data = pd.read_pickle(os.path.join(model_dir, 'preprocessed_data.pkl'))
                
            print("Model, scalers, and data loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def load_and_preprocess_data(self):
        """Load and preprocess all company data"""
        all_data = []
        
        # Get all CSV files
        csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
        self.companies = [os.path.basename(file).split('_')[0] for file in csv_files]
        
        for file in csv_files:
            company_name = os.path.basename(file).split('_')[0]
            df = pd.read_csv(file)
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
            df['Company'] = company_name
            
            # Calculate features
            df['Returns'] = df['Close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)  # Annualized volatility
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            df['MA_50'] = df['Close'].rolling(window=50).mean()
            df['RSI'] = self.calculate_rsi(df['Close'])
            df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
            df['High_Low_Ratio'] = df['High'] / df['Low']
            df['Open_Close_Ratio'] = df['Open'] / df['Close']
            
            all_data.append(df.dropna())
        
        self.data = pd.concat(all_data, ignore_index=True)
        self.data = self.data.sort_values(['Company', 'Date'])
        
        return self.data
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def prepare_features(self):
        """Prepare features for model training"""
        feature_columns = ['Returns', 'Volatility', 'MA_20', 'MA_50', 'RSI', 
                         'Volume_Ratio', 'High_Low_Ratio', 'Open_Close_Ratio']
        
        # Scale features
        self.feature_scaler.fit(self.data[feature_columns])
        self.volatility_scaler.fit(self.data[['Volatility']])
        
        X, y = [], []
        
        for company in self.data['Company'].unique():
            company_data = self.data[self.data['Company'] == company]
            scaled_features = self.feature_scaler.transform(company_data[feature_columns])
            scaled_volatility = self.volatility_scaler.transform(company_data[['Volatility']])
            
            for i in range(len(company_data) - self.sequence_length):
                X.append(scaled_features[i:(i + self.sequence_length)])
                y.append(scaled_volatility[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def build_model(self):
        """Create and compile the LSTM model"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(100, input_shape=(self.sequence_length, 8), return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(30, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train_model(self):
        """Train the model and save it"""
        print("Preparing features for training...")
        X, y = self.prepare_features()
        
        print("Building and training model...")
        self.model = self.build_model()
        history = self.model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
        
        print("Saving model and data...")
        self.save_model()
        return history
    
    def predict_volatility(self, company_name, months_ahead):
        """Predict future volatility for a specific company"""
        if company_name not in self.companies:
            raise ValueError(f"Company {company_name} not found in dataset")
        
        # Get the most recent data for the company
        company_data = self.data[self.data['Company'] == company_name].copy()
        feature_columns = ['Returns', 'Volatility', 'MA_20', 'MA_50', 'RSI', 
                         'Volume_Ratio', 'High_Low_Ratio', 'Open_Close_Ratio']
        
        # Scale the features
        recent_features = self.feature_scaler.transform(company_data[feature_columns].tail(self.sequence_length))
        
        # Make predictions
        num_predictions = months_ahead * 21  # Approximate trading days per month
        predictions = []
        current_sequence = recent_features.copy()
        
        for _ in range(num_predictions):
            # Predict next day
            pred = self.model.predict(current_sequence.reshape(1, self.sequence_length, 8), verbose=0)
            predictions.append(pred[0][0])
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = np.append(current_sequence[-2][:-1], pred[0][0])
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.volatility_scaler.inverse_transform(predictions)
        
        # Generate future dates
        last_date = company_data['Date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                   periods=num_predictions, freq='B')
        
        return pd.DataFrame({
            'Date': future_dates,
            'Predicted_Volatility': predictions.flatten()
        })
    
    def plot_forecast(self, company_name, months_ahead):
        """Plot historical and predicted volatility"""
        # Get historical data
        historical_data = self.data[self.data['Company'] == company_name].copy()
        
        # Get predictions
        predictions = self.predict_volatility(company_name, months_ahead)
        
        plt.figure(figsize=(15, 7))
        
        # Plot historical volatility
        plt.plot(historical_data['Date'], historical_data['Volatility'], 
                label='Historical Volatility', color='blue')
        
        # Plot predicted volatility
        plt.plot(predictions['Date'], predictions['Predicted_Volatility'], 
                label='Predicted Volatility', color='red', linestyle='--')
        
        plt.title(f'Stock Volatility Forecast for {company_name}')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        return plt

def main():
    # Initialize forecaster
    forecaster = StockVolatilityForecaster("dataset/")
    
    # Check if saved model exists
    if os.path.exists(os.path.join("saved_model", "volatility_model.keras")):
        print("Loading existing model...")
        if forecaster.load_model():
            print("Model loaded successfully")
        else:
            print("Error loading model, will train new model")
            print("Loading and preprocessing data...")
            forecaster.load_and_preprocess_data()
            forecaster.train_model()
    else:
        print("Training new model...")
        print("Loading and preprocessing data...")
        forecaster.load_and_preprocess_data()
        forecaster.train_model()
    
    # Example prediction
    company_name = "JIOFIN.NS"  # Replace with desired company
    months_ahead = 2  # Replace with desired number of months (1-3)
    
    print(f"\nMaking predictions for {company_name} for {months_ahead} months ahead...")
    predictions = forecaster.predict_volatility(company_name, months_ahead)
    print("\nPredicted Volatility:")
    print(predictions)
    
    plt = forecaster.plot_forecast(company_name, months_ahead)
    plt.show()

if __name__ == "__main__":
    main()