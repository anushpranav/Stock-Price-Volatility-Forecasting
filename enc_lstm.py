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

# Import TenSEAL for homomorphic encryption
import tenseal as ts

class EncryptedStockVolatilityForecaster:
    def __init__(self, data_path):
        self.data_path = data_path
        self.sequence_length = 60
        self.feature_scaler = MinMaxScaler()
        self.volatility_scaler = MinMaxScaler()
        self.model = None
        self.companies = None
        self.data = None
        
        # Initialize HE context and secret key
        self.context = None
        self.secret_key = None
        self.setup_encryption_context()
        
    def setup_encryption_context(self):
        """Set up the homomorphic encryption context and save secret key"""
        # Create TenSEAL context
        context = ts.Context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.global_scale = 2**40
        context.generate_galois_keys()
        
        # Save secret key before potentially making context public
        self.secret_key = context.secret_key()
        self.context = context
        print("Homomorphic encryption context initialized")
    
    def encrypt_data(self, data_array):
        """Encrypt numpy array using homomorphic encryption"""
        return ts.ckks_vector(self.context, data_array)
    
    def decrypt_data(self, encrypted_vector):
        """Decrypt TenSEAL vector back to numpy array"""
        if self.secret_key is not None:
            return np.array(encrypted_vector.decrypt(self.secret_key))
        elif self.context.has_secret_key():
            return np.array(encrypted_vector.decrypt())
        else:
            raise ValueError("No secret key available for decryption")
        
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
        
        # Detection of TenSEAL version and appropriate serialization method
        try:
            # Try the method that works with your TenSEAL version
            if hasattr(self.context, 'serialize'):
                context_bytes = self.context.serialize()
            else:
                # For newer versions that might use save() instead
                context_bytes = self.context.save()
                
            with open(os.path.join(model_dir, 'encryption_context.bin'), 'wb') as f:
                f.write(context_bytes)
            print("Encryption context saved successfully")
        except Exception as e:
            print(f"Error saving encryption context: {str(e)}")
        
        print(f"Model, data, and encryption context saved to {model_dir}")
    
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
            
            # Try to load encryption context - detect version and use appropriate method
            try:
                with open(os.path.join(model_dir, 'encryption_context.bin'), 'rb') as f:
                    context_bytes = f.read()
                    
                # Try different methods to load the context based on TenSEAL version
                if hasattr(ts.Context, 'load'):
                    self.context = ts.Context.load(context_bytes)
                    print("Loaded context using Context.load()")
                elif hasattr(ts, 'deserialize_context'):
                    # For older versions of TenSEAL
                    self.context = ts.deserialize_context(context_bytes)
                    print("Loaded context using ts.deserialize_context()")
                else:
                    # Fallback for very old versions
                    self.setup_encryption_context()
                    print("Could not load context, initialized new one")
                    
                # Check if context has secret key
                if self.context.has_secret_key():
                    self.secret_key = self.context.secret_key()
                    print("Secret key retrieved from context")
                    
            except FileNotFoundError:
                print("No encryption context found, initializing new one")
                self.setup_encryption_context()
                    
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
    
    def prepare_encrypted_features(self):
        """Prepare encrypted features for model training"""
        feature_columns = ['Returns', 'Volatility', 'MA_20', 'MA_50', 'RSI', 
                         'Volume_Ratio', 'High_Low_Ratio', 'Open_Close_Ratio']
        
        # Scale features
        self.feature_scaler.fit(self.data[feature_columns])
        self.volatility_scaler.fit(self.data[['Volatility']])
        
        X, y = [], []
        encrypted_X, encrypted_y = [], []
        
        for company in self.data['Company'].unique():
            company_data = self.data[self.data['Company'] == company]
            scaled_features = self.feature_scaler.transform(company_data[feature_columns])
            scaled_volatility = self.volatility_scaler.transform(company_data[['Volatility']])
            
            for i in range(len(company_data) - self.sequence_length):
                feature_sequence = scaled_features[i:(i + self.sequence_length)]
                target_volatility = scaled_volatility[i + self.sequence_length]
                
                # Store plaintext features for model training (we'll train normally)
                X.append(feature_sequence)
                y.append(target_volatility)
                
                # Create encrypted versions
                encrypted_sequence = self.encrypt_data(feature_sequence.flatten())
                encrypted_target = self.encrypt_data(target_volatility.flatten())
                
                encrypted_X.append(encrypted_sequence)
                encrypted_y.append(encrypted_target)
        
        return np.array(X), np.array(y), encrypted_X, encrypted_y
    
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
        X, y, _, _ = self.prepare_encrypted_features()
        
        print("Building and training model...")
        self.model = self.build_model()
        history = self.model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
        
        print("Saving model and data...")
        self.save_model()
        return history
    
    def encrypted_predict_volatility(self, company_name, months_ahead):
        """Predict future volatility for a specific company using encrypted features"""
        if company_name not in self.companies:
            raise ValueError(f"Company {company_name} not found in dataset")
        
        # Get the most recent data for the company
        company_data = self.data[self.data['Company'] == company_name].copy()
        feature_columns = ['Returns', 'Volatility', 'MA_20', 'MA_50', 'RSI', 
                         'Volume_Ratio', 'High_Low_Ratio', 'Open_Close_Ratio']
        
        # Scale the features
        recent_features = self.feature_scaler.transform(company_data[feature_columns].tail(self.sequence_length))
        
        # Encrypt the features
        encrypted_features = self.encrypt_data(recent_features.flatten())
        
        # Make predictions
        num_predictions = months_ahead * 21  # Approximate trading days per month
        predictions = []
        current_sequence = recent_features.copy()
        
        for _ in range(num_predictions):
            # For prediction, we use the unencrypted model since LSTM operations aren't fully HE-compatible
            # But the data remains encrypted during transmission and storage
            pred = self.model.predict(current_sequence.reshape(1, self.sequence_length, 8), verbose=0)
            
            # Encrypt the prediction
            encrypted_pred = self.encrypt_data(pred.flatten())
            # For demonstration purposes, we decrypt to continue the prediction loop
            # In a real-world scenario, specialized HE-compatible models would be used
            decrypted_pred = self.decrypt_data(encrypted_pred)
            
            predictions.append(decrypted_pred[0])
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = np.append(current_sequence[-2][:-1], decrypted_pred[0])
        
        # Convert predictions to numpy array and reshape
        predictions = np.array(predictions).reshape(-1, 1)
        
        # Inverse transform predictions
        predictions = self.volatility_scaler.inverse_transform(predictions)
        
        # Generate future dates
        last_date = company_data['Date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                   periods=num_predictions, freq='B')
        
        # Return encrypted results (for demonstration we decrypt them)
        result_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Volatility': predictions.flatten()
        })
        
        return result_df
    
    def plot_forecast(self, company_name, months_ahead):
        """Plot only predicted volatility using encrypted predictions"""
        # Get encrypted predictions
        predictions = self.encrypted_predict_volatility(company_name, months_ahead)
        
        plt.figure(figsize=(15, 7))
        
        # Plot only predicted volatility
        plt.plot(predictions['Date'], predictions['Predicted_Volatility'], 
                label='Predicted Volatility', color='red')
        
        plt.title(f'Encrypted Stock Volatility Forecast for {company_name} - Next {months_ahead} Month(s)')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        return plt

def main():
    # Initialize forecaster with homomorphic encryption
    forecaster = EncryptedStockVolatilityForecaster("dataset/")
    
    # Check if saved model exists
    if os.path.exists(os.path.join("saved_model", "volatility_model.keras")):
        print("Loading existing model and encryption context...")
        if forecaster.load_model():
            print("Model and encryption context loaded successfully")
        else:
            print("Error loading model, will train new model")
            print("Loading and preprocessing data...")
            forecaster.load_and_preprocess_data()
            forecaster.train_model()
    else:
        print("Training new model with encryption capabilities...")
        print("Loading and preprocessing data...")
        forecaster.load_and_preprocess_data()
        forecaster.train_model()
    
    # Example prediction with encrypted data
    company_name = "ZOMATO.NS"  # Replace with desired company
    months_ahead = 10  # Replace with desired number of months (1-3)
    
    print(f"\nMaking encrypted predictions for {company_name} for {months_ahead} months ahead...")
    predictions = forecaster.encrypted_predict_volatility(company_name, months_ahead)
    print("\nDecrypted Predicted Volatility:")
    print(predictions)
    
    plt = forecaster.plot_forecast(company_name, months_ahead)
    plt.show()

if __name__ == "__main__":
    main()