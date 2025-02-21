import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
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
        # Use a more conservative scaling range
        self.feature_scaler = MinMaxScaler(feature_range=(-0.1, 0.1))
        self.volatility_scaler = MinMaxScaler(feature_range=(-0.1, 0.1))
        self.model = None
        self.companies = None
        self.data = None
        self.context = None
        self.secret_key = None
        self.setup_encryption_context()
        
    def setup_encryption_context(self):
        """Set up the homomorphic encryption context with more conservative parameters"""
        context = ts.Context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]  # Increased bit sizes for better precision
        )
        # Use a more conservative scale
        context.global_scale = 2**40
        context.generate_galois_keys()
        self.secret_key = context.secret_key()
        self.context = context
        print("Homomorphic encryption context initialized with enhanced parameters")
    
    def encrypt_data(self, data_array):
        """Encrypt numpy array with improved error handling"""
        try:
            # Ensure data is a numpy array
            data_array = np.array(data_array, dtype=np.float64)
            
            # Apply stricter value bounds
            max_allowed = 0.1
            if np.any(np.abs(data_array) > max_allowed):
                print(f"Scaling down values exceeding ±{max_allowed}")
                data_array = np.clip(data_array, -max_allowed, max_allowed)
            
            # Check for NaN or infinite values
            if np.any(np.isnan(data_array)) or np.any(np.isinf(data_array)):
                print("Warning: Invalid values detected, replacing with zeros")
                data_array = np.nan_to_num(data_array, nan=0.0, posinf=max_allowed, neginf=-max_allowed)
            
            return ts.ckks_vector(self.context, data_array)
        except Exception as e:
            print(f"Encryption error: {str(e)}")
            # Return a vector of zeros as fallback
            return ts.ckks_vector(self.context, np.zeros_like(data_array))
    
    def decrypt_data(self, encrypted_vector):
        """Decrypt TenSEAL vector back to numpy array"""
        if self.secret_key is not None:
            return np.array(encrypted_vector.decrypt(self.secret_key))
        elif self.context.has_secret_key():
            return np.array(encrypted_vector.decrypt())
        else:
            raise ValueError("No secret key available for decryption")
        
    def save_model(self, model_dir="saved_model_ridge"):
        """Save the trained model and scalers"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Save the Ridge model
        joblib.dump(self.model, os.path.join(model_dir, 'volatility_model.pkl'))
        
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
    
    def load_model(self, model_dir="saved_model_ridge"):
        """Load the trained model and scalers"""
        try:
            # Load the Ridge model
            self.model = joblib.load(os.path.join(model_dir, 'volatility_model.pkl'))
            
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
                # For Ridge Regression, we flatten the sequence into a single feature vector
                feature_sequence = scaled_features[i:(i + self.sequence_length)].flatten()
                target_volatility = scaled_volatility[i + self.sequence_length]
                
                # Store plaintext features for model training
                X.append(feature_sequence)
                y.append(target_volatility)
                
                # Create encrypted versions
                encrypted_sequence = self.encrypt_data(feature_sequence)
                encrypted_target = self.encrypt_data(target_volatility.flatten())
                
                encrypted_X.append(encrypted_sequence)
                encrypted_y.append(encrypted_target)
        
        return np.array(X), np.array(y).reshape(-1, 1), encrypted_X, encrypted_y
    
    def build_model(self):
        """Create a Ridge Regression model"""
        # Ridge Regression works well with homomorphic encryption as it's a linear model
        # Alpha parameter controls regularization strength
        return Ridge(alpha=0.5)
    
    def train_model(self):
        """Train the model and save it"""
        print("Preparing features for training...")
        X, y, _, _ = self.prepare_encrypted_features()
        
        print("Building and training Ridge Regression model...")
        self.model = self.build_model()
        self.model.fit(X, y)
        
        # Normalize model coefficients to help with encryption stability
        coef_norm = np.linalg.norm(self.model.coef_[0])
        if coef_norm > 1:
            print(f"Normalizing model coefficients (norm: {coef_norm:.4f})")
            self.model.coef_ = self.model.coef_ / coef_norm
            self.model.intercept_ = self.model.intercept_ / coef_norm
        
        # Calculate training score
        score = self.model.score(X, y)
        print(f"Model R² score on training data: {score:.4f}")
        
        print("Saving model and data...")
        self.save_model()
        return self.model
    
    def encrypted_predict_volatility(self, company_name, months_ahead):
        """Predict future volatility with improved numerical stability"""
        if company_name not in self.companies:
            raise ValueError(f"Company {company_name} not found in dataset")
        
        company_data = self.data[self.data['Company'] == company_name].copy()
        feature_columns = ['Returns', 'Volatility', 'MA_20', 'MA_50', 'RSI', 
                         'Volume_Ratio', 'High_Low_Ratio', 'Open_Close_Ratio']
        
        # Scale features with additional safety checks
        recent_features = self.feature_scaler.transform(company_data[feature_columns].tail(self.sequence_length))
        recent_features = np.clip(recent_features, -0.1, 0.1)  # Additional clipping for safety
        
        flattened_features = recent_features.flatten()
        
        # Make predictions with error handling
        num_predictions = months_ahead * 21
        predictions = []
        current_sequence = recent_features.copy()
        
        for _ in range(num_predictions):
            try:
                current_flat = current_sequence.flatten()
                current_flat = np.clip(current_flat, -0.1, 0.1)  # Safety clipping
                
                coef = self.model.coef_[0]
                intercept = self.model.intercept_[0]
                
                # Scale down coefficients if they're too large
                if np.any(np.abs(coef) > 0.1):
                    scale_factor = 0.1 / np.max(np.abs(coef))
                    coef = coef * scale_factor
                    intercept = intercept * scale_factor
                
                # Encrypted prediction with careful scaling
                encrypted_flat = self.encrypt_data(current_flat)
                pred_value = 0.0
                
                # Process in smaller batches to maintain stability
                batch_size = 10
                for i in range(0, len(coef), batch_size):
                    batch_coef = coef[i:i+batch_size]
                    batch_features = current_flat[i:i+batch_size]
                    batch_encrypted = self.encrypt_data(batch_features)
                    
                    for j, c in enumerate(batch_coef):
                        weighted_input = batch_encrypted * c
                        pred_value += self.decrypt_data(weighted_input)[j]
                
                pred_value += intercept
                pred_value = np.clip(pred_value, -0.1, 0.1)  # Final safety clip
                
            except Exception as e:
                print(f"Prediction error, using fallback: {str(e)}")
                # Fallback to simple moving average
                pred_value = np.mean(current_flat)
            
            predictions.append(pred_value)
            
            # Update sequence carefully
            current_sequence = np.roll(current_sequence, -1, axis=0)
            new_features = np.zeros(8)
            new_features[1] = pred_value
            current_sequence[-1] = new_features
        
        # Process predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.volatility_scaler.inverse_transform(predictions)
        
        # Generate dates
        last_date = company_data['Date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                   periods=num_predictions, freq='B')
        
        return pd.DataFrame({
            'Date': future_dates,
            'Predicted_Volatility': predictions.flatten()
        })
    
    def plot_forecast(self, company_name, months_ahead):
        """Plot predicted volatility"""
        # Get predictions
        predictions = self.encrypted_predict_volatility(company_name, months_ahead)
        
        plt.figure(figsize=(15, 7))
        
        # Plot predicted volatility
        plt.plot(predictions['Date'], predictions['Predicted_Volatility'], 
                label='Predicted Volatility', color='red')
        
        plt.title(f'Encrypted Stock Volatility Forecast for {company_name} - Next {months_ahead} Month(s)')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        return plt
    
    def display_model_metrics(self):
        """Display model accuracy and classification report"""
        # Get training data
        X, y, encrypted_X, encrypted_y = self.prepare_encrypted_features()
        
        # Get predictions on training data
        y_pred = self.model.predict(X)
        
        # Convert continuous predictions to binary classes for classification metrics
        # Using median as threshold for binary classification
        threshold = np.median(y)
        y_binary = (y > threshold).astype(int)
        y_pred_binary = (y_pred > threshold).astype(int)
        
        # Calculate accuracy and classification report
        from sklearn.metrics import accuracy_score, classification_report
        
        accuracy = accuracy_score(y_binary, y_pred_binary)
        report = classification_report(y_binary, y_pred_binary)
        
        print(f"Accuracy: {accuracy:.4f}\n")
        print("Classification Report:")
        print(report)
        
        # Display first 5 encrypted vectors
        print("\nFirst 5 encrypted vectors:")
        for i in range(5):
            print(f"\nVector {i+1}:")
            print(encrypted_X[i])


def main():
    # Initialize forecaster with homomorphic encryption
    forecaster = EncryptedStockVolatilityForecaster("dataset/")
    
    # Check if saved model exists
    if os.path.exists(os.path.join("saved_model_ridge", "volatility_model.pkl")):
        print("Loading existing model and encryption context...")
        if forecaster.load_model():
            print("Model and encryption context loaded successfully")
        else:
            print("Error loading model, will train new model")
            print("Loading and preprocessing data...")
            forecaster.load_and_preprocess_data()
            forecaster.train_model()
    else:
        print("Training new Ridge Regression model with encryption capabilities...")
        print("Loading and preprocessing data...")
        forecaster.load_and_preprocess_data()
        forecaster.train_model()

    print("\nDisplaying model metrics...")
    forecaster.display_model_metrics()
    
    # Example prediction with encrypted data
    company_name = "ITC.NS"  # Replace with desired company
    months_ahead = 1  # Replace with desired number of months
    
    print(f"\nMaking encrypted predictions for {company_name} for {months_ahead} months ahead...")
    try:
        predictions = forecaster.encrypted_predict_volatility(company_name, months_ahead)
        print("\nDecrypted Predicted Volatility:")
        print(predictions)
        
        plt = forecaster.plot_forecast(company_name, months_ahead)
        plt.show()
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()