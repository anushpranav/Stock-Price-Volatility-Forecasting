import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
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
        self.feature_scaler = MinMaxScaler(feature_range=(-0.1, 0.1))
        self.volatility_scaler = MinMaxScaler(feature_range=(-0.1, 0.1))
        self.model = None
        self.companies = None
        self.data = None
        self.context = None
        self.secret_key = None
        self.setup_encryption_context()
        
    def setup_encryption_context(self):
        """Set up the homomorphic encryption context for secure data storage"""
        context = ts.Context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.global_scale = 2**40
        context.generate_galois_keys()
        self.secret_key = context.secret_key()
        self.context = context
        print("Homomorphic encryption context initialized with enhanced parameters")
    
    def encrypt_data(self, data_array):
        """Encrypt numpy array with improved error handling"""
        try:
            data_array = np.array(data_array, dtype=np.float64)
            max_allowed = 0.1
            if np.any(np.abs(data_array) > max_allowed):
                data_array = np.clip(data_array, -max_allowed, max_allowed)
            
            if np.any(np.isnan(data_array)) or np.any(np.isinf(data_array)):
                data_array = np.nan_to_num(data_array, nan=0.0, posinf=max_allowed, neginf=-max_allowed)
            
            return ts.ckks_vector(self.context, data_array)
        except Exception as e:
            print(f"Encryption error: {str(e)}")
            return ts.ckks_vector(self.context, np.zeros_like(data_array))
    
    def decrypt_data(self, encrypted_vector):
        """Decrypt TenSEAL vector back to numpy array"""
        if self.secret_key is not None:
            return np.array(encrypted_vector.decrypt(self.secret_key))
        elif self.context.has_secret_key():
            return np.array(encrypted_vector.decrypt())
        else:
            raise ValueError("No secret key available for decryption")
    
    def save_model(self, model_dir="saved_model_rf"):
        """Save the trained model and scalers"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Save the Random Forest model
        joblib.dump(self.model, os.path.join(model_dir, 'volatility_model.pkl'))
        
        # Save the scalers and data
        joblib.dump(self.feature_scaler, os.path.join(model_dir, 'feature_scaler.pkl'))
        joblib.dump(self.volatility_scaler, os.path.join(model_dir, 'volatility_scaler.pkl'))
        
        with open(os.path.join(model_dir, 'companies.pkl'), 'wb') as f:
            pickle.dump(self.companies, f)
        
        self.data.to_pickle(os.path.join(model_dir, 'preprocessed_data.pkl'))
        
        try:
            if hasattr(self.context, 'serialize'):
                context_bytes = self.context.serialize()
            else:
                context_bytes = self.context.save()
                
            with open(os.path.join(model_dir, 'encryption_context.bin'), 'wb') as f:
                f.write(context_bytes)
            print("Encryption context saved successfully")
        except Exception as e:
            print(f"Error saving encryption context: {str(e)}")
        
        print(f"Model, data, and encryption context saved to {model_dir}")
    
    def load_model(self, model_dir="saved_model_rf"):
        """Load the trained model and scalers"""
        try:
            self.model = joblib.load(os.path.join(model_dir, 'volatility_model.pkl'))
            self.feature_scaler = joblib.load(os.path.join(model_dir, 'feature_scaler.pkl'))
            self.volatility_scaler = joblib.load(os.path.join(model_dir, 'volatility_scaler.pkl'))
            
            with open(os.path.join(model_dir, 'companies.pkl'), 'rb') as f:
                self.companies = pickle.load(f)
            
            self.data = pd.read_pickle(os.path.join(model_dir, 'preprocessed_data.pkl'))
            
            try:
                with open(os.path.join(model_dir, 'encryption_context.bin'), 'rb') as f:
                    context_bytes = f.read()
                    
                if hasattr(ts.Context, 'load'):
                    self.context = ts.Context.load(context_bytes)
                elif hasattr(ts, 'deserialize_context'):
                    self.context = ts.deserialize_context(context_bytes)
                else:
                    self.setup_encryption_context()
                    
                if self.context.has_secret_key():
                    self.secret_key = self.context.secret_key()
                    
            except FileNotFoundError:
                print("No encryption context found, initializing new one")
                self.setup_encryption_context()
                    
            print("Model, scalers, and data loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def load_and_preprocess_data(self):
        """Load and preprocess all company data with improved data cleaning"""
        all_data = []
        
        csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
        self.companies = [os.path.basename(file).split('_')[0] for file in csv_files]
        
        for file in csv_files:
            try:
                company_name = os.path.basename(file).split('_')[0]
                df = pd.read_csv(file)
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
                df['Company'] = company_name
                
                # Replace infinite values with NaN
                df = df.replace([np.inf, -np.inf], np.nan)
                
                # Basic feature calculations with error handling
                df['Returns'] = df['Close'].pct_change().fillna(0)
                
                # Handle extreme returns
                returns_std = df['Returns'].std()
                df['Returns'] = df['Returns'].clip(-3 * returns_std, 3 * returns_std)
                
                # Calculate volatility with bounds
                df['Volatility'] = df['Returns'].rolling(window=20).std().fillna(0) * np.sqrt(252)
                df['Volatility'] = df['Volatility'].clip(0, 5)  # Cap at 500% annualized volatility
                
                # Moving averages
                df['MA_20'] = df['Close'].rolling(window=20).mean().fillna(method='bfill')
                df['MA_50'] = df['Close'].rolling(window=50).mean().fillna(method='bfill')
                
                # RSI with error handling
                df['RSI'] = self.calculate_rsi(df['Close']).fillna(50)  # Fill NaN with neutral RSI
                df['RSI'] = df['RSI'].clip(0, 100)  # Ensure RSI stays within bounds
                
                # Volume-based features with error handling
                df['Volume'] = df['Volume'].replace(0, df['Volume'].mean())  # Replace zero volume
                df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean().fillna(method='bfill')
                df['Volume_Ratio'] = (df['Volume'] / df['Volume_MA_20']).clip(0, 10)  # Cap at 10x average
                
                # Price-based ratios with error handling
                df['High_Low_Ratio'] = (df['High'] / df['Low']).clip(1, 2)  # Cap at reasonable range
                df['Open_Close_Ratio'] = (df['Open'] / df['Close']).clip(0.5, 2)
                
                # Additional features for Random Forest with bounds
                df['Price_Range'] = ((df['High'] - df['Low']) / df['Close']).clip(0, 1)
                df['Price_Momentum'] = df['Close'].pct_change(periods=5).fillna(0).clip(-1, 1)
                df['Volume_Momentum'] = df['Volume'].pct_change(periods=5).fillna(0).clip(-1, 1)
                
                # Final cleanup
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.fillna(method='ffill').fillna(method='bfill')
                
                all_data.append(df)
                
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
                continue
        
        if not all_data:
            raise ValueError("No valid data files were processed")
        
        self.data = pd.concat(all_data, ignore_index=True)
        self.data = self.data.sort_values(['Company', 'Date'])
        
        # Final validation
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            self.data[col] = self.data[col].clip(
                lower=self.data[col].quantile(0.01),
                upper=self.data[col].quantile(0.99)
            )
        
        return self.data
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator with improved error handling"""
        try:
            delta = prices.diff()
            delta = delta.fillna(0)
            
            # Separate gains and losses
            gains = delta.copy()
            losses = delta.copy()
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = abs(losses)
            
            # Calculate rolling averages
            avg_gain = gains.rolling(window=period, min_periods=1).mean()
            avg_loss = losses.rolling(window=period, min_periods=1).mean()
            
            # Calculate RS and RSI with error handling
            rs = avg_gain / avg_loss
            rs = rs.replace([np.inf, -np.inf], np.nan)
            rsi = 100 - (100 / (1 + rs))
            
            # Clean up results
            rsi = rsi.fillna(50)  # Fill NaN with neutral RSI
            rsi = rsi.clip(0, 100)  # Ensure RSI stays within bounds
            
            return rsi
        except Exception as e:
            print(f"Error calculating RSI: {str(e)}")
            return pd.Series(50, index=prices.index)  # Return neutral RSI in case of error
    
    def prepare_features(self):
        """Prepare features for model training"""
        feature_columns = [
            'Returns', 'Volatility', 'MA_20', 'MA_50', 'RSI', 
            'Volume_Ratio', 'High_Low_Ratio', 'Open_Close_Ratio',
            'Price_Range', 'Price_Momentum', 'Volume_Momentum'
        ]
        
        self.feature_scaler.fit(self.data[feature_columns])
        self.volatility_scaler.fit(self.data[['Volatility']])
        
        X, y = [], []
        encrypted_storage = []  # For secure storage of features
        
        for company in self.data['Company'].unique():
            company_data = self.data[self.data['Company'] == company]
            scaled_features = self.feature_scaler.transform(company_data[feature_columns])
            scaled_volatility = self.volatility_scaler.transform(company_data[['Volatility']])
            
            for i in range(len(company_data) - self.sequence_length):
                feature_sequence = scaled_features[i:(i + self.sequence_length)].flatten()
                target_volatility = scaled_volatility[i + self.sequence_length]
                
                X.append(feature_sequence)
                y.append(target_volatility)
                
                # Encrypt for secure storage
                encrypted_storage.append(self.encrypt_data(feature_sequence))
        
        return np.array(X), np.array(y).reshape(-1, 1), encrypted_storage
    
    def build_model(self):
        """Create a Random Forest model"""
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    
    def train_model(self):
        """Train the Random Forest model"""
        print("Preparing features for training...")
        X, y, _ = self.prepare_features()
        
        print("Building and training Random Forest model...")
        self.model = self.build_model()
        self.model.fit(X, y)
        
        # Calculate training score
        score = self.model.score(X, y)
        print(f"Model R² score on training data: {score:.4f}")
        
        # Feature importance analysis
        feature_columns = [
            'Returns', 'Volatility', 'MA_20', 'MA_50', 'RSI', 
            'Volume_Ratio', 'High_Low_Ratio', 'Open_Close_Ratio',
            'Price_Range', 'Price_Momentum', 'Volume_Momentum'
        ]
        importances = self.model.feature_importances_
        for feature, importance in zip(feature_columns, importances[:11]):
            print(f"{feature}: {importance:.4f}")
        
        print("Saving model and data...")
        self.save_model()
        return self.model
    
    def predict_volatility(self, company_name, months_ahead):
        """Predict future volatility using Random Forest"""
        if company_name not in self.companies:
            raise ValueError(f"Company {company_name} not found in dataset")
        
        company_data = self.data[self.data['Company'] == company_name].copy()
        feature_columns = [
            'Returns', 'Volatility', 'MA_20', 'MA_50', 'RSI', 
            'Volume_Ratio', 'High_Low_Ratio', 'Open_Close_Ratio',
            'Price_Range', 'Price_Momentum', 'Volume_Momentum'
        ]
        
        recent_features = self.feature_scaler.transform(company_data[feature_columns].tail(self.sequence_length))
        
        num_predictions = months_ahead * 21  # Trading days per month
        predictions = []
        current_sequence = recent_features.copy()
        
        for _ in range(num_predictions):
            try:
                # Make prediction with Random Forest
                current_flat = current_sequence.flatten()
                pred_value = self.model.predict(current_flat.reshape(1, -1))[0]
                pred_value = np.clip(pred_value, -0.1, 0.1)
                
            except Exception as e:
                print(f"Prediction error, using fallback: {str(e)}")
                pred_value = np.mean(current_sequence[:, 1])  # Use mean volatility as fallback
            
            predictions.append(pred_value)
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=0)
            new_features = np.zeros(len(feature_columns))
            new_features[1] = pred_value  # Update volatility
            current_sequence[-1] = new_features
        
        # Process predictions
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
        """Plot predicted volatility without historical data"""
        predictions = self.predict_volatility(company_name, months_ahead)
        
        plt.figure(figsize=(15, 7))
        
        # Plot only predicted volatility
        plt.plot(predictions['Date'], predictions['Predicted_Volatility'], 
                label='Predicted Volatility', color='red', linewidth=2)
        
        plt.title(f'Stock Volatility Forecast for {company_name} - Next {months_ahead} Month(s)')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        return plt

def main():
    # Initialize forecaster
    forecaster = EncryptedStockVolatilityForecaster("dataset/")
    
    # Check if saved model exists
    if os.path.exists(os.path.join("saved_model_rf", "volatility_model.pkl")):
        print("Loading existing Random Forest model and encryption context...")
        if forecaster.load_model():
            print("Model and encryption context loaded successfully")
        else:
            print("Error loading model, will train new model")
            print("Loading and preprocessing data...")
            forecaster.load_and_preprocess_data()
            forecaster.train_model()
    else:
        print("Training new Random Forest model with encryption capabilities...")
        print("Loading and preprocessing data...")
        forecaster.load_and_preprocess_data()
        forecaster.train_model()
    
    # Example prediction
    company_name = "ZOMATO.NS"  # Replace with desired company
    months_ahead = 5  # Replace with desired number of months
    
    print(f"\nMaking predictions for {company_name} for {months_ahead} months ahead...")
    try:
        predictions = forecaster.predict_volatility(company_name, months_ahead)
        print("\nPredicted Volatility:")
        print(predictions)
        
        plt = forecaster.plot_forecast(company_name, months_ahead)
        plt.show()
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()