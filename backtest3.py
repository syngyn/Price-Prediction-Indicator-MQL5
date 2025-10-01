"""
GGTH Prediction Generator for Backtesting - MT5 Version
Generates historical predictions using ML models for EA backtesting
Auto-downloads data from MetaTrader 5
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, accuracy_score
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import ML libraries
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("TensorFlow not found. Install with: pip install tensorflow")

# Try to import MT5
try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False
    print("MetaTrader5 not found. Install with: pip install MetaTrader5")

class PredictionGenerator:
    def __init__(self, symbol='EURUSD', data_file=None):
        """
        Initialize the prediction generator
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            data_file: Path to CSV file with historical data
        """
        self.symbol = symbol
        self.data_file = data_file
        self.data = None
        self.models = {}
        self.scalers = {}
        self.predictions = {}
        
        # Timeframe configurations (in hours)
        self.timeframes = {
            '1H': 1,
            '4H': 4,
            '1D': 24,
            '5D': 120
        }
        
    def load_data(self, data_file=None):
        """Load historical price data from CSV"""
        if data_file:
            self.data_file = data_file
            
        if not self.data_file:
            raise ValueError("No data file specified")
        
        print(f"Loading data from {self.data_file}...")
        
        # Expected format: timestamp, open, high, low, close, volume
        self.data = pd.read_csv(self.data_file)
        
        # Standardize column names
        self.data.columns = [col.lower().strip() for col in self.data.columns]
        
        # Convert timestamp to datetime
        if 'timestamp' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        elif 'time' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['time'])
        elif 'date' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['date'])
        elif 'datetime' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['datetime'])
        else:
            # If no timestamp, create one
            self.data['timestamp'] = pd.date_range(
                start='2020-01-01', 
                periods=len(self.data), 
                freq='1H'
            )
        
        # Ensure we have required columns
        required_cols = ['close', 'high', 'low', 'open']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Sort by timestamp
        self.data = self.data.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Loaded {len(self.data)} data points")
        print(f"Date range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
        
        return self.data
    
    def create_features(self, data):
        """Create technical indicators as features"""
        df = data.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        # Price momentum
        df['momentum'] = df['close'] - df['close'].shift(10)
        
        # High-Low range
        df['hl_range'] = df['high'] - df['low']
        df['hl_ratio'] = df['hl_range'] / df['close']
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def prepare_sequences(self, data, timeframe_hours, lookback=60):
        """Prepare sequences for LSTM model"""
        feature_cols = [col for col in data.columns 
                       if col not in ['timestamp', 'close']]
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[feature_cols])
        
        X, y, timestamps = [], [], []
        
        target_steps = timeframe_hours  # Predict N hours ahead
        
        for i in range(lookback, len(scaled_data) - target_steps):
            X.append(scaled_data[i-lookback:i])
            y.append(data['close'].iloc[i + target_steps])
            timestamps.append(data['timestamp'].iloc[i])
        
        return np.array(X), np.array(y), timestamps, scaler
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model for predictions"""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow required. Install with: pip install tensorflow")
        
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        
        return model
    
    def train_model(self, timeframe, epochs=50, batch_size=32):
        """Train model for specific timeframe"""
        print(f"\n{'='*60}")
        print(f"Training model for {timeframe} predictions")
        print(f"{'='*60}")
        
        # Prepare data
        df_features = self.create_features(self.data)
        
        # Prepare sequences
        timeframe_hours = self.timeframes[timeframe]
        X, y, timestamps, scaler = self.prepare_sequences(
            df_features, 
            timeframe_hours,
            lookback=60
        )
        
        self.scalers[timeframe] = scaler
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        timestamps_test = timestamps[split_idx:]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Build and train model
        model = self.build_lstm_model((X.shape[1], X.shape[2]))
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        
        # Make predictions on test set
        y_pred = model.predict(X_test).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Direction accuracy
        y_test_direction = np.sign(np.diff(y_test, prepend=y_test[0]))
        y_pred_direction = np.sign(np.diff(y_pred, prepend=y_pred[0]))
        direction_accuracy = accuracy_score(
            y_test_direction[1:] > 0, 
            y_pred_direction[1:] > 0
        ) * 100
        
        print(f"\nResults for {timeframe}:")
        print(f"MAE: {mae:.5f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Direction Accuracy: {direction_accuracy:.2f}%")
        
        # Store predictions
        self.predictions[timeframe] = {
            'timestamps': timestamps_test,
            'actual': y_test,
            'predicted': y_pred,
            'accuracy': direction_accuracy,
            'mae': mae,
            'mape': mape
        }
        
        self.models[timeframe] = model
        
        return model, history
    
    def generate_all_predictions(self, epochs=50):
        """Generate predictions for all timeframes"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        for timeframe in self.timeframes.keys():
            try:
                self.train_model(timeframe, epochs=epochs)
            except Exception as e:
                print(f"Error training {timeframe} model: {e}")
                continue
    
    def export_for_mt5(self, output_dir='predictions'):
        """Export predictions in MT5-compatible format"""
        if not self.predictions:
            raise ValueError("No predictions generated yet")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create main predictions file
        output_file = os.path.join(output_dir, f'predictions_{self.symbol}.json')
        
        export_data = {
            'symbol': self.symbol,
            'generated_at': datetime.now().isoformat(),
            'timeframes': {}
        }
        
        # Export each timeframe
        for timeframe, pred_data in self.predictions.items():
            # Create detailed CSV for this timeframe
            csv_file = os.path.join(output_dir, f'{self.symbol}_{timeframe}_predictions.csv')
            
            df = pd.DataFrame({
                'timestamp': pred_data['timestamps'],
                'actual': pred_data['actual'],
                'predicted': pred_data['predicted'],
                'error': pred_data['actual'] - pred_data['predicted'],
            })
            
            df.to_csv(csv_file, index=False)
            print(f"Saved detailed predictions: {csv_file}")
            
            # Add summary to JSON
            export_data['timeframes'][timeframe] = {
                'accuracy': float(pred_data['accuracy']),
                'mae': float(pred_data['mae']),
                'mape': float(pred_data['mape']),
                'total_predictions': len(pred_data['predicted']),
                'csv_file': csv_file
            }
        
        # Save JSON summary
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\nSaved predictions summary: {output_file}")
        
        # Also create a simple lookup file for the EA
        self._create_ea_lookup_file(output_dir)
        
        return output_file
    
    def _create_ea_lookup_file(self, output_dir):
        """Create a simple timestamp->prediction lookup file for EA"""
        for timeframe, pred_data in self.predictions.items():
            lookup_file = os.path.join(
                output_dir, 
                f'{self.symbol}_{timeframe}_lookup.csv'
            )
            
            df = pd.DataFrame({
                'timestamp': [ts.strftime('%Y.%m.%d %H:%M') 
                             for ts in pred_data['timestamps']],
                'prediction': pred_data['predicted']
            })
            
            df.to_csv(lookup_file, index=False, header=False)
            print(f"Created EA lookup file: {lookup_file}")
    
    def plot_predictions(self, timeframe='4H', num_points=500):
        """Plot actual vs predicted prices"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not installed. Install with: pip install matplotlib")
            return
        
        if timeframe not in self.predictions:
            print(f"No predictions for {timeframe}")
            return
        
        pred_data = self.predictions[timeframe]
        
        # Plot last N points
        actual = pred_data['actual'][-num_points:]
        predicted = pred_data['predicted'][-num_points:]
        timestamps = pred_data['timestamps'][-num_points:]
        
        plt.figure(figsize=(15, 7))
        plt.plot(timestamps, actual, label='Actual', alpha=0.7)
        plt.plot(timestamps, predicted, label='Predicted', alpha=0.7)
        plt.title(f'{self.symbol} - {timeframe} Predictions')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.symbol}_{timeframe}_predictions.png', dpi=300)
        print(f"Saved plot: {self.symbol}_{timeframe}_predictions.png")
        plt.show()


def download_from_mt5(symbol='EURUSD', days=730, timeframe='1H'):
    """Download data directly from MetaTrader 5"""
    
    if not HAS_MT5:
        print("="*60)
        print("ERROR: MetaTrader5 package not installed")
        print("="*60)
        print("Install with: pip install MetaTrader5")
        print("\nOr export data manually from MT5:")
        print("1. Open MT5 → Select symbol")
        print("2. Open Data Window (Ctrl+D)")
        print("3. Right-click → Export to CSV")
        return None
    
    print(f"Connecting to MetaTrader 5...")
    
    # Initialize MT5 connection
    if not mt5.initialize():
        print("="*60)
        print("ERROR: Failed to initialize MT5")
        print("="*60)
        print(f"Error code: {mt5.last_error()}")
        print("\nTroubleshooting:")
        print("1. Make sure MetaTrader 5 is installed")
        print("2. Make sure MT5 is running")
        print("3. Make sure you're logged into an account")
        print("4. Try running Python as Administrator")
        print("="*60)
        return None
    
    print(f"✓ MT5 connected successfully")
    print(f"  Terminal: {mt5.terminal_info().name}")
    print(f"  Version: {mt5.version()}")
    
    # Map timeframe string to MT5 constant
    timeframe_map = {
        '1M': mt5.TIMEFRAME_M1,
        '5M': mt5.TIMEFRAME_M5,
        '15M': mt5.TIMEFRAME_M15,
        '30M': mt5.TIMEFRAME_M30,
        '1H': mt5.TIMEFRAME_H1,
        '4H': mt5.TIMEFRAME_H4,
        '1D': mt5.TIMEFRAME_D1,
        '1W': mt5.TIMEFRAME_W1,
    }
    
    mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)
    
    # Calculate date range
    from_date = datetime.now() - timedelta(days=days)
    to_date = datetime.now()
    
    print(f"\nDownloading {days} days of {symbol} data...")
    print(f"  Timeframe: {timeframe}")
    print(f"  From: {from_date.strftime('%Y-%m-%d')}")
    print(f"  To: {to_date.strftime('%Y-%m-%d')}")
    
    # Get rates
    rates = mt5.copy_rates_range(symbol, mt5_timeframe, from_date, to_date)
    
    # Shutdown MT5
    mt5.shutdown()
    
    if rates is None or len(rates) == 0:
        print("="*60)
        print(f"ERROR: Failed to get data for {symbol}")
        print("="*60)
        print(f"MT5 Error: {mt5.last_error()}")
        print("\nTroubleshooting:")
        print(f"1. Check symbol name - try these variations:")
        print(f"   - {symbol}")
        print(f"   - {symbol}m")
        print(f"   - {symbol}.a")
        print(f"   - {symbol}#")
        print("2. In MT5 Market Watch:")
        print("   - Right-click → Symbols → Show All")
        print("   - Find your symbol and note EXACT name")
        print("3. Make sure symbol is in Market Watch")
        print("4. Load more history:")
        print("   - Tools → Options → Charts")
        print("   - Max bars in history: 999999")
        print("   - Scroll chart backwards to load history")
        print("="*60)
        return None
    
    # Convert to DataFrame
    data = pd.DataFrame(rates)
    
    # Rename columns to standard format
    data['timestamp'] = pd.to_datetime(data['time'], unit='s')
    data = data.rename(columns={
        'tick_volume': 'volume'
    })
    
    # Select and reorder columns
    data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # Save to CSV
    filename = f'{symbol}_historical_{timeframe}.csv'
    data.to_csv(filename, index=False)
    
    print(f"\n✓ Downloaded {len(data)} bars")
    print(f"  Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"  Saved to: {filename}")
    
    return filename


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("GGTH Prediction Generator - MT5 Version")
    print("="*60)
    
    # ==================== CONFIGURATION ====================
    SYMBOL = 'EURUSD'       # Change to your broker's exact symbol name
    DATA_FILE = None        # Set to CSV path, or None for auto-download
    TIMEFRAME = '1H'        # 1M, 5M, 15M, 30M, 1H, 4H, 1D, 1W
    DAYS = 730              # Days of history (730 = 2 years)
    EPOCHS = 30             # Training epochs (30 fast, 50-100 better)
    # ======================================================
    
    print(f"\nConfiguration:")
    print(f"  Symbol: {SYMBOL}")
    print(f"  Timeframe: {TIMEFRAME}")
    print(f"  History: {DAYS} days")
    print(f"  Epochs: {EPOCHS}")
    
    # Check if required packages are installed
    if not HAS_TENSORFLOW:
        print("\n" + "="*60)
        print("ERROR: TensorFlow not installed")
        print("="*60)
        print("Install with: pip install tensorflow")
        exit(1)
    
    # Download or load data
    if DATA_FILE is None:
        print("\n" + "="*60)
        print("STEP 1: Downloading data from MT5")
        print("="*60)
        print("Make sure MT5 is running and logged in!")
        input("Press Enter when ready...")
        
        DATA_FILE = download_from_mt5(SYMBOL, days=DAYS, timeframe=TIMEFRAME)
        
        if DATA_FILE is None:
            print("\n" + "="*60)
            print("FAILED TO DOWNLOAD DATA")
            print("="*60)
            print("\nPlease either:")
            print("1. Fix MT5 connection issues (see above)")
            print("2. Export CSV manually and set DATA_FILE path")
            print("="*60)
            exit(1)
    
    # Initialize generator
    print("\n" + "="*60)
    print("STEP 2: Loading and preparing data")
    print("="*60)
    generator = PredictionGenerator(symbol=SYMBOL, data_file=DATA_FILE)
    generator.load_data()
    
    # Generate predictions for all timeframes
    print("\n" + "="*60)
    print("STEP 3: Training ML models")
    print("="*60)
    print("This may take 10-30 minutes depending on your hardware...")
    
    generator.generate_all_predictions(epochs=EPOCHS)
    
    # Export for MT5
    print("\n" + "="*60)
    print("STEP 4: Exporting predictions")
    print("="*60)
    generator.export_for_mt5()
    
    # Plot results
    try:
        generator.plot_predictions('4H')
    except:
        print("Could not create plot (matplotlib may not be installed)")
    
    # Print final instructions
    print("\n" + "="*60)
    print("SUCCESS! Predictions generated")
    print("="*60)
    print("\nGenerated files in 'predictions' folder:")
    for tf in ['1H', '4H', '1D', '5D']:
        print(f"  - {SYMBOL}_{tf}_lookup.csv")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Copy prediction files to MT5:")
    print(f"   From: predictions\\")
    print(f"   To: MT5_DATA_FOLDER\\MQL5\\Files\\")
    print("\n   To find MT5 Data Folder:")
    print("   - Open MT5 → File → Open Data Folder")
    print("\n2. In MT5 Strategy Tester:")
    print("   - Load GGTH Predictor EA")
    print("   - Set: UsePredictionFile = true")
    print("   - Set: PredictionTimeframe = '4H' (or 1H/1D/5D)")
    print("   - Select backtest period")
    print("   - Run backtest")
    print("="*60)