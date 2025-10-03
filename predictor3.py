"""
Enhanced Multi-Currency MT5 LSTM Predictor with Backtesting Support
Author: Jason Rusk
Version: 4.1 (Added historical prediction generation for EA backtesting)

New in v4.1:
- Generate historical predictions for backtesting
- Export MT5-compatible lookup files
- Maintain exact same model/features between live and backtest
"""

# --- CONFIGURATION: SET YOUR MT5 PATH HERE ---
HARDCODED_MT5_FILES_PATH = r"C:\Users\jason\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files"

# --- Do not edit below this line ---
import sys
import os
import subprocess
import warnings
import time
import json
import pickle
import glob
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

def install_package(package_name):
    try: 
        __import__(package_name)
    except ImportError:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

required_packages = ['MetaTrader5', 'pandas', 'numpy', 'tensorflow', 'scikit-learn']
for package in required_packages: 
    install_package(package)

import MetaTrader5 as mt5
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, LayerNormalization, 
                                      MultiHeadAttention, GlobalAveragePooling1D, 
                                      Bidirectional, Conv1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler

np.random.seed(42)
tf.random.set_seed(42)

class UnifiedLSTMPredictor:
    def __init__(self, symbol="EURUSD", related_symbols=None):
        self.symbol = symbol
        self.related_symbols = related_symbols or []
        self.lookback_periods = 60
        self.base_path = self.get_mt5_files_path()
        self.predictions_file = os.path.join(self.base_path, f"predictions_{self.symbol}.json")
        self.status_file = os.path.join(self.base_path, f"lstm_status_{self.symbol}.json")
        self.model_path = os.path.join(self.base_path, f"model_{self.symbol}.h5")
        self.feature_scaler_path = os.path.join(self.base_path, f"feature_scaler_{self.symbol}.pkl")
        self.target_scaler_path = os.path.join(self.base_path, f"target_scaler_{self.symbol}.pkl")
        print(f"Enhanced LSTM Predictor Initialized for {self.symbol} (v4.1)")
        self.initialize_mt5()

    def get_mt5_files_path(self):
        if HARDCODED_MT5_FILES_PATH and os.path.exists(HARDCODED_MT5_FILES_PATH):
            print(f"Using hardcoded path: '{HARDCODED_MT5_FILES_PATH}'")
            return HARDCODED_MT5_FILES_PATH
        else:
            print(f"FATAL ERROR: Hardcoded path NOT FOUND or is empty.")
            sys.exit(1)
    
    def get_real_world_accuracy(self, timeframe):
        """Reads the MQL5 indicator's log file and calculates true accuracy."""
        log_file_path = os.path.join(self.base_path, f"prediction_log_{self.symbol}.csv")
        
        if not os.path.exists(log_file_path):
            return 50.0
        
        try:
            df = pd.read_csv(log_file_path, header=None, 
                             names=['Timeframe', 'PredTime', 'StartPrice', 'PredPrice', 'Expiry', 'Status'])
            
            tf_df = df[(df['Timeframe'] == timeframe) & (df['Status'].isin(['Hit', 'Miss']))]
            
            if len(tf_df) == 0:
                return 50.0
            
            hits = len(tf_df[tf_df['Status'] == 'Hit'])
            total_completed = len(tf_df)
            
            accuracy = (hits / total_completed) * 100.0
            print(f"   -> Real-world accuracy for {timeframe}: {accuracy:.1f}% ({hits}/{total_completed})")
            return accuracy
        except Exception as e:
            print(f"   -> Warning: Could not read accuracy log: {e}")
            return 50.0

    def initialize_mt5(self):
        if not mt5.initialize(): 
            print("Initialize failed")
            return False
        print(f"Connected to MT5: {mt5.account_info().login}")
        return True

    def download_data(self, bars=35000):
        """Download historical data. Default 35000 bars = ~5 years of H1 data"""
        print(f"Downloading multi-timeframe data for {self.symbol}...")
        print(f"   Requesting {bars} H1 bars (~{bars/24/260:.1f} years of trading days)")
        df_h1 = pd.DataFrame(mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H1, 0, bars))
        df_h4 = pd.DataFrame(mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H4, 0, bars//4))
        df_d1 = pd.DataFrame(mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_D1, 0, bars//20))
        
        for df, name in [(df_h1, "H1"), (df_h4, "H4"), (df_d1, "D1")]:
            if df.empty: 
                print(f"Failed to download {name} data")
                return None, None, None
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        
        # Show actual data range downloaded
        date_range_days = (df_h1.index[-1] - df_h1.index[0]).days
        print(f"Downloaded {len(df_h1)} H1 bars")
        print(f"   Date range: {df_h1.index[0].strftime('%Y-%m-%d')} to {df_h1.index[-1].strftime('%Y-%m-%d')}")
        print(f"   Total span: {date_range_days} days (~{date_range_days/365:.1f} years)")
        return df_h1, df_h4, df_d1

    def add_correlation_features(self, df):
        """Add features based on correlated currency pairs"""
        print(f"   Adding correlation features from {len(self.related_symbols)} related pairs...")
        
        for related in self.related_symbols:
            try:
                df_related = pd.DataFrame(
                    mt5.copy_rates_from_pos(related, mt5.TIMEFRAME_H1, 0, len(df) + 100)
                )
                if df_related.empty:
                    continue
                    
                df_related['time'] = pd.to_datetime(df_related['time'], unit='s')
                df_related.set_index('time', inplace=True)
                
                # Add related pair's close price
                df[f'{related}_close'] = df_related['close'].reindex(df.index, method='ffill')
                
                # Add price ratio if valid
                if df[f'{related}_close'].notna().any():
                    df[f'ratio_{related}'] = df['close'] / df[f'{related}_close']
                
                print(f"      Added {related} features")
                
            except Exception as e:
                print(f"      Could not add {related}: {e}")
        
        return df

    def create_features(self, df_h1, df_h4, df_d1):
        print("Creating advanced features (40+ indicators)...")
        df = df_h1.copy()
        
        # Multi-timeframe moving averages
        sma20_h4 = df_h4['close'].rolling(window=20).mean()
        sma20_d1 = df_d1['close'].rolling(window=20).mean()
        df['sma_20_h4'] = sma20_h4.reindex(df.index, method='ffill')
        df['sma_20_d1'] = sma20_d1.reindex(df.index, method='ffill')
        df['sma_20_h1'] = df['close'].rolling(window=20).mean()
        df['sma_50_h1'] = df['close'].rolling(window=50).mean()
        df['ema_20_h1'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50_h1'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14_h1'] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14_h1'] = tr.rolling(14).mean()
        
        # Volatility features
        df['volatility_10'] = df['close'].rolling(10).std()
        df['volatility_30'] = df['close'].rolling(30).std()
        df['volatility_ratio'] = df['volatility_10'] / (df['volatility_30'] + 1e-8)
        
        # Momentum indicators
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['roc_10'] = (df['close'] - df['close'].shift(10)) / (df['close'].shift(10) + 1e-8)
        df['roc_20'] = (df['close'] - df['close'].shift(20)) / (df['close'].shift(20) + 1e-8)
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-8)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)
        df['volume_momentum'] = df['volume'] - df['volume'].shift(5)
        
        # Price action patterns
        df['high_low_range'] = (df['high'] - df['low']) / (df['close'] + 1e-8)
        df['body_size'] = abs(df['close'] - df['open']) / (df['close'] + 1e-8)
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['close'] + 1e-8)
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['close'] + 1e-8)
        
        # Price vs moving averages
        df['price_vs_sma20_h1'] = df['close'] / (df['sma_20_h1'] + 1e-8)
        df['price_vs_sma50_h1'] = df['close'] / (df['sma_50_h1'] + 1e-8)
        df['price_vs_sma20_h4'] = df['close'] / (df['sma_20_h4'] + 1e-8)
        df['price_vs_sma20_d1'] = df['close'] / (df['sma_20_d1'] + 1e-8)
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # Returns
        df['log_return'] = np.log(df['close'] / (df['close'].shift(1) + 1e-8))
        df['log_return_5'] = np.log(df['close'] / (df['close'].shift(5) + 1e-8))
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Add correlation features if available
        if self.related_symbols:
            df = self.add_correlation_features(df)
        
        df.dropna(inplace=True)
        print(f"   Created {len(df.columns)} features from {len(df)} bars")
        return df

    def prepare_data(self, df):
        # Select features (excluding target and time features)
        exclude_cols = ['log_return', 'hour', 'day_of_week']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        target_col = 'log_return'
        
        # Split data: 70% train, 15% validation, 15% test
        train_size = int(len(df) * 0.70)
        val_size = int(len(df) * 0.15)
        
        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]
        
        print(f"   Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        # Use RobustScaler for better outlier handling
        feature_scaler = RobustScaler()
        target_scaler = RobustScaler()
        
        feature_scaler.fit(train_df[feature_cols])
        target_scaler.fit(train_df[[target_col]])
        
        train_scaled_features = feature_scaler.transform(train_df[feature_cols])
        train_scaled_target = target_scaler.transform(train_df[[target_col]])
        
        val_scaled_features = feature_scaler.transform(val_df[feature_cols])
        val_scaled_target = target_scaler.transform(val_df[[target_col]])
        
        def create_sequences(features, target, lookback):
            X, y = [], []
            for i in range(lookback, len(features)):
                X.append(features[i-lookback:i])
                y.append(target[i])
            return np.array(X), np.array(y)
        
        X_train, y_train = create_sequences(train_scaled_features, train_scaled_target, self.lookback_periods)
        X_val, y_val = create_sequences(val_scaled_features, val_scaled_target, self.lookback_periods)
        
        print(f"   Sequence shapes: X_train={X_train.shape}, y_train={y_train.shape}")
        
        return X_train, y_train, X_val, y_val, feature_scaler, target_scaler, feature_cols

    def build_model(self, input_shape):
        """Enhanced model with Bidirectional LSTM, CNN, and attention"""
        print("   Building enhanced neural network architecture...")
        
        inputs = Input(shape=input_shape)
        
        # Bidirectional LSTM layers
        x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
        x = Dropout(0.3)(x)
        x = LayerNormalization()(x)
        
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Dropout(0.3)(x)
        x_lstm = LayerNormalization()(x)
        
        # Multi-head attention with residual connection
        attention = MultiHeadAttention(num_heads=8, key_dim=64)(x_lstm, x_lstm)
        x = LayerNormalization()(attention + x_lstm)
        
        # CNN layer for local pattern detection
        conv = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = LayerNormalization()(conv + x)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        outputs = Dense(1, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        print(f"   Model built with {model.count_params():,} parameters")
        return model

    def train_or_load_model(self, df):
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            try:
                model = load_model(self.model_path)
                with open(self.feature_scaler_path, 'rb') as f: 
                    feature_scaler = pickle.load(f)
                with open(self.target_scaler_path, 'rb') as f: 
                    target_scaler = pickle.load(f)
                
                # Get current feature columns
                _, _, _, _, _, _, feature_cols = self.prepare_data(df)
                
                # Check if feature count matches
                expected_features = feature_scaler.n_features_in_
                actual_features = len(feature_cols)
                
                if expected_features != actual_features:
                    print(f"   Feature mismatch: Model expects {expected_features}, but data has {actual_features}")
                    print(f"   Retraining model with new feature set...")
                    raise ValueError("Feature count mismatch - retraining required")
                
                print("   Model and scalers loaded successfully")
                return model, feature_scaler, target_scaler, feature_cols
                
            except Exception as e: 
                print(f"   Cannot load existing model: {str(e)[:100]}")
                print(f"   Training new model...")
        
        print("Training new enhanced model...")
        X_train, y_train, X_val, y_val, feature_scaler, target_scaler, feature_cols = self.prepare_data(df)
        
        model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1)
        ]
        
        print("\n   Starting training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=150,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model and scalers
        model.save(self.model_path)
        with open(self.feature_scaler_path, 'wb') as f: 
            pickle.dump(feature_scaler, f)
        with open(self.target_scaler_path, 'wb') as f: 
            pickle.dump(target_scaler, f)
        
        print(f"\n   Model saved to {self.model_path}")
        
        return model, feature_scaler, target_scaler, feature_cols

    def generate_historical_predictions(self, df, model, feature_scaler, target_scaler, feature_cols):
        """Generate predictions for all historical bars for backtesting"""
        print("\n" + "="*60)
        print("GENERATING HISTORICAL PREDICTIONS FOR BACKTESTING")
        print("="*60)
        
        timeframes = {"1H": 1, "4H": 4, "1D": 24, "5D": 120}
        all_predictions = {tf: [] for tf in timeframes.keys()}
        timestamps = []
        
        # Prepare all features
        features_scaled = feature_scaler.transform(df[feature_cols].values)
        
        # Generate predictions for each bar (after lookback period)
        print(f"Generating predictions for {len(df) - self.lookback_periods} bars...")
        
        for i in range(self.lookback_periods, len(df)):
            # Get sequence for this bar
            sequence = features_scaled[i-self.lookback_periods:i]
            X_pred = sequence.reshape(1, self.lookback_periods, len(feature_cols))
            
            # Current price
            current_price = df['close'].iloc[i]
            timestamp = df.index[i]
            
            # Predict for each timeframe
            for tf_name, steps in timeframes.items():
                pred_log_return_scaled = model.predict(X_pred, verbose=0)[0][0]
                pred_log_return = target_scaler.inverse_transform([[pred_log_return_scaled]])[0][0]
                predicted_price = current_price * np.exp(pred_log_return * steps)
                
                all_predictions[tf_name].append(predicted_price)
            
            timestamps.append(timestamp)
            
            # Progress update
            if (i - self.lookback_periods) % 500 == 0:
                progress = ((i - self.lookback_periods) / (len(df) - self.lookback_periods)) * 100
                print(f"   Progress: {progress:.1f}% ({i - self.lookback_periods}/{len(df) - self.lookback_periods})")
        
        print(f"\nGenerated {len(timestamps)} predictions per timeframe")
        
        # Export to MT5-compatible files
        self.export_backtest_files(timestamps, all_predictions, df['close'].iloc[self.lookback_periods:].values)
        
        return timestamps, all_predictions

    def export_backtest_files(self, timestamps, predictions, actual_prices):
        """Export predictions in MT5 EA-compatible format"""
        print("\nExporting backtest files...")
        
        output_dir = os.path.join(self.base_path, 'predictions')
        os.makedirs(output_dir, exist_ok=True)
        
        for tf_name, pred_values in predictions.items():
            # Create lookup file for EA (timestamp, prediction)
            lookup_file = os.path.join(self.base_path, f'{self.symbol}_{tf_name}_lookup.csv')
            
            with open(lookup_file, 'w') as f:
                for ts, pred in zip(timestamps, pred_values):
                    ts_str = ts.strftime('%Y.%m.%d %H:%M')
                    f.write(f'{ts_str},{pred:.5f}\n')
            
            print(f"   Created: {lookup_file}")
            
            # Create detailed CSV with actual vs predicted
            detail_file = os.path.join(output_dir, f'{self.symbol}_{tf_name}_predictions.csv')
            
            df_export = pd.DataFrame({
                'timestamp': timestamps,
                'actual': actual_prices,
                'predicted': pred_values,
                'error': actual_prices - pred_values,
                'direction_correct': np.sign(pred_values - actual_prices) == np.sign(actual_prices - np.roll(actual_prices, 1))
            })
            
            df_export.to_csv(detail_file, index=False)
            print(f"   Created: {detail_file}")
        
        # Create summary JSON
        summary = {
            'symbol': self.symbol,
            'generated_at': datetime.now().isoformat(),
            'total_bars': len(timestamps),
            'date_range': {
                'start': timestamps[0].isoformat(),
                'end': timestamps[-1].isoformat()
            },
            'timeframes': list(predictions.keys())
        }
        
        summary_file = os.path.join(output_dir, f'predictions_{self.symbol}.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   Created: {summary_file}")
        print("\nBacktest files ready for MT5 EA!")

    def run_prediction_cycle(self):
        print("\n" + "="*60)
        print(f"Starting Prediction Cycle for {self.symbol}")
        print(f"   Time: {datetime.now()}")
        print("="*60)
        
        df_h1, df_h4, df_d1 = self.download_data()
        if df_h1 is None: 
            print("Failed to download data")
            return
        
        df = self.create_features(df_h1, df_h4, df_d1)
        model, feature_scaler, target_scaler, feature_cols = self.train_or_load_model(df)
        
        current_price = df['close'].iloc[-1]
        
        # Prepare last sequence for prediction
        last_sequence_raw = df.iloc[-self.lookback_periods:][feature_cols].values
        last_sequence_scaled = feature_scaler.transform(last_sequence_raw)
        X_pred = last_sequence_scaled.reshape(1, self.lookback_periods, last_sequence_scaled.shape[1])
        
        predictions = {}
        timeframes = {"1H": 1, "4H": 4, "1D": 24, "5D": 120}
        
        print("\nMaking predictions and checking real-world accuracy...")
        
        for tf_name, steps in timeframes.items():
            pred_log_return_scaled = model.predict(X_pred, verbose=0)[0][0]
            pred_log_return = target_scaler.inverse_transform([[pred_log_return_scaled]])[0][0]
            predicted_price = current_price * np.exp(pred_log_return * steps)
            
            # Get real-world accuracy from MQL5 log
            accuracy = self.get_real_world_accuracy(tf_name)
            
            predictions[tf_name] = {
                'prediction': round(predicted_price, 5),
                'accuracy': round(accuracy, 2)
            }
        
        status = {
            'last_update': datetime.now().isoformat(),
            'status': 'online',
            'symbol': self.symbol,
            'current_price': round(current_price, 5)
        }
        
        self.save_to_file(self.predictions_file, predictions)
        self.save_to_file(self.status_file, status)
        
        print("\nPrediction Cycle Complete!")
        print(f"Current Price: {current_price:.5f}")
        for tf, data in predictions.items():
            direction = "UP" if data['prediction'] > current_price else "DOWN"
            print(f"   {direction} {tf}: {data['prediction']:.5f} (Accuracy: {data['accuracy']:.1f}%)")

    def run_backtest_generation(self):
        """Generate all historical predictions for backtesting"""
        print("\n" + "="*60)
        print(f"BACKTEST MODE: Generating Historical Predictions")
        print(f"   Symbol: {self.symbol}")
        print("="*60)
        
        # Download 5 years of historical data
        df_h1, df_h4, df_d1 = self.download_data(bars=35000)
        if df_h1 is None:
            print("Failed to download data")
            return
        
        df = self.create_features(df_h1, df_h4, df_d1)
        model, feature_scaler, target_scaler, feature_cols = self.train_or_load_model(df)
        
        # Generate historical predictions
        self.generate_historical_predictions(df, model, feature_scaler, target_scaler, feature_cols)
        
        print("\n" + "="*60)
        print("BACKTEST GENERATION COMPLETE!")
        print("="*60)
        print("\nGenerated files:")
        print(f"  - {self.symbol}_1H_lookup.csv")
        print(f"  - {self.symbol}_4H_lookup.csv")
        print(f"  - {self.symbol}_1D_lookup.csv")
        print(f"  - {self.symbol}_5D_lookup.csv")
        print("\nThese files are ready to use with your MT5 EA for backtesting.")

    def save_to_file(self, file_path, data):
        try:
            with open(file_path, 'w') as f: 
                json.dump(data, f, indent=4)
        except Exception as e: 
            print(f"Error saving to {file_path}: {e}")

    def run_continuous(self, interval_minutes=60):
        print(f"\nStarting Continuous Mode for {self.symbol}")
        print(f"   Update interval: {interval_minutes} minutes")
        
        while True:
            try:
                self.run_prediction_cycle()
                print(f"\nWaiting {interval_minutes} minutes until next cycle...")
                time.sleep(interval_minutes * 60)
            except KeyboardInterrupt:
                print("\nService stopped by user.")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                print("   Retrying in 5 minutes...")
                time.sleep(300)


def main():
    print("""
    ╔═════════════════════════════════════════════════════════╗
    ║   Enhanced MT5 LSTM Predictor v4.1                      ║
    ║   Now with Backtesting Support                          ║
    ╚═════════════════════════════════════════════════════════╝
    """)
    
    if len(sys.argv) > 1 and sys.argv[1] == 'backtest':
        # Backtest mode - generate historical predictions
        symbol = sys.argv[2] if len(sys.argv) > 2 else "EURUSD"
        
        related_map = {
            "EURUSD": ["GBPUSD", "USDJPY"],
            "GBPUSD": ["EURUSD", "USDJPY"],
            "USDJPY": ["EURUSD", "GBPUSD"],
            "AUDUSD": ["EURUSD", "USDJPY"]
        }
        related = related_map.get(symbol, [])
        
        predictor = UnifiedLSTMPredictor(symbol, related_symbols=related)
        predictor.run_backtest_generation()
        
    else:
        # Normal mode - single prediction or continuous
        symbol = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != 'continuous' else "EURUSD"
        
        related_map = {
            "EURUSD": ["GBPUSD", "USDJPY"],
            "GBPUSD": ["EURUSD", "USDJPY"],
            "USDJPY": ["EURUSD", "GBPUSD"],
            "AUDUSD": ["EURUSD", "USDJPY"]
        }
        related = related_map.get(symbol, [])
        
        predictor = UnifiedLSTMPredictor(symbol, related_symbols=related)
        
        if 'continuous' in sys.argv:
            interval = 60
            for arg in sys.argv:
                if arg.isdigit():
                    interval = int(arg)
                    break
            predictor.run_continuous(interval)
        else:
            predictor.run_prediction_cycle()
    
    mt5.shutdown()
    print("\nShutdown complete. Thank you!")


if __name__ == "__main__":
    main()