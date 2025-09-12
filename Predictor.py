"""
Unified and Corrected MT5 LSTM Predictor
Author: Jason Rusk
Version: 3.0 (Adaptive Feedback Loop)

This version creates a true feedback loop with the MQL5 indicator:
1.  Reads the MQL5 'prediction_log.csv' to get real-world performance.
2.  Reports the ACTUAL tracked accuracy back to the indicator instead of an internal metric.
3.  Includes Multi-Timeframe Analysis and Advanced Features from v2.3.1.
"""

# --- ❗ CONFIGURATION: SET YOUR MT5 PATH HERE ❗ ---
HARDCODED_MT5_FILES_PATH = r"C:\Users\jason\AppData\Roaming\MetaQuotes\Terminal\5C659F0E64BA794E712EE4C936BCFED5\MQL5\Files"


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
    try: __import__(package_name)
    except ImportError:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

required_packages = ['MetaTrader5', 'pandas', 'numpy', 'tensorflow', 'scikit-learn']
for package in required_packages: install_package(package)

import MetaTrader5 as mt5
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)
tf.random.set_seed(42)

class UnifiedLSTMPredictor:
    def __init__(self, symbol="EURUSD"):
        self.symbol = symbol
        self.lookback_periods = 60
        self.base_path = self.get_mt5_files_path()
        self.predictions_file = os.path.join(self.base_path, f"predictions_{self.symbol}.json")
        self.status_file = os.path.join(self.base_path, f"lstm_status_{self.symbol}.json")
        self.model_path = os.path.join(self.base_path, f"model_{self.symbol}.h5")
        self.feature_scaler_path = os.path.join(self.base_path, f"feature_scaler_{self.symbol}.pkl")
        self.target_scaler_path = os.path.join(self.base_path, f"target_scaler_{self.symbol}.pkl")
        print("🧠 Unified LSTM Predictor Initialized (v3.0 - Adaptive).")
        self.initialize_mt5()

    def get_mt5_files_path(self):
        if HARDCODED_MT5_FILES_PATH and os.path.exists(HARDCODED_MT5_FILES_PATH):
            print(f"✅ Using hardcoded path: '{HARDCODED_MT5_FILES_PATH}'")
            return HARDCODED_MT5_FILES_PATH
        else:
            print(f"❌ FATAL ERROR: Hardcoded path NOT FOUND or is empty.")
            sys.exit(1)
    
    # --- NEW: FUNCTION TO READ MQL5 LOG AND CALCULATE REAL ACCURACY ---
    def get_real_world_accuracy(self, timeframe):
        """Reads the MQL5 indicator's log file and calculates true accuracy."""
        log_file_path = os.path.join(self.base_path, f"prediction_log_{self.symbol}.csv")
        
        if not os.path.exists(log_file_path):
            return 50.0  # Default if no log exists yet

        try:
            # The log file has no header, so we name the columns
            df = pd.read_csv(log_file_path, header=None, 
                             names=['Timeframe', 'PredTime', 'StartPrice', 'PredPrice', 'Expiry', 'Status'])
            
            # Filter for the specific timeframe and completed predictions
            tf_df = df[(df['Timeframe'] == timeframe) & (df['Status'].isin(['Hit', 'Miss']))]
            
            if len(tf_df) == 0:
                return 50.0  # Default if no completed predictions for this timeframe

            hits = len(tf_df[tf_df['Status'] == 'Hit'])
            total_completed = len(tf_df)
            
            accuracy = (hits / total_completed) * 100.0
            print(f"   -> Real-world accuracy for {timeframe}: {accuracy:.1f}% ({hits}/{total_completed})")
            return accuracy
        except Exception as e:
            print(f"   -> Warning: Could not read or parse MQL5 accuracy log: {e}")
            return 50.0 # Return default on error

    def initialize_mt5(self):
        if not mt5.initialize(): print("❌ Initialize failed"); return False
        print(f"✅ Connected to MT5: {mt5.account_info().login}"); return True

    def download_data(self):
        print(f"📊 Downloading multi-timeframe data for {self.symbol}...")
        df_h1 = pd.DataFrame(mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H1, 0, 10000))
        df_h4 = pd.DataFrame(mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H4, 0, 2500))
        df_d1 = pd.DataFrame(mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_D1, 0, 500))
        for df, name in [(df_h1, "H1"), (df_h4, "H4"), (df_d1, "D1")]:
            if df.empty: return None, None, None
            df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        return df_h1, df_h4, df_d1

    def create_features(self, df_h1, df_h4, df_d1):
        print("⚙️ Creating advanced features (Multi-Timeframe, ATR)...")
        df = df_h1.copy()
        sma20_h4 = df_h4['close'].rolling(window=20).mean()
        sma20_d1 = df_d1['close'].rolling(window=20).mean()
        df['sma_20_h4'] = sma20_h4.reindex(df.index, method='ffill')
        df['sma_20_d1'] = sma20_d1.reindex(df.index, method='ffill')
        df['sma_20_h1'] = df['close'].rolling(window=20).mean()
        df['ema_20_h1'] = df['close'].ewm(span=20, adjust=False).mean()
        delta = df['close'].diff(); gain = (delta.where(delta > 0, 0)).rolling(window=14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean(); rs = gain / loss
        df['rsi_14_h1'] = 100 - (100 / (1 + rs))
        high_low = df['high'] - df['low']; high_close = np.abs(df['high'] - df['close'].shift()); low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14_h1'] = tr.rolling(14).mean()
        df['price_vs_sma20_h1'] = df['close'] / df['sma_20_h1']; df['price_vs_sma20_h4'] = df['close'] / df['sma_20_h4']; df['price_vs_sma20_d1'] = df['close'] / df['sma_20_d1']
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df.dropna(inplace=True)
        return df

    def prepare_data(self, df):
        feature_cols = ['open','high','low','close','volume','sma_20_h1','ema_20_h1','rsi_14_h1','sma_20_h4','sma_20_d1','atr_14_h1','price_vs_sma20_h1','price_vs_sma20_h4','price_vs_sma20_d1']
        target_col = 'log_return'
        train_size = int(len(df) * 0.85); train_df, val_df = df[:train_size], df[train_size:]
        feature_scaler = MinMaxScaler(feature_range=(0, 1)); target_scaler = MinMaxScaler(feature_range=(0, 1))
        feature_scaler.fit(train_df[feature_cols]); target_scaler.fit(train_df[[target_col]])
        train_scaled_features = feature_scaler.transform(train_df[feature_cols]); train_scaled_target = target_scaler.transform(train_df[[target_col]])
        val_scaled_features = feature_scaler.transform(val_df[feature_cols]); val_scaled_target = target_scaler.transform(val_df[[target_col]])
        def create_sequences(features, target, lookback):
            X, y = [], []; [X.append(features[i-lookback:i]) or y.append(target[i]) for i in range(lookback, len(features))]; return np.array(X), np.array(y)
        X_train, y_train = create_sequences(train_scaled_features, train_scaled_target, self.lookback_periods)
        X_val, y_val = create_sequences(val_scaled_features, val_scaled_target, self.lookback_periods)
        return X_train, y_train, X_val, y_val, feature_scaler, target_scaler, feature_cols

    def build_model(self, input_shape):
        inputs = Input(shape=input_shape); x = LSTM(100, return_sequences=True)(inputs); x = Dropout(0.2)(x); x = LayerNormalization()(x)
        x = LSTM(50, return_sequences=True)(x); x = Dropout(0.2)(x); x = LayerNormalization()(x)
        attention = MultiHeadAttention(num_heads=8, key_dim=50)(x, x); x = GlobalAveragePooling1D()(attention)
        x = Dense(50, activation='relu')(x); x = Dropout(0.2)(x); outputs = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae']); return model

    def train_or_load_model(self, df):
        if os.path.exists(self.model_path):
            print(f"🧠 Loading existing model from {self.model_path}")
            try:
                model = load_model(self.model_path)
                with open(self.feature_scaler_path, 'rb') as f: feature_scaler = pickle.load(f)
                with open(self.target_scaler_path, 'rb') as f: target_scaler = pickle.load(f)
                _, _, _, _, _, _, feature_cols = self.prepare_data(df)
                return model, feature_scaler, target_scaler, feature_cols
            except Exception as e: print(f"❌ Failed to load model/scalers: {e}. Retraining...")
        print("💪 No existing model found. Training a new one...")
        X_train, y_train, X_val, y_val, feature_scaler, target_scaler, feature_cols = self.prepare_data(df)
        model = self.build_model((X_train.shape[1], X_train.shape[2]))
        callbacks = [EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True),ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=5,min_lr=1e-6)]
        model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=100,batch_size=32,callbacks=callbacks,verbose=1)
        model.save(self.model_path)
        with open(self.feature_scaler_path, 'wb') as f: pickle.dump(feature_scaler, f)
        with open(self.target_scaler_path, 'wb') as f: pickle.dump(target_scaler, f)
        return model, feature_scaler, target_scaler, feature_cols

    def run_prediction_cycle(self):
        print("\n" + "="*60 + f"\n🚀 Starting Prediction Cycle for {self.symbol} at {datetime.now()}\n" + "="*60)
        df_h1, df_h4, df_d1 = self.download_data()
        if df_h1 is None: return
        df = self.create_features(df_h1, df_h4, df_d1)
        model, feature_scaler, target_scaler, feature_cols = self.train_or_load_model(df)
        current_price = df['close'].iloc[-1]
        last_sequence_raw = df.iloc[-self.lookback_periods:][feature_cols].values
        last_sequence_scaled = feature_scaler.transform(last_sequence_raw)
        X_pred = last_sequence_scaled.reshape(1, self.lookback_periods, last_sequence_scaled.shape[1])
        predictions = {}; timeframes = {"1H": 1, "4H": 4, "1D": 24, "5D": 120}
        print("🎯 Making predictions and checking real-world accuracy...")
        for tf_name, steps in timeframes.items():
            pred_log_return_scaled = model.predict(X_pred, verbose=0)[0][0]
            pred_log_return = target_scaler.inverse_transform([[pred_log_return_scaled]])[0][0]
            predicted_price = current_price * np.exp(pred_log_return * steps)
            # --- ADAPTIVE ACCURACY ---
            accuracy = self.get_real_world_accuracy(tf_name)
            predictions[tf_name] = {'prediction': round(predicted_price, 5), 'accuracy': round(accuracy, 2)}
        status = {'last_update': datetime.now().isoformat(), 'status': 'online', 'symbol': self.symbol, 'current_price': round(current_price, 5)}
        self.save_to_file(self.predictions_file, predictions)
        self.save_to_file(self.status_file, status)
        print("\n🎉 Prediction Cycle Complete!"); print(f"💰 Current Price: {current_price:.5f}")
        for tf, data in predictions.items(): print(f"   {tf}: {data['prediction']:.5f} (Reported Acc: {data['accuracy']:.1f}%)")

    def save_to_file(self, file_path, data):
        try:
            with open(file_path, 'w') as f: json.dump(data, f, indent=4)
        except Exception as e: print(f"❌ Error saving to {file_path}: {e}")

    def run_continuous(self, interval_minutes=60):
        print(f"\n🚀 Starting Continuous Mode. Update interval: {interval_minutes} minutes.")
        while True:
            try:
                self.run_prediction_cycle()
                print(f"\n⏰ Waiting for {interval_minutes} minutes until next cycle...")
                time.sleep(interval_minutes * 60)
            except KeyboardInterrupt: print("\n🛑 Service stopped by user."); break
            except Exception as e: print(f"❌ An error occurred: {e}\n   Retrying in 5 minutes..."); time.sleep(300)

def main():
    predictor = UnifiedLSTMPredictor("EURUSD")
    if len(sys.argv) > 1 and sys.argv[1] == 'continuous':
        interval = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 60
        predictor.run_continuous(interval)
    else:
        predictor.run_prediction_cycle()
    mt5.shutdown()

if __name__ == "__main__":
    main()