"""
COMPLETE FOREX PREDICTION SYSTEM
Save this as: forex_system.py

This file contains everything you need to train and predict forex prices.
"""

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import ta
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# MT5 Data Extraction
# ============================================================================

class MT5DataExtractor:
    def __init__(self, symbol="EURUSD", timeframe=mt5.TIMEFRAME_H1):
        self.symbol = symbol
        self.timeframe = timeframe
        
    def initialize(self):
        if not mt5.initialize():
            print("MT5 initialization failed")
            return False
        return True
    
    def get_historical_data(self, years=10):
        """Extract historical data from MT5"""
        if not self.initialize():
            return None
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*years)
        
        rates = mt5.copy_rates_range(self.symbol, self.timeframe, start_date, end_date)
        
        if rates is None or len(rates) == 0:
            print(f"Failed to get data for {self.symbol}")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        mt5.shutdown()
        return df

# ============================================================================
# Feature Engineering
# ============================================================================

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
        
    def add_technical_indicators(self):
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']
        volume = self.df['tick_volume']
        
        # Trend Indicators
        self.df['sma_20'] = SMAIndicator(close, window=20).sma_indicator()
        self.df['sma_50'] = SMAIndicator(close, window=50).sma_indicator()
        self.df['sma_200'] = SMAIndicator(close, window=200).sma_indicator()
        self.df['ema_12'] = EMAIndicator(close, window=12).ema_indicator()
        self.df['ema_26'] = EMAIndicator(close, window=26).ema_indicator()
        
        # MACD
        macd = MACD(close)
        self.df['macd'] = macd.macd()
        self.df['macd_signal'] = macd.macd_signal()
        self.df['macd_diff'] = macd.macd_diff()
        
        # ADX
        adx = ADXIndicator(high, low, close, window=14)
        self.df['adx'] = adx.adx()
        self.df['adx_pos'] = adx.adx_pos()
        self.df['adx_neg'] = adx.adx_neg()
        
        # Momentum
        self.df['rsi'] = RSIIndicator(close, window=14).rsi()
        self.df['rsi_6'] = RSIIndicator(close, window=6).rsi()
        self.df['rsi_24'] = RSIIndicator(close, window=24).rsi()
        
        stoch = StochasticOscillator(high, low, close)
        self.df['stoch_k'] = stoch.stoch()
        self.df['stoch_d'] = stoch.stoch_signal()
        
        self.df['williams_r'] = WilliamsRIndicator(high, low, close).williams_r()
        
        # Volatility
        bb = BollingerBands(close)
        self.df['bb_high'] = bb.bollinger_hband()
        self.df['bb_low'] = bb.bollinger_lband()
        self.df['bb_mid'] = bb.bollinger_mavg()
        self.df['bb_width'] = bb.bollinger_wband()
        
        atr = AverageTrueRange(high, low, close)
        self.df['atr'] = atr.average_true_range()
        
        kc = KeltnerChannel(high, low, close)
        self.df['kc_high'] = kc.keltner_channel_hband()
        self.df['kc_low'] = kc.keltner_channel_lband()
        
        # Volume
        self.df['obv'] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        
        return self
    
    def add_price_features(self):
        self.df['returns'] = self.df['close'].pct_change()
        self.df['log_returns'] = np.log(self.df['close'] / self.df['close'].shift(1))
        self.df['price_change'] = self.df['close'] - self.df['open']
        self.df['high_low_range'] = self.df['high'] - self.df['low']
        
        for window in [5, 10, 20, 50]:
            self.df[f'rolling_mean_{window}'] = self.df['close'].rolling(window).mean()
            self.df[f'rolling_std_{window}'] = self.df['close'].rolling(window).std()
            
        self.df['dist_sma_20'] = (self.df['close'] - self.df['sma_20']) / self.df['sma_20']
        
        return self
    
    def add_time_features(self):
        self.df['hour'] = self.df.index.hour
        self.df['day_of_week'] = self.df.index.dayofweek
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        self.df['day_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['day_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)
        
        return self
    
    def add_lagged_features(self, lags=[1, 2, 3, 6, 12, 24]):
        for lag in lags:
            self.df[f'close_lag_{lag}'] = self.df['close'].shift(lag)
            self.df[f'returns_lag_{lag}'] = self.df['returns'].shift(lag)
            
        return self
    
    def get_features(self):
        self.add_technical_indicators()
        self.add_price_features()
        self.add_time_features()
        self.add_lagged_features()
        self.df.dropna(inplace=True)
        return self.df

# ============================================================================
# Dataset
# ============================================================================

class ForexDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================================
# Models
# ============================================================================

class iTransformer(nn.Module):
    def __init__(self, n_features, seq_len, pred_len, d_model=512, n_heads=8, 
                 n_layers=3, d_ff=2048, dropout=0.1):
        super(iTransformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.embedding = nn.Linear(seq_len, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, n_features, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.projection = nn.Linear(d_model, pred_len)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.embedding(x)
        x = x + self.pos_encoding
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.projection(x)
        x = x[:, 0, :]
        return x

class TSMixerBlock(nn.Module):
    def __init__(self, seq_len, n_features, expansion_factor=2, dropout=0.1):
        super(TSMixerBlock, self).__init__()
        
        self.time_norm = nn.LayerNorm(n_features)
        self.time_fc1 = nn.Linear(seq_len, seq_len * expansion_factor)
        self.time_fc2 = nn.Linear(seq_len * expansion_factor, seq_len)
        
        self.feat_norm = nn.LayerNorm(n_features)
        self.feat_fc1 = nn.Linear(n_features, n_features * expansion_factor)
        self.feat_fc2 = nn.Linear(n_features * expansion_factor, n_features)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        residual = x
        x = self.time_norm(x)
        x = x.transpose(1, 2)
        x = self.time_fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.time_fc2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = x + residual
        
        residual = x
        x = self.feat_norm(x)
        x = self.feat_fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.feat_fc2(x)
        x = self.dropout(x)
        x = x + residual
        
        return x

class TSMixer(nn.Module):
    def __init__(self, n_features, seq_len, pred_len, n_blocks=4, 
                 expansion_factor=2, dropout=0.1):
        super(TSMixer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.blocks = nn.ModuleList([
            TSMixerBlock(seq_len, n_features, expansion_factor, dropout)
            for _ in range(n_blocks)
        ])
        
        self.temporal_proj = nn.Linear(seq_len, pred_len)
        self.feature_proj = nn.Linear(n_features, 1)
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        
        x = x.transpose(1, 2)
        x = self.temporal_proj(x)
        x = x.transpose(1, 2)
        x = self.feature_proj(x)
        x = x.squeeze(-1)
        
        return x

class FrequencyAttention(nn.Module):
    def __init__(self, d_model, n_heads, modes=32):
        super(FrequencyAttention, self).__init__()
        self.modes = modes
        self.weights = nn.Parameter(torch.randn(n_heads, modes, d_model // n_heads, 2))
        
    def forward(self, q, k, v):
        B, L, H, E = q.shape
        
        q_ft = torch.fft.rfft(q, dim=1, norm='ortho')
        k_ft = torch.fft.rfft(k, dim=1, norm='ortho')
        v_ft = torch.fft.rfft(v, dim=1, norm='ortho')
        
        modes = min(self.modes, q_ft.shape[1])
        out_ft = torch.zeros_like(v_ft)
        for i in range(modes):
            out_ft[:, i] = v_ft[:, i]
        
        out = torch.fft.irfft(out_ft, n=L, dim=1, norm='ortho')
        return out

class FEDformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, modes=32, dropout=0.1):
        super(FEDformerLayer, self).__init__()
        
        self.attention = FrequencyAttention(d_model, n_heads, modes)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, L, D = x.shape
        H = 8
        
        x_reshaped = x.view(B, L, H, D // H)
        attn_out = self.attention(x_reshaped, x_reshaped, x_reshaped)
        attn_out = attn_out.reshape(B, L, D)
        
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x

class FEDformer(nn.Module):
    def __init__(self, n_features, seq_len, pred_len, d_model=512, n_heads=8,
                 n_layers=2, d_ff=2048, modes=32, dropout=0.1):
        super(FEDformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.embedding = nn.Linear(n_features, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        self.layers = nn.ModuleList([
            FEDformerLayer(d_model, n_heads, d_ff, modes, dropout)
            for _ in range(n_layers)
        ])
        
        self.projection = nn.Sequential(
            nn.Linear(d_model * seq_len, pred_len),
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_encoding
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = x.reshape(x.shape[0], -1)
        x = self.projection(x)
        
        return x

# ============================================================================
# Ensemble
# ============================================================================

class EnsembleModel:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
        
    def predict(self, x):
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        ensemble_pred = sum(w * p for w, p in zip(self.weights, predictions))
        return ensemble_pred

# ============================================================================
# Training Pipeline
# ============================================================================

class ForexPredictor:
    def __init__(self, symbol="EURUSD", seq_len=168, pred_len=24):
        self.symbol = symbol
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scaler = RobustScaler()
        self.target_scaler = RobustScaler()
        
    def prepare_data(self, df):
        feature_cols = [col for col in df.columns if col not in ['close']]
        
        features_scaled = self.scaler.fit_transform(df[feature_cols])
        target = df['close'].values.reshape(-1, 1)
        target_scaled = self.target_scaler.fit_transform(target)
        
        X, y = [], []
        for i in range(len(df) - self.seq_len - self.pred_len + 1):
            X.append(features_scaled[i:i+self.seq_len])
            y.append(target_scaled[i+self.seq_len:i+self.seq_len+self.pred_len, 0])
        
        return np.array(X), np.array(y), len(feature_cols)
    
    def train_model(self, model, train_loader, val_loader, epochs=50, lr=0.001):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'{model.__class__.__name__}_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        model.load_state_dict(torch.load(f'{model.__class__.__name__}_best.pth'))
        return model
    
    def run(self, years=10, epochs=50, batch_size=64):
        print("=" * 80)
        print("FOREX ML PREDICTION SYSTEM")
        print("=" * 80)
        
        # Extract data
        print(f"\n1. Extracting {years} years of {self.symbol} data from MT5...")
        extractor = MT5DataExtractor(self.symbol)
        df = extractor.get_historical_data(years)
        
        if df is None:
            print("Failed to extract data. Please ensure MT5 is running and logged in.")
            return None, None, None
        
        print(f"   Data shape: {df.shape}")
        
        # Engineer features
        print("\n2. Engineering features...")
        engineer = FeatureEngineer(df)
        df_features = engineer.get_features()
        print(f"   Total features: {len(df_features.columns)}")
        
        # Prepare data
        print("\n3. Preparing sequences...")
        X, y, n_features = self.prepare_data(df_features)
        print(f"   X shape: {X.shape}, y shape: {y.shape}")
        
        # Split data
        train_size = int(0.8 * len(X))
        val_size = int(0.1 * len(X))
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
        X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
        
        train_dataset = ForexDataset(X_train, y_train)
        val_dataset = ForexDataset(X_val, y_val)
        test_dataset = ForexDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize models
        print("\n4. Initializing models...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Using device: {device}")
        
        itransformer = iTransformer(n_features, self.seq_len, self.pred_len, 
                                    d_model=256, n_heads=8, n_layers=3)
        tsmixer = TSMixer(n_features, self.seq_len, self.pred_len, n_blocks=4)
        fedformer = FEDformer(n_features, self.seq_len, self.pred_len, 
                             d_model=256, n_heads=8, n_layers=2)
        
        # Train models
        print("\n5. Training models...")
        
        print("\n   Training iTransformer...")
        itransformer = self.train_model(itransformer, train_loader, val_loader, epochs)
        
        print("\n   Training TSMixer...")
        tsmixer = self.train_model(tsmixer, train_loader, val_loader, epochs)
        
        print("\n   Training FEDformer...")
        fedformer = self.train_model(fedformer, train_loader, val_loader, epochs)
        
        # Evaluate
        print("\n6. Evaluating ensemble...")
        ensemble = EnsembleModel([itransformer, tsmixer, fedformer])
        
        test_loss = 0
        all_predictions = []
        all_actuals = []
        
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = ensemble.predict(X_batch)
            
            y_pred_cpu = y_pred.cpu().numpy()
            y_batch_cpu = y_batch.numpy()
            
            y_pred_original = self.target_scaler.inverse_transform(
                y_pred_cpu.reshape(-1, 1)
            ).reshape(-1, self.pred_len)
            
            y_actual_original = self.target_scaler.inverse_transform(
                y_batch_cpu.reshape(-1, 1)
            ).reshape(-1, self.pred_len)
            
            all_predictions.extend(y_pred_original)
            all_actuals.extend(y_actual_original)
            
            mse = np.mean((y_pred_original - y_actual_original) ** 2)
            test_loss += mse
        
        test_loss /= len(test_loader)
        rmse = np.sqrt(test_loss)
        
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        
        mae = np.mean(np.abs(all_predictions - all_actuals))
        mape = np.mean(np.abs((all_actuals - all_predictions) / all_actuals)) * 100
        
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"Test RMSE: {rmse:.6f}")
        print(f"Test MAE:  {mae:.6f}")
        print(f"Test MAPE: {mape:.2f}%")
        
        # Save scalers
        import joblib
        joblib.dump(self.scaler, 'feature_scaler.pkl')
        joblib.dump(self.target_scaler, 'target_scaler.pkl')
        
        print("\nModels and scalers saved successfully!")
        
        return ensemble, self.scaler, self.target_scaler

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    predictor = ForexPredictor(
        symbol="EURUSD",
        seq_len=168,
        pred_len=24
    )
    
    ensemble, feature_scaler, target_scaler = predictor.run(
        years=10,
        epochs=50,
        batch_size=64
    )