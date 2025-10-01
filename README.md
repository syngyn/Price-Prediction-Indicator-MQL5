GGTH AI Prediction & Backtesting System
Complete guide for generating ML predictions and backtesting your EA with real prediction data.
📋 Table of Contents

System Overview
Installation
Step-by-Step Workflow
Python Script Usage
EA Backtesting
Troubleshooting


System Overview
This system consists of two components:

Python Prediction Generator - Trains LSTM models and generates predictions
MT5 Expert Advisor - Uses predictions for backtesting and live trading

How It Works
Historical Data → Python ML Model → Predictions File → MT5 EA → Backtest Results

Installation
Python Requirements
bash# Install required packages
pip install tensorflow numpy pandas scikit-learn matplotlib MetaTrader5

# Or install individually:
pip install tensorflow        # Deep learning framework
pip install numpy pandas      # Data manipulation
pip install scikit-learn      # ML utilities
pip install matplotlib        # Plotting
pip install MetaTrader5       # MT5 integration for data download
Note: MetaTrader5 package requires MT5 terminal to be installed and running.
MT5 Setup

Place the EA in: MQL5/Experts/
Place prediction files in: MQL5/Files/
Compile the EA in MetaEditor


Step-by-Step Workflow
Step 1: Get Historical Data
Option A: Auto-download from MT5 (Recommended)

Make sure MetaTrader 5 is running and logged in
The Python script will automatically download data
Configure symbol and timeframe in the script

Option B: Manual Export from MT5

Open MT5 → Select symbol (e.g., EURUSD)
Open Data Window (Ctrl+D)
Right-click → Export to CSV
Include: Timestamp, Open, High, Low, Close, Volume

Important: Ensure you have enough historical data loaded in MT5. Go to Tools → Options → Charts → Max bars in history (set to 999999).
Step 2: Run Python Script
bashpython backtest3.py
The script will:

Load your historical data
Create technical indicators
Train LSTM models for each timeframe (1H, 4H, 1D, 5D)
Generate predictions
Export files for MT5

Output Files:
predictions/
├── predictions_EURUSD.json          # Summary with accuracies
├── EURUSD_1H_predictions.csv        # Detailed 1H predictions
├── EURUSD_4H_predictions.csv        # Detailed 4H predictions
├── EURUSD_1D_predictions.csv        # Detailed 1D predictions
├── EURUSD_5D_predictions.csv        # Detailed 5D predictions
├── EURUSD_1H_lookup.csv             # EA lookup file
├── EURUSD_4H_lookup.csv             # EA lookup file
├── EURUSD_1D_lookup.csv             # EA lookup file
└── EURUSD_5D_lookup.csv             # EA lookup file
Step 3: Copy Files to MT5
bash# Copy lookup files to MT5
cp predictions/*_lookup.csv C:/Users/YourName/AppData/Roaming/MetaQuotes/Terminal/XXXXX/MQL5/Files/
Or manually copy:

Source: predictions/ folder
Destination: MT5 Data Folder → MQL5 → Files

Step 4: Run Backtest

Open MT5 Strategy Tester (Ctrl+R)
Select the EA: GGTH Predictorv11
Set parameters:

UsePredictionFile = true
PredictionTimeframe = "4H" (or whichever you want to test)
Configure risk and strategy settings


Choose backtest period (must overlap with prediction dates)
Click Start


Python Script Usage
Configuration Options
python# In the script's __main__ section:

SYMBOL = 'EURUSD'                    # Trading symbol (check MT5 for exact name)
DATA_FILE = None                     # Set to CSV path, or None for auto-download
TIMEFRAME = '1H'                     # 1M, 5M, 15M, 30M, 1H, 4H, 1D, 1W
DAYS = 730                           # Days of history to download (730 = 2 years)
EPOCHS = 30                          # Training epochs (30-100)
Important Symbol Names:

Your broker may use different symbol names
Common variations: 'EURUSD', 'EURUSDm', 'EURUSD.a', 'EURUSD.raw'
Check Market Watch in MT5 for exact symbol name
The script will tell you if symbol not found

Custom Data Format
Your CSV should have these columns:
timestamp,open,high,low,close,volume
2024-01-01 00:00,1.1050,1.1055,1.1045,1.1052,1000
2024-01-01 01:00,1.1052,1.1060,1.1050,1.1058,1200
...
Training Configuration
python# Adjust these in the script:
lookback = 60        # How many bars to look back
epochs = 50          # Training iterations
batch_size = 32      # Training batch size
Understanding the Output
Results for 4H:
MAE: 0.00123         # Mean Absolute Error (lower = better)
MAPE: 2.45%          # Mean Absolute Percentage Error
Direction Accuracy: 72.5%  # % of correct direction predictions

EA Backtesting
EA Input Parameters
Backtesting Mode

UsePredictionFile = true - Use Python-generated predictions
SimulationMode = false - Fallback simulation mode
SimAccuracy = 70.0 - Simulation accuracy (if SimulationMode = true)

Indicator Settings

IndicatorName = "Predictor3" - Custom indicator name
PredictionTimeframe = "4H" - Which timeframe to use (1H/4H/1D/5D)
UseIndicatorAccuracy = true - Use indicator's accuracy

Risk Management

RiskPercent = 1.0 - Risk per trade (% of balance)
MaxDailyLoss = 5.0 - Stop trading after this daily loss
MaxOpenTrades = 3 - Maximum concurrent positions
UseFixedLot = false - Use fixed lot size
FixedLotSize = 0.01 - If UseFixedLot = true

Trading Strategy

MinConfidence = 60.0 - Minimum confidence threshold
StopLossPips = 50 - Stop loss distance
TakeProfitPips = 100 - Take profit distance
UsePredictionAsTP = true - Use predicted price as TP
UseTrailingStop = true - Enable trailing stop
TrailingStopPips = 30 - Trailing stop distance
TrailingStepPips = 10 - Trailing stop step

Backtesting Modes
The EA has 3 modes:

File Mode (Recommended) - Uses Python predictions

Set: UsePredictionFile = true
Most accurate for backtesting


Indicator Mode - Uses custom indicator

Set: UsePredictionFile = false, SimulationMode = false
Requires indicator to be installed


Simulation Mode - Fallback testing

Set: SimulationMode = true
Uses look-ahead for quick testing



Reading Backtest Results
The EA outputs detailed statistics:
═══════════════════════════════════════════════
  BACKTEST RESULTS
═══════════════════════════════════════════════
Total Trades: 125
Winning Trades: 78
Losing Trades: 47
Win Rate: 62.40%
Total Profit: $5,234.50
Total Loss: $-2,145.30
Net P&L: $3,089.20
Max Drawdown: $567.80
═══════════════════════════════════════════════
Strategy Tester Settings
Recommended settings:
Model: Every tick (most accurate)
Optimization: Use custom criterion (OnTester)
Period: M15 or H1
Dates: Match your prediction file dates
Visualization: Off (faster)

Troubleshooting
Python Issues
"MetaTrader5 not installed"
bashpip install MetaTrader5
"Failed to initialize MT5"

Make sure MetaTrader 5 is running
Make sure you're logged into a trading account
Try running Python script as Administrator
Check MT5 is not already in use by another script

"Failed to get data for EURUSD"

Check exact symbol name in MT5 Market Watch
Try variations: 'EURUSDm', 'EURUSD.a', 'EURUSD.raw'
Right-click in Market Watch → Show All
Select symbol → Right-click → Chart Window
Verify you have historical data: Tools → Options → Charts → Max bars in history

"TensorFlow not found"
bashpip install tensorflow
# Or for Apple Silicon Mac:
pip install tensorflow-macos tensorflow-metal
"Downloaded 0 bars"

Symbol doesn't exist or wrong name
Not enough historical data in MT5
Download more history: Tools → Options → Charts → Max bars = 999999
May need to scroll chart backwards to load history

"Model training slow"

Reduce EPOCHS (try 20-30)
Reduce DAYS (try 365 instead of 730)
Use GPU acceleration if available
Close other applications

EA Issues
"Could not open prediction file"

Check file is in MQL5/Files/ folder
File must be named: SYMBOL_TIMEFRAME_lookup.csv
Example: EURUSD_4H_lookup.csv

"No prediction data"

Ensure backtest dates overlap with prediction dates
Check prediction file has data for selected timeframe
Verify file format (timestamp,prediction)

"No trades executed"

Check MinConfidence - may be too high
Verify predictions exist for test period
Check risk settings (lot size may be too small)

"Indicator not loaded"

Switch to UsePredictionFile = true
Or install Predictor3.ex5 indicator
Or use SimulationMode = true

File Format Issues
Prediction file format:
2024.01.01 00:00,1.10523
2024.01.01 04:00,1.10589
2024.01.01 08:00,1.10634
Data file format:
csvtimestamp,open,high,low,close,volume
2024-01-01 00:00,1.1050,1.1055,1.1045,1.1052,1000

Advanced Usage
Optimization
Optimize these parameters in Strategy Tester:

RiskPercent (0.5 - 3.0)
MinConfidence (50 - 80)
StopLossPips (30 - 100)
TakeProfitPips (50 - 200)

Multiple Timeframes
Test different timeframes:

Generate predictions for all timeframes (Python does this automatically)
Run separate backtests for each timeframe
Compare results to find best performing timeframe

Model Improvement
Improve prediction accuracy:

Add more historical data (2+ years)
Increase training epochs (50-100)
Add more features in create_features()
Tune LSTM architecture
Use ensemble models
