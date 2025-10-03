# GGTH AI Trading System - Usage Instructions
# Copyright 2025 Jason.w.rusk@gmail.com

Complete guide for using the Python prediction generator, MT5 indicator, and Expert Advisor together.

---

## Initial Setup (One-Time Configuration)

### Step 1: Install Python Components

Open Command Prompt and install required packages:

```bash
pip install tensorflow MetaTrader5 pandas numpy scikit-learn matplotlib
```

**For Mac users (M1/M2/M3):**
```bash
pip install tensorflow-macos tensorflow-metal MetaTrader5 pandas numpy scikit-learn matplotlib
```

### Step 2: Configure Python Script

Edit `predictor3.py` and set your MT5 Files path:

```python
HARDCODED_MT5_FILES_PATH = r"C:\Users\YOUR_USERNAME\AppData\Roaming\MetaQuotes\Terminal\YOUR_ID\MQL5\Files"
```

**To find your path:**
1. Open MT5
2. File → Open Data Folder
3. Navigate to MQL5 → Files
4. Copy the full path from address bar
5. Paste into script

### Step 3: Install MT5 Components

**Install Indicator:**
1. Copy `Predictor3.mq5` to: `MQL5\Indicators\`
2. Open MetaEditor (F4 in MT5)
3. Open the indicator file
4. Click Compile (F7)
5. Verify no errors

**Install Expert Advisor:**
1. Copy `GGTH EA v1.3.mq5` to: `MQL5\Experts\`
2. Open MetaEditor
3. Open the EA file
4. Click Compile (F7)
5. Verify no errors

### Step 4: Prepare MT5 Historical Data

**Critical for accuracy:**
1. Open MT5
2. Go to: Tools → Options → Charts
3. Set: Max bars in history = 999999
4. Set: Max bars in chart = 999999
5. Click OK

**Load history for your symbol:**
1. Open EURUSD chart (or your symbol)
2. Set timeframe to H1
3. Press Home key
4. Scroll left to load more data
5. Wait 1-2 minutes
6. You should see 5+ years of data

---

## Daily Workflow

### Scenario 1: Live Trading with Continuous Predictions

Use this workflow when actively trading with the EA.

**Morning Setup (One Time):**

1. **Start Python Prediction Service**

Open Command Prompt:
make sure your in the folder that predictor3 is in then run this
python predictor3.py EURUSD continuous 60


This will:
- Update predictions every 60 minutes
- Run continuously until you stop it
- Save predictions to MQL5 Files folder

**Leave this window open all day.**

2. **Attach Indicator to Chart**

In MT5:
- Open EURUSD chart
- Set timeframe to H1
- Navigator → Indicators → Custom → Predictor3
- Drag onto chart
- Click OK

You'll see:
- Service status (Online/Offline)
- Current predictions for all timeframes
- Real-time accuracy tracking
- Color-coded confidence levels

3. **Attach EA to Chart**

In MT5:
- Same chart (EURUSD H1)
- Navigator → Expert Advisors → GGTH predictor
- Drag onto chart
- Configure parameters (see Configuration section)
- Enable Algo Trading button (top toolbar)
- Click OK

---

### Scenario 2: Backtesting Strategy

Use this to test your strategy on historical data before live trading.

**Generate Historical Predictions:**

1. **Make sure MT5 is running and logged in**

2. **Run Python in Backtest Mode**

Open Command Prompt:
```bash
cd C:\Trading
python predictor3.py backtest EURUSD
```

What happens:
- Downloads 5 years of H1 data
- Creates 40+ technical indicators
- Trains LSTM model (20-60 min first time)
- Generates predictions for every bar
- Exports CSV files to MQL5 Files folder

**Wait for completion** (30-90 minutes first run)

Output files created:
```
EURUSD_1H_lookup.csv
EURUSD_4H_lookup.csv
EURUSD_1D_lookup.csv
EURUSD_5D_lookup.csv
```

3. **Run Strategy Tester**

In MT5:
- Press Ctrl+R (or View → Strategy Tester)
- Expert: Select "GGTH EA v1.3"
- Symbol: EURUSD (must match your predictions)
- Period: H1 (chart timeframe)
- Date: 2021.01.01 to 2024.12.31
- Mode: Every tick

4. **Configure EA Parameters**

Click "Expert properties":

```
Backtesting Mode:
  UsePredictionFile = true
  PredictionTimeframe = "4H"
  SimulationMode = false

Risk Management:
  RiskPercent = 1.0
  MaxDailyLoss = 5.0
  MaxOpenTrades = 3

Trading Strategy:
  MinConfidence = 60.0
  StopLossPips = 50
  TakeProfitPips = 100
  UsePredictionAsTP = true
```

5. **Start Backtest**

Click Start button.

Monitor progress bar at bottom.

6. **Analyze Results**

After completion:

**Results Tab:**
- View every trade
- Entry/exit prices
- Profit/loss per trade

**Graph Tab:**
- Balance curve
- Drawdown visualization
- Should show steady growth

**Report Tab:**
```
Total Trades: 125
Win Rate: 62.4%
Profit Factor: 1.85
Max Drawdown: 5.7%
```

**Good results:**
- Win rate above 55%
- Profit factor above 1.5
- Max drawdown below 20%
- Consistent equity curve

**Red flags:**
- Win rate below 45%
- Profit factor below 1.2
- Drawdown above 30%
- Erratic equity curve

---

### Scenario 3: Forward Testing (Paper Trading)

After successful backtest, test on demo account with live predictions.

**Setup:**

1. **Open Demo Account** (if you don't have one)
   - File → Open an Account
   - Select broker
   - Choose "Demo Account"
   - Balance: $10,000

2. **Start Python Service**
```bash
python predictor3.py EURUSD continuous 60
```

3. **Attach Indicator**
   - Shows live predictions on chart
   - Monitor accuracy in real-time

4. **Attach EA**
   - Use conservative settings initially:
   ```
   RiskPercent = 0.5
   MaxDailyLoss = 3.0
   MaxOpenTrades = 2
   MinConfidence = 65.0
   ```

5. **Monitor for 2-4 Weeks**
   - Check daily
   - Track performance
   - Compare to backtest
   - Adjust if needed

**Only proceed to live trading if:**
- Demo results similar to backtest
- Win rate consistent
- You understand every trade
- No emotional stress

---

## Configuration Reference

### Python Script Configuration

Edit these at the top of `predictor3.py`:

```python
SYMBOL = 'EURUSD'           # Trading pair
TIMEFRAME = '1H'            # Data timeframe
DAYS = 2000                 # History days (5+ years)
EPOCHS = 100                # Training iterations
```

**Common symbols:**
- EURUSD, GBPUSD, USDJPY, AUDUSD
- Check exact name in MT5 Market Watch
- May have suffix: EURUSDm, EURUSD.a

### Indicator Settings

When attaching to chart:

```
EnableDebug = true          # Show detailed logs
UpdateInterval = 30         # File check frequency (seconds)
AccuracyCheckInterval = 300 # Accuracy update (seconds)
ShowDetailedStats = true    # Show hit/miss counts
DisplayXOffset = 20         # Horizontal position
DisplayYOffset = 20         # Vertical position
```

### EA Settings

**For Backtesting:**
```
UsePredictionFile = true
PredictionTimeframe = "4H"
SimulationMode = false
RiskPercent = 1.0
MinConfidence = 60.0
```

**For Live Trading (Conservative):**
```
UsePredictionFile = true
PredictionTimeframe = "4H"
RiskPercent = 0.5
MaxDailyLoss = 3.0
MaxOpenTrades = 2
MinConfidence = 65.0
StopLossPips = 50
TakeProfitPips = 100
UsePredictionAsTP = true
UseTrailingStop = true
```

**For Aggressive Trading:**
```
RiskPercent = 2.0
MaxDailyLoss = 5.0
MaxOpenTrades = 5
MinConfidence = 55.0
```

---

## File Locations and Data Flow

### Data Flow Diagram

```
Python Script (predictor3.py)
    |
    | Generates
    v
predictions_EURUSD.json -----> Read by Indicator
    |                            |
    |                            | Displays on chart
    |                            | Tracks accuracy
    |                            v
    |                      accuracy_EURUSD_indicator.csv
    |
    +--------------------> Read by EA
                              |
                              | Uses for trading
                              | Tracks accuracy
                              v
                         accuracy_EURUSD_12345.csv
```

### Key Files

**Input Files (in MQL5/Files/):**
```
predictions_EURUSD.json      - Current predictions from Python
EURUSD_4H_lookup.csv         - Historical predictions (for backtest)
```

**Output Files (in MQL5/Files/):**
```
prediction_log_EURUSD.csv              - Prediction history log
accuracy_EURUSD_indicator.csv          - Indicator accuracy tracking
accuracy_EURUSD_12345.csv              - EA accuracy (by magic number)
```

**Python Files (in same folder as script):**
```
model_EURUSD.h5                        - Trained LSTM model
feature_scaler_EURUSD.pkl              - Feature scaling parameters
target_scaler_EURUSD.pkl               - Target scaling parameters
EURUSD_historical_1H.csv               - Downloaded data
```

---

## Common Tasks

### Task: Update Predictions Manually

If Python script not running continuously:

```bash
# Generate single prediction update
python predictor3.py EURUSD

# This runs once and exits
```

Files updated:
- predictions_EURUSD.json

Indicator will pick up changes within 30 seconds (UpdateInterval).

### Task: Retrain Model with Latest Data

Every month or after significant market changes:

```bash
# Delete old model files
del model_EURUSD.h5
del feature_scaler_EURUSD.pkl
del target_scaler_EURUSD.pkl

# Run backtest mode (will retrain)
python predictor3.py backtest EURUSD
```

This downloads fresh data and trains new model.

### Task: Test Different Timeframe

**In Python (backtest):**
Already generates all timeframes (1H, 4H, 1D, 5D).

**In EA:**
Change `PredictionTimeframe` parameter:
- "1H" - More trades, shorter holds
- "4H" - Balanced (recommended)
- "1D" - Fewer trades, longer holds
- "5D" - Very few trades, swing trading

**In Indicator:**
Shows all timeframes automatically.

### Task: Run Multiple Symbols

**Option 1: Multiple Python Instances**

Open separate Command Prompt windows:

```bash
# Window 1
python predictor3.py EURUSD continuous 60

# Window 2
python predictor3.py GBPUSD continuous 60
```

**Option 2: Sequential Predictions**

Create batch file:
```batch
python predictor3.py EURUSD
python predictor3.py GBPUSD
python predictor3.py USDJPY
```

Run every hour with Task Scheduler.

### Task: Check Prediction Accuracy

**From Indicator:**
Look at display on chart:
```
Confidence: 72.3% | Real Acc: 74.1% (20/27)
                     ^        ^      ^
                     |        |      |
              Real accuracy  Hits  Total
```

**From Files:**

View `prediction_log_EURUSD.csv`:
```csv
4H,1696262400,1.08234,1.08456,1696276800,Hit
1H,1696262400,1.08234,1.08345,1696266000,Miss
```

Count Hit vs Miss for each timeframe.

**From EA:**
Check log in OnDeinit output:
```
Prediction Accuracy:
  1H: 65.0% (13/20)
  4H: 72.0% (18/25)
```

### Task: Stop Everything Safely

1. **Stop Python Script**
   - Go to Command Prompt window
   - Press Ctrl+C
   - Wait for "Shutdown complete"

2. **Remove EA from Chart**
   - Right-click chart
   - Expert Advisors → Remove
   - Or close MT5

3. **Remove Indicator**
   - Right-click chart
   - Indicators List
   - Select GGTH Predict v4.1
   - Delete

Files and accuracy data are saved automatically.


## Performance Expectations



**Prediction Accuracy:**
- 60-75% direction accuracy is excellent
- 50-60% is average
- Below 50% needs investigation

**Trading Results:**
- 5-15% annual return is realistic
- 55-65% win rate is good
- 10-20% max drawdown expected

### When to Adjust

Consider adjusting when:
- Win rate 45-50% for 2+ weeks
- Drawdown 10-15%
- Confidence levels consistently low
- Market conditions change dramatically

**Adjustments to try:**
- Change MinConfidence threshold
- Adjust stop loss / take profit
- Switch timeframe
- Retrain model with recent data
- Reduce position size

---

1. Retrain AI model with latest data
2. Run fresh backtests
3. Compare live vs backtest results
4. Review all parameters
5. Update software if needed

## Summary: Quick Start

**Absolute minimum to get running:**

1. Install Python packages
2. Edit MT5 path in predictor3.py
3. Compile indicator and EA in MT5
4. Run: `python predictor3.py backtest EURUSD`
5. Wait for completion
6. Open Strategy Tester
7. Load EA, configure for file mode
8. Run backtest
9. Review results
10. If good, try demo account

**For live trading:**

1. Run: `python predictor3.py EURUSD continuous 60`
2. Attach indicator to chart
3. Attach EA to chart
4. Enable algo trading
5. Monitor closely

**Remember:**
- Start small
- Use demo first
- Track everything
- Stay disciplined
- Manage risk

---

This system provides sophisticated AI-powered trading capabilities, but success requires proper setup, testing, and ongoing monitoring. Take time to understand each component before risking real money.
