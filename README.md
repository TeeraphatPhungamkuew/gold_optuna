# XAUUSD Quantitative Trading Pipeline (v3)

A robust, machine learning-driven algorithmic trading pipeline for Gold (XAUUSD / GC=F). This project leverages an Ensemble Model (Deep Neural Networks + XGBoost) with a Multi-Timeframe Analysis (MTFA) approach, heavily optimized using Bayesian Optimization (Optuna).

## Core Architecture

This bot is designed with a strict quantitative research framework to prevent overfitting and look-ahead bias:

* **Multi-Timeframe Analysis (MTFA):** Integrates Daily trend filters (SMA 200, EMA 50) with Hourly execution signals (RSI, MACD, ATR, Volatility Ratio) completely leakage-free.
* **Soft Voting Ensemble Engine:** Combines the probabilistic outputs of a PyTorch Deep Neural Network (DNN) and an XGBoost Classifier. Trades are executed only when the average confidence `(P_dnn + P_xgb) / 2` exceeds the dynamic threshold, balancing aggressive entry with smart filtering.
* **Bayesian Hyperparameter Optimization (Optuna):** Automatically searches for the optimal risk-reward parameters (Take Profit, Stop Loss, Trailing ATR, Threshold) to maximize **Net P&L** while penalizing zero-trade outputs.
* **Dynamic Risk Management:** Implements a volatility-adjusted ATR Trailing Stop to lock in profits during strong trends while cutting losses early.

## Three-Partition Execution Flow (Walk-Forward Validation)

To ensure the model performs well in live markets, the dataset is strictly partitioned into three chronological slices:

    ├─────── TRAIN (70%) ──────────┤── VAL (15%) ──┤── TEST (15%) ──┤

1. **Phase 1 (Optimization):** Models are trained on `TRAIN` and evaluated on `VAL` for 30 Optuna trials.
2. **Phase 2 (Re-training):** The best parameters are injected, and models are re-trained on the combined `TRAIN + VAL` dataset.
3. **Phase 3 (Final Evaluation):** The strategy is simulated on the strictly sequestered `TEST` partition (Unseen Data) to report unbiased performance.

## Performance Summary (Out-of-Sample Test)
*Tested on strictly unseen data: 2025-04-28 to 2025-12-29 (8 Months)*

* **Total Trades:** 61
* **Win Rate:** 55.7%
* **Net P&L:** +$12,308.00 
* **Profit Factor:** 1.441
* **Average Win:** +$1,182.62
* **Average Loss:** -$1,033.37
* **Reward/Risk Ratio:** 1.14

## Getting Started

### Prerequisites
Make sure you have Python 3.10+ installed. Install the required dependencies:

    pip install torch xgboost optuna pandas numpy scikit-learn yfinance

### Data Requirements
Place your 1-hour historical data file named `XAU_1h_data.csv` in the root directory. The daily data will be automatically fetched via `yfinance`.

### Running the Pipeline
Execute the main optimization and backtesting script:

    python gold_optuna.py

The script will run the Optuna study, output the best parameters, and generate the final trade log in `gold_optuna_backtest.csv` and `gold_ensemble_v3_final.csv`.

## 🗺️ Roadmap
- [x] Baseline Strategy & Feature Engineering
- [x] Ensemble Modeling (DNN + XGBoost)
- [x] Bayesian Optimization with Optuna
- [x] Switch to Aggressive Soft-Voting & Net P&L Maximization
- [ ] **Next:** Forward Testing / Paper Trading integration via Broker API (e.g., MetaTrader 5)
- [ ] **Future:** Smart Contract / Vault integration for DeFi Quantitative Strategy execution.

---
*Developed for Quantitative Research & Algorithmic Trading.*
