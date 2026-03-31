"""
================================================================================
  GOLD (XAUUSD / GC=F) ALGORITHMIC TRADING PIPELINE  โ”€โ”€ v3: OPTUNA EDITION
  Strategy : Multi-Timeframe Analysis (MTFA)
             + DNN โ— XGBoost  "Soft Voting" Ensemble
             + Optuna Bayesian Hyperparameter Optimisation
             + ATR Trailing Stop Execution

  Three-Partition Execution Flow
  โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
  The dataset is divided ONCE into three non-overlapping chronological slices:

      โ”โ”€โ”€โ”€โ”€โ”€โ”€โ”€ TRAIN (70%) โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”คโ”€โ”€ VAL (15%) โ”€โ”€โ”คโ”€โ”€ TEST (15%) โ”€โ”€โ”ค

  Phase 1 โ€” Optimisation  (Optuna, n_trials):
    โ€ข Models are trained on TRAIN only.
    โ€ข Objective is evaluated on VAL only.
    โ€ข TEST is completely invisible โ€” never touched.
    โ€ข Prevents the optimiser from over-fitting to test-set noise.

  Phase 2 โ€” Final Evaluation  (one run with best params):
    โ€ข Models are re-trained on TRAIN + VAL combined ("walk-forward expand").
    โ€ข Final simulation and all reported metrics run on TEST only.
    โ€ข This gives an unbiased estimate of live performance.

  Why this matters
  โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
  If we optimised directly on the test set โ€” even indirectly by inspecting
  test results โ€” we would be fitting to noise and reporting inflated metrics.
  The val-set firewall during Phase 1 guarantees that the chosen parameters
  generalised beyond training data before we ever look at test performance.

  Architecture Overview
  โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
  โ”โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”
  โ”  ยง0  Config              โ€” all hyperparameters (DNN + XGB + sim)         โ”
  โ”  ยง1  DataIngestion       โ€” yfinance Daily + local CSV Hourly             โ”
  โ”  ยง2  FeatureEngineer     โ€” MTFA indicators, leakage-free merge           โ”
  โ”  ยง3  TargetLabeler       โ€” meta-labeling (TP/SL binary classification)   โ”
  โ”  ยง4  GoldDNN             โ€” PyTorch: Linearโ’BNโ’ReLUโ’Dropout              โ”
  โ”  ยง5  ModelTrainer        โ€” chronological split ยท dual training ยท eval    โ”
  โ”  ยง6  TradingSimulator    โ€” macro filter โ’ soft-voting gate โ’ trailing stop      โ”
  โ”  ยง7  GoldTradingPipeline โ€” data prep + feature/label pipeline            โ”
  โ”  ยง8  OptunaOptimiser     โ€” objective fn ยท study ยท best-param injection   โ”
  โ””โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”

  Dependencies:
      pip install yfinance torch scikit-learn xgboost optuna pandas numpy

  Author  : Senior Quant / AI-ML Engineer (template)
  Version : 3.0  (Optuna Bayesian optimisation added)
================================================================================
"""

# โ”€โ”€ Standard library โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
import copy
import sys
import warnings
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# โ”€โ”€ Third-party โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
import numpy as np
import pandas as pd
import yfinance as yf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)   # Suppress Optuna spam
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)


# โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•
#  ยง0  GLOBAL CONFIGURATION
# โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•

@dataclass
class Config:
    """
    Single source of truth for every tuneable knob in the pipeline.

    Fields marked  [OPTUNA]  are overwritten by OptunaOptimiser.apply_params()
    after the study completes.  All other fields remain fixed across trials.
    """

    # โ”€โ”€ Data โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
    ticker: str = "GC=F"
    daily_start: str = "2015-01-01"
    hourly_start: str = "2022-01-01"
    end_date: Optional[str] = None
    hourly_csv_candidates: list = field(default_factory=lambda: [
        "xauusd_1h.csv",
        "XAU_1h_data.csv",
        "XAUUSD_1h.csv",
    ])
    daily_csv_candidates: list = field(default_factory=lambda: [
        "xauusd_1d.csv",
        "XAU_1d_data.csv",
        "XAUUSD_1d.csv",
    ])
    daily_from_hourly_fallback: bool = True

    # โ”€โ”€ Technical indicators (fixed โ€” not tuned by Optuna) โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
    ema_fast: int = 50
    sma_slow: int = 200
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14

    # โ”€โ”€ Meta-labeling  [OPTUNA] โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
    tp_multiplier: float = 3.0      # Optuna range : [1.5, 5.0]
    sl_multiplier: float = 1.5      # Optuna range : [0.5, 2.5]
    label_horizon: int   = 72       # Optuna range : int [24, 120]

    # โ”€โ”€ DNN hyper-parameters (fixed โ€” fast trials matter more) โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
    hidden_sizes: list = field(default_factory=lambda: [256, 128, 64])
    dropout_rate: float = 0.3
    dnn_learning_rate: float = 1e-3
    batch_size: int = 512
    epochs: int = 30               # Reduced for trial speed; final run uses 50
    final_epochs: int = 50         # Used only in the final re-train
    early_stopping_patience: int = 7

    # โ”€โ”€ XGBoost hyper-parameters (fixed) โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
    xgb_n_estimators: int = 400
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.80
    xgb_colsample_bytree: float = 0.80
    xgb_min_child_weight: int = 5
    xgb_gamma: float = 1.0
    xgb_reg_alpha: float = 0.1
    xgb_reg_lambda: float = 1.0
    xgb_early_stopping_rounds: int = 30

    # โ”€โ”€ Chronological split โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
    train_ratio: float = 0.70      # TRAIN  = rows [0   โ€ฆ t1)
    val_ratio:   float = 0.15      # VAL    = rows [t1  โ€ฆ t2)  โ Optuna target
                                   # TEST   = rows [t2  โ€ฆ end) โ final eval only

    # โ”€โ”€ Ensemble soft-voting gate  [OPTUNA] โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
    signal_threshold: float = 0.55  # Optuna range : [0.50, 0.75]

    # โ”€โ”€ Execution & risk  [OPTUNA] โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
    trailing_atr_mult: float = 2.0  # Optuna range : [1.0, 3.5]
    initial_capital: float = 100_000
    risk_per_trade_pct: float = 0.01

    # โ”€โ”€ Optuna study โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
    n_trials: int = 30             # Number of Bayesian search trials
    optuna_seed: int = 42          # Reproducible TPE sampler

    # โ”€โ”€ Misc โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()
torch.manual_seed(CFG.random_seed)
np.random.seed(CFG.random_seed)


# โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•
#  ยง1  DATA INGESTION
# โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•

class DataIngestion:
    """
    Downloads Daily OHLCV via yfinance and loads Hourly from local CSV.

    Fallback order:
    1) yfinance daily
    2) local daily CSV
    3) resample daily bars from local hourly CSV
    """

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self._hourly_cache: Optional[pd.DataFrame] = None

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        cleaned = {
            c: c.strip().lower().replace("#", "").replace(" ", "_")
            for c in df.columns
        }
        d = df.rename(columns=cleaned).copy()
        mapping = {
            "datetime": "Datetime",
            "date": "Datetime",
            "time": "Datetime",
            "timestamp": "Datetime",
            "local_time": "Datetime",
            "gmt_time": "Datetime",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "tick_volume": "Volume",
            "vol": "Volume",
        }
        for src, dst in mapping.items():
            if src in d.columns:
                d.rename(columns={src: dst}, inplace=True)
        return d

    @staticmethod
    def _try_parse_datetime(series: pd.Series) -> pd.Series:
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().mean() < 0.8:
            parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
        return parsed

    @staticmethod
    def _read_csv_auto(path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path, sep=None, engine="python")
        except Exception:
            return pd.read_csv(path, sep=";")

    def _find_existing_file(self, candidates, hint_tokens, allow_any: bool = True) -> Path:
        for name in candidates:
            p = Path(name)
            if p.exists() and p.is_file():
                return p

        csv_files = list(Path(".").glob("*.csv"))
        for p in csv_files:
            name = p.name.lower()
            if any(token in name for token in hint_tokens):
                return p

        if allow_any and csv_files:
            return csv_files[0]

        raise FileNotFoundError("No CSV files found in current working directory.")

    def _prepare_ohlcv(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        d = self._normalise_columns(raw_df)

        if "Datetime" not in d.columns:
            for col in d.columns:
                parsed = self._try_parse_datetime(d[col])
                if parsed.notna().mean() > 0.8:
                    d["Datetime"] = parsed
                    break
        else:
            d["Datetime"] = self._try_parse_datetime(d["Datetime"])

        required = ["Datetime", "Open", "High", "Low", "Close"]
        missing = [c for c in required if c not in d.columns]
        if missing:
            raise ValueError(
                f"Missing required OHLC columns: {missing}. "
                f"Available columns: {d.columns.tolist()}"
            )

        d = d.dropna(subset=["Datetime"]).copy()
        d.set_index("Datetime", inplace=True)
        if d.index.tz is not None:
            d.index = d.index.tz_localize(None)
        d.sort_index(inplace=True)
        d = d[~d.index.duplicated(keep="last")]

        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")

        d = d.dropna(subset=["Open", "High", "Low", "Close"])
        return d

    def _apply_date_filter(self, df: pd.DataFrame, start: Optional[str]) -> pd.DataFrame:
        out = df
        if start:
            out = out[out.index >= pd.to_datetime(start)]
        if self.cfg.end_date:
            out = out[out.index < pd.to_datetime(self.cfg.end_date)]
        if out.empty:
            raise ValueError("Data became empty after applying date filters.")
        return out

    def _load_hourly_local(self) -> pd.DataFrame:
        if self._hourly_cache is not None:
            return self._hourly_cache.copy()

        path = self._find_existing_file(
            self.cfg.hourly_csv_candidates, ["1h", "hour"], allow_any=True
        )
        log.info(f"Loading hourly data from local CSV ({path}) ...")
        df = self._prepare_ohlcv(self._read_csv_auto(path))
        df = self._apply_date_filter(df, self.cfg.hourly_start)
        self._hourly_cache = df.copy()
        log.info(f"  Hourly -> {len(df)} rows "
                 f"[{df.index[0].date()} -> {df.index[-1].date()}]")
        return df

    def _build_daily_from_hourly(self) -> pd.DataFrame:
        h = self._load_hourly_local()
        agg = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        }
        if "Volume" in h.columns:
            agg["Volume"] = "sum"
        d = h.resample("1D").agg(agg).dropna(subset=["Open", "High", "Low", "Close"])
        d = self._apply_date_filter(d, self.cfg.daily_start)
        log.info(f"  Daily(from hourly) -> {len(d)} rows "
                 f"[{d.index[0].date()} -> {d.index[-1].date()}]")
        return d

    def get_daily(self) -> pd.DataFrame:
        try:
            log.info(f"Downloading {self.cfg.ticker} daily data from yfinance ...")
            df = yf.download(
                self.cfg.ticker,
                start=self.cfg.daily_start,
                end=self.cfg.end_date,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                df.index = pd.to_datetime(df.index)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df.sort_index(inplace=True)
                df = self._apply_date_filter(df, self.cfg.daily_start)
                log.info(f"  Daily(yf) -> {len(df)} rows "
                         f"[{df.index[0].date()} -> {df.index[-1].date()}]")
                return df
            log.warning("No daily data from yfinance. Trying local daily CSV fallback.")
        except Exception as e:
            log.warning(f"yfinance daily download failed: {e}")

        try:
            daily_path = self._find_existing_file(
                self.cfg.daily_csv_candidates, ["1d", "daily"], allow_any=False
            )
            log.info(f"Loading daily data from local CSV ({daily_path}) ...")
            d = self._prepare_ohlcv(self._read_csv_auto(daily_path))
            d = self._apply_date_filter(d, self.cfg.daily_start)
            log.info(f"  Daily(local) -> {len(d)} rows "
                     f"[{d.index[0].date()} -> {d.index[-1].date()}]")
            return d
        except Exception as e:
            if not self.cfg.daily_from_hourly_fallback:
                raise
            log.warning(f"Local daily CSV fallback failed: {e}")
            log.info("Building daily bars from hourly CSV as final fallback ...")
            return self._build_daily_from_hourly()

    def get_hourly(self) -> pd.DataFrame:
        return self._load_hourly_local()


class FeatureEngineer:
    """
    Builds all indicators and performs the leakage-free dailyโ’hourly merge.

    Look-ahead bias prevention (unchanged from v2):
    โ€ข Daily bar D's close is shifted +1 calendar day before forward-fill.
    โ€ข Every hourly bar in day D therefore sees D-1's trend โ€” the only info
      a real trader would possess at that point in time.
    """

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg

    @staticmethod
    def _ema(s, span):  return s.ewm(span=span, adjust=False).mean()
    @staticmethod
    def _sma(s, w):     return s.rolling(w).mean()

    @staticmethod
    def _rsi(close, period):
        d    = close.diff()
        gain = d.clip(lower=0).rolling(period).mean()
        loss = (-d.clip(upper=0)).rolling(period).mean()
        return 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    @staticmethod
    def _atr(high, low, close, period):
        prev = close.shift(1)
        tr   = pd.concat([high - low,
                           (high - prev).abs(),
                           (low  - prev).abs()], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def build_daily_features(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["EMA_50"]      = self._ema(d["Close"], self.cfg.ema_fast)
        d["SMA_200"]     = self._sma(d["Close"], self.cfg.sma_slow)
        d["Daily_Trend"] = (d["Close"] > d["SMA_200"]).astype(int)
        return d[["EMA_50", "SMA_200", "Daily_Trend"]].dropna()

    def build_hourly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["RSI"]          = self._rsi(d["Close"], self.cfg.rsi_period)
        ef                = self._ema(d["Close"], self.cfg.macd_fast)
        es                = self._ema(d["Close"], self.cfg.macd_slow)
        d["MACD"]         = ef - es
        d["MACD_Signal"]  = self._ema(d["MACD"], self.cfg.macd_signal)
        d["MACD_Hist"]    = d["MACD"] - d["MACD_Signal"]
        d["ATR"]          = self._atr(d["High"], d["Low"], d["Close"],
                                      self.cfg.atr_period)
        d["Close_Ret_1h"] = d["Close"].pct_change(1)
        d["Close_Ret_4h"] = d["Close"].pct_change(4)
        d["Close_Ret_24h"]= d["Close"].pct_change(24)
        d["High_Low_Pct"] = (d["High"] - d["Low"]) / d["Close"]

        # เน€เธเธดเนเธกเธเธฒเธฃเธ”เธฑเธเธเธฑเธ Error เธ•เธฃเธเธเธตเน: เธเธณเธเธงเธ“ Vol_Ratio เธเนเธ•เนเธญเน€เธกเธทเนเธญเธกเธตเธเธญเธฅเธฑเธกเธเน Volume เน€เธ—เนเธฒเธเธฑเนเธ
        if "Volume" in d.columns:
            d["Vol_Ratio"]    = d["Volume"] / d["Volume"].rolling(24).mean()
        else:
            log.info("  No 'Volume' column found in hourly data. Skipping Vol_Ratio feature.")

        return d

    def merge_timeframes(self, df_hourly: pd.DataFrame,
                         df_daily_feats: pd.DataFrame) -> pd.DataFrame:
        """Leakage-free merge: daily close of day D arrives on day D+1."""
        daily_shifted = df_daily_feats.copy()
        daily_shifted.index = daily_shifted.index + pd.Timedelta(days=1)
        daily_reindexed = daily_shifted.reindex(df_hourly.index, method="ffill")
        merged = pd.concat([df_hourly, daily_reindexed], axis=1)
        before = len(merged)
        merged.dropna(subset=["Daily_Trend", "ATR"], inplace=True)
        log.info(f"  Merge: {before} โ’ {len(merged)} rows "
                 f"({before - len(merged)} NaN warm-up removed)")
        return merged


# โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•
#  ยง3  TARGET LABELING  (Meta-Labeling)
# โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•

class TargetLabeler:
    """
    Binary meta-label:
        1 โ’ TP (entry + tp_mult ร— ATR) hit before SL within label_horizon bars
        0 โ’ SL hit first, or time-stop (horizon exhausted)

    NOTE: tp_multiplier, sl_multiplier, and label_horizon are Optuna search
    parameters.  Because they change the *labels* themselves, they must be
    recomputed fresh for every trial โ€” handled automatically by instantiating
    a new TargetLabeler(cfg) inside each trial.
    """

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg

    def label(self, df: pd.DataFrame) -> pd.Series:
        closes = df["Close"].values
        highs  = df["High"].values
        lows   = df["Low"].values
        atrs   = df["ATR"].values
        n      = len(df)
        labels = np.zeros(n, dtype=np.int8)

        for i in range(n - self.cfg.label_horizon):
            atr_i = atrs[i]
            if np.isnan(atr_i) or atr_i == 0:
                continue
            entry    = closes[i]
            tp_price = entry + self.cfg.tp_multiplier * atr_i
            sl_price = entry - self.cfg.sl_multiplier * atr_i
            for j in range(self.cfg.label_horizon):
                tp_hit = highs[i + 1 + j] >= tp_price
                sl_hit = lows [i + 1 + j] <= sl_price
                if tp_hit and sl_hit:
                    labels[i] = 0; break
                elif tp_hit:
                    labels[i] = 1; break
                elif sl_hit:
                    labels[i] = 0; break

        result = pd.Series(labels, index=df.index, name="Target")
        log.info(f"  Labels: {result.mean()*100:.1f}% positive "
                 f"({result.sum()}/{len(result)}) "
                 f"[TPร—{self.cfg.tp_multiplier} / SLร—{self.cfg.sl_multiplier} "
                 f"/ H={self.cfg.label_horizon}]")
        return result


# โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•
#  ยง4  PYTORCH DNN
# โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•

class GoldDNN(nn.Module):
    """
    Input โ’ [Linear โ’ BatchNorm1d โ’ ReLU โ’ Dropout] ร— N โ’ Linear(1)

    Returns raw logits; Sigmoid is applied externally for inference.
    """

    def __init__(self, input_size: int, cfg: Config = CFG):
        super().__init__()
        layers, prev = [], input_size
        for h in cfg.hidden_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(p=cfg.dropout_rate)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, x):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


# โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•
#  ยง5  MODEL TRAINER
# โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•

class ModelTrainer:
    """
    Trains DNN + XGBoost on identical chronological splits.

    Two operating modes (controlled by the `final` flag in `train()`):
    โ”โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”ฌโ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”
    โ”  final=False โ”  TRIAL mode  (Optuna)                                    โ”
    โ”              โ”  โ€ข train_data = TRAIN partition only                     โ”
    โ”              โ”  โ€ข internal val used for DNN early stopping + XGB ES     โ”
    โ”              โ”  โ€ข epochs = cfg.epochs  (reduced for speed)             โ”
    โ”โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”ผโ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”ค
    โ”  final=True  โ”  FINAL mode  (post-optimisation)                         โ”
    โ”              โ”  โ€ข train_data = TRAIN + VAL combined                     โ”
    โ”              โ”  โ€ข uses a small internal pseudo-val (last 10% of input)  โ”
    โ”              โ”    solely for DNN early stopping โ€” never evaluated as    โ”
    โ”              โ”    a simulation target                                   โ”
    โ”              โ”  โ€ข epochs = cfg.final_epochs  (full budget)             โ”
    โ””โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”ดโ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”

    The scaler is always fit on whatever the first portion of input data is,
    never on validation or test data โ€” preventing any numeric leakage.
    """

    def __init__(self, cfg: Config = CFG):
        self.cfg          = cfg
        self.scaler       = StandardScaler()
        self.dnn_model    = None
        self.xgb_model    = None
        self.feature_cols: list = []

    # โ”€โ”€ Shared split helper โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€

    @staticmethod
    def _chrono_split(X: np.ndarray, y: np.ndarray,
                      train_frac: float, val_frac: float
                      ) -> Tuple[np.ndarray, ...]:
        """Generic chronological split into three contiguous partitions."""
        n  = len(X)
        t1 = int(n * train_frac)
        t2 = int(n * (train_frac + val_frac))
        return (X[:t1], X[t1:t2], X[t2:],
                y[:t1], y[t1:t2], y[t2:])

    # โ”€โ”€ Master entry point โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€

    def train(self,
              X: pd.DataFrame,
              y: pd.Series,
              final: bool = False) -> "ModelTrainer":
        """
        Parameters
        โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
        X, y   : full feature/label arrays (already NaN-cleaned)
        final  : False โ’ trial mode (train on TRAIN only, val = VAL)
                 True  โ’ final mode  (train on TRAIN+VAL, pseudo-val = last 10%)
        """
        self.feature_cols = list(X.columns)
        Xv = X.values
        yv = y.values

        epochs = self.cfg.final_epochs if final else self.cfg.epochs

        if not final:
            # โ”€โ”€ Trial mode: strict three-way split โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
            Xtr, Xva, Xte, ytr, yva, yte = self._chrono_split(
                Xv, yv, self.cfg.train_ratio, self.cfg.val_ratio
            )
            log.info(f"  [TRIAL]  train:{len(Xtr)} val:{len(Xva)} test:{len(Xte)}")
        else:
            # โ”€โ”€ Final mode: TRAIN+VAL as training corpus โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
            t2 = int(len(Xv) * (self.cfg.train_ratio + self.cfg.val_ratio))
            Xtr_full = Xv[:t2]
            ytr_full = yv[:t2]
            Xte      = Xv[t2:]
            yte      = yv[t2:]

            # Use last 10 % of train+val as pseudo-val for DNN early stopping
            # only โ€” it is never used for simulation or metric reporting.
            split_es = int(len(Xtr_full) * 0.90)
            Xtr, Xva = Xtr_full[:split_es], Xtr_full[split_es:]
            ytr, yva = ytr_full[:split_es], ytr_full[split_es:]
            log.info(f"  [FINAL]  train+val:{len(Xtr_full)} "
                     f"(ES-val:{len(Xva)}) | test:{len(Xte)}")

        # โ”€โ”€ Scale โ€” always fit only on the first (Xtr) partition โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
        Xstr = self.scaler.fit_transform(Xtr)
        Xsva = self.scaler.transform(Xva)
        Xste = self.scaler.transform(Xte)

        # โ”€โ”€ Train both models on identical arrays โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
        self._train_dnn(Xstr, Xsva, Xste, ytr, yva, yte, epochs=epochs)
        self._train_xgb(Xstr, Xsva, Xste, ytr, yva, yte)

        return self

    # โ”€โ”€ DNN training โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€

    def _train_dnn(self, Xtr, Xva, Xte, ytr, yva, yte, epochs: int) -> None:
        def _loader(X, y, shuffle=False):
            ds = TensorDataset(torch.FloatTensor(X),
                               torch.FloatTensor(y).unsqueeze(1))
            return DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=shuffle)

        pos_w = torch.tensor(
            [(ytr == 0).sum() / max((ytr == 1).sum(), 1)],
            dtype=torch.float32,
        ).to(self.cfg.device)

        self.dnn_model = GoldDNN(Xtr.shape[1], self.cfg).to(self.cfg.device)
        optim     = torch.optim.Adam(self.dnn_model.parameters(),
                                     lr=self.cfg.dnn_learning_rate,
                                     weight_decay=1e-4)
        sched     = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optim, patience=3, factor=0.5)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        best_vl, pat, best_state = float("inf"), 0, None
        tr_loader = _loader(Xtr, ytr)
        va_loader = _loader(Xva, yva)

        for ep in range(1, epochs + 1):
            self.dnn_model.train()
            tl = 0.0
            for Xb, yb in tr_loader:
                Xb, yb = Xb.to(self.cfg.device), yb.to(self.cfg.device)
                optim.zero_grad()
                loss = criterion(self.dnn_model(Xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.dnn_model.parameters(), 1.0)
                optim.step()
                tl += loss.item() * len(Xb)
            tl /= len(tr_loader.dataset)

            self.dnn_model.eval()
            vl = 0.0
            with torch.no_grad():
                for Xb, yb in va_loader:
                    Xb, yb = Xb.to(self.cfg.device), yb.to(self.cfg.device)
                    vl += criterion(self.dnn_model(Xb), yb).item() * len(Xb)
            vl /= len(va_loader.dataset)
            sched.step(vl)

            if ep % 10 == 0 or ep == 1:
                log.debug(f"    DNN epoch {ep:3d}/{epochs} | "
                          f"train={tl:.4f} val={vl:.4f}")

            if vl < best_vl - 1e-5:
                best_vl, pat = vl, 0
                best_state = {k: v.cpu().clone()
                              for k, v in self.dnn_model.state_dict().items()}
            else:
                pat += 1
                if pat >= self.cfg.early_stopping_patience:
                    log.debug(f"    DNN early stop @ epoch {ep}")
                    break

        if best_state:
            self.dnn_model.load_state_dict(best_state)

    # โ”€โ”€ XGBoost training โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€

    def _train_xgb(self, Xtr, Xva, Xte, ytr, yva, yte) -> None:
        spw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
        self.xgb_model = xgb.XGBClassifier(
            n_estimators      = self.cfg.xgb_n_estimators,
            max_depth         = self.cfg.xgb_max_depth,
            learning_rate     = self.cfg.xgb_learning_rate,
            subsample         = self.cfg.xgb_subsample,
            colsample_bytree  = self.cfg.xgb_colsample_bytree,
            min_child_weight  = self.cfg.xgb_min_child_weight,
            gamma             = self.cfg.xgb_gamma,
            reg_alpha         = self.cfg.xgb_reg_alpha,
            reg_lambda        = self.cfg.xgb_reg_lambda,
            scale_pos_weight  = spw,
            objective         = "binary:logistic",
            eval_metric       = "logloss",
            use_label_encoder = False,
            random_state      = self.cfg.random_seed,
            n_jobs            = -1,
            verbosity         = 0,
        )
        self.xgb_model.fit(
            Xtr, ytr,
            eval_set              = [(Xva, yva)],
            #early_stopping_rounds = self.cfg.xgb_early_stopping_rounds,
            verbose               = False,
        )

    # โ”€โ”€ Evaluation (used for final run only โ€” suppressed during trials) โ”€โ”€โ”€โ”€

    def evaluate(self, X: pd.DataFrame, y: pd.Series,
                 label: str = "test") -> None:
        """Full classification report + ensemble confusion matrix."""
        p_dnn, p_xgb = self.predict_proba_df(X)
        thr  = self.cfg.signal_threshold
        preds_dnn  = (p_dnn >= thr).astype(int)
        preds_xgb  = (p_xgb >= thr).astype(int)
        avg_p      = (p_dnn + p_xgb) / 2.0
        preds_soft = (avg_p >= thr).astype(int)
        yv = y.values

        log.info(f"\n{'โ”€'*60}")
        log.info(f"  Model Evaluation โ€” partition: {label.upper()}")
        log.info(f"{'โ”€'*60}")
        log.info(f"  [DNN  solo] ROC-AUC: {roc_auc_score(yv, p_dnn):.4f}")
        log.info(f"\n{classification_report(yv, preds_dnn, digits=3)}")
        log.info(f"  [XGB  solo] ROC-AUC: {roc_auc_score(yv, p_xgb):.4f}")
        log.info(f"\n{classification_report(yv, preds_xgb, digits=3)}")
        log.info(f"  [Soft ens.] ROC-AUC: {roc_auc_score(yv, avg_p):.4f}")
        log.info(f"\n{classification_report(yv, preds_soft, digits=3)}")
        log.info(f"  Confusion Matrix (soft):\n{confusion_matrix(yv, preds_soft)}")

    # โ”€โ”€ Inference helpers โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€

    def _raw_probas_from_scaled(self, Xs: np.ndarray
                                 ) -> Tuple[np.ndarray, np.ndarray]:
        self.dnn_model.eval()
        Xt    = torch.FloatTensor(Xs).to(self.cfg.device)
        p_dnn = self.dnn_model.predict_proba(Xt).cpu().numpy().flatten()
        p_xgb = self.xgb_model.predict_proba(Xs)[:, 1]
        return p_dnn, p_xgb

    def predict_proba_df(self, X: pd.DataFrame
                          ) -> Tuple[np.ndarray, np.ndarray]:
        Xs = self.scaler.transform(X[self.feature_cols].values)
        return self._raw_probas_from_scaled(Xs)


# โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•
#  ยง6  TRADING SIMULATOR
# โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•

@dataclass
class Trade:
    entry_bar: int; entry_time: pd.Timestamp
    entry_price: float; atr_at_entry: float
    initial_sl: float; tp_price: float; position_size: float
    exit_bar: int = -1; exit_time: Optional[pd.Timestamp] = None
    exit_price: float = 0.0; pnl_usd: float = 0.0; exit_reason: str = ""
    peak_price: float = 0.0; trailing_sl: float = 0.0
    proba_dnn: float = 0.0; proba_xgb: float = 0.0


class TradingSimulator:
    """
    Bar-by-bar simulation implementing:
        Macro filter (Daily_Trend == 1)
        ? Soft Voting gate (avg(DNN, XGB) > threshold)
        โ’ ATR position sizing
        โ’ Ratcheting trailing stop

    The `run()` method is intentionally stateless across calls โ€” a fresh
    `TradingSimulator` is created for each Optuna trial to avoid trade list
    carry-over between trials.
    """

    def __init__(self, cfg: Config = CFG):
        self.cfg    = cfg
        self.trades: list = []

    def run(self, df: pd.DataFrame,
            proba_dnn: np.ndarray,
            proba_xgb: np.ndarray,
            capital_override: Optional[float] = None) -> pd.DataFrame:

        capital    = capital_override or self.cfg.initial_capital
        open_trade = None
        closes = df["Close"].values; highs  = df["High"].values
        lows   = df["Low"].values;  atrs   = df["ATR"].values
        trends = df["Daily_Trend"].values; times = df.index
        thr    = self.cfg.signal_threshold

        for i in range(len(df)):
            price, high, low, atr = closes[i], highs[i], lows[i], atrs[i]
            p_dnn, p_xgb = proba_dnn[i], proba_xgb[i]
            avg_p = (p_dnn + p_xgb) / 2.0

            # โ”€โ”€ Manage open trade โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
            if open_trade is not None:
                if high > open_trade.peak_price:
                    open_trade.peak_price = high
                new_tsl = (open_trade.peak_price
                           - self.cfg.trailing_atr_mult * open_trade.atr_at_entry)
                if new_tsl > open_trade.trailing_sl:
                    open_trade.trailing_sl = new_tsl

                ep, er = None, ""
                if low <= open_trade.trailing_sl:
                    ep, er = open_trade.trailing_sl, "trailing_stop"
                elif high >= open_trade.tp_price:
                    ep, er = open_trade.tp_price, "take_profit"

                if ep is not None:
                    pnl = (ep - open_trade.entry_price) * open_trade.position_size
                    open_trade.exit_bar, open_trade.exit_time  = i, times[i]
                    open_trade.exit_price, open_trade.pnl_usd  = ep, pnl
                    open_trade.exit_reason = er
                    capital += pnl
                    self.trades.append(open_trade)
                    open_trade = None

            # โ”€โ”€ Seek new entry โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
            if open_trade is None and not np.isnan(atr) and atr > 0:
                # Soft Voting Ensemble: macro trend + average model confidence
                if (trends[i] == 1 and avg_p > thr):
                    sl   = price - self.cfg.sl_multiplier * atr
                    tp   = price + self.cfg.tp_multiplier * atr
                    risk = capital * self.cfg.risk_per_trade_pct
                    dist = price - sl
                    pos  = risk / dist if dist > 0 else 0.0
                    if pos <= 0:
                        continue
                    open_trade = Trade(
                        entry_bar=i, entry_time=times[i],
                        entry_price=price, atr_at_entry=atr,
                        initial_sl=sl, tp_price=tp, position_size=pos,
                        peak_price=price, trailing_sl=sl,
                        proba_dnn=float(p_dnn), proba_xgb=float(p_xgb),
                    )

        # Force-close any open trade at last bar
        if open_trade is not None:
            pnl = (closes[-1] - open_trade.entry_price) * open_trade.position_size
            open_trade.exit_bar, open_trade.exit_time   = len(df)-1, times[-1]
            open_trade.exit_price, open_trade.pnl_usd   = closes[-1], pnl
            open_trade.exit_reason = "end_of_data"
            capital += pnl
            self.trades.append(open_trade)

        return self._build_results()

    def _build_results(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        rows = []
        for t in self.trades:
            rows.append({
                "entry_time": t.entry_time, "exit_time": t.exit_time,
                "entry_price": round(t.entry_price, 2),
                "exit_price":  round(t.exit_price,  2),
                "position_sz": round(t.position_size, 4),
                "initial_sl":  round(t.initial_sl,  2),
                "trailing_sl": round(t.trailing_sl, 2),
                "tp_price":    round(t.tp_price,    2),
                "pnl_usd":     round(t.pnl_usd,     2),
                "exit_reason": t.exit_reason,
                "atr_entry":   round(t.atr_at_entry, 2),
                "bars_held":   t.exit_bar - t.entry_bar,
                "proba_dnn":   round(t.proba_dnn, 4),
                "proba_xgb":   round(t.proba_xgb, 4),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def compute_objective(results: pd.DataFrame) -> float:
        """
        Returns the scalar value that Optuna maximises.

        Metric: Net P&L (USD).
        """
        if results.empty:
            return -10_000.0

        net_pnl = results["pnl_usd"].sum()
        return float(net_pnl)

    def print_stats(self, results: pd.DataFrame, label: str = "") -> None:
        if results.empty:
            log.info(f"  [{label}] No trades executed.")
            return
        wins = results[results["pnl_usd"] > 0]
        loss = results[results["pnl_usd"] <= 0]
        wr   = len(wins) / max(len(results), 1) * 100
        aw   = wins["pnl_usd"].mean() if len(wins) else 0.0
        al   = loss["pnl_usd"].mean() if len(loss) else 0.0
        pf   = wins["pnl_usd"].sum() / abs(loss["pnl_usd"].sum() + 1e-9)

        log.info(f"\n{'โ•'*62}")
        log.info(f"  Performance Summary  [{label}]")
        log.info(f"{'โ•'*62}")
        log.info(f"  Trades        : {len(results)}")
        log.info(f"  Win rate      : {wr:.1f}%")
        log.info(f"  Avg win  ($)  : {aw:+,.2f}")
        log.info(f"  Avg loss ($)  : {al:+,.2f}")
        log.info(f"  Profit factor : {pf:.3f}")
        log.info(f"  Net P&L  ($)  : {results['pnl_usd'].sum():+,.2f}")
        log.info(f"  R/R ratio     : {abs(aw/al):.2f}" if al != 0 else "  R/R ratio     : โ")
        log.info(f"  Exit types    : {results['exit_reason'].value_counts().to_dict()}")
        log.info(f"  Avg bars held : {results['bars_held'].mean():.1f}")
        log.info(f"  Avg DNN prob  : {results['proba_dnn'].mean():.4f}")
        log.info(f"  Avg XGB prob  : {results['proba_xgb'].mean():.4f}")


# โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•
#  ยง7  GOLD TRADING PIPELINE  (data preparation layer)
# โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•

FEATURE_COLS = [
    "RSI", "MACD", "MACD_Signal", "MACD_Hist",
    "ATR", "Close_Ret_1h", "Close_Ret_4h", "Close_Ret_24h",
    "High_Low_Pct", "Vol_Ratio", "EMA_50", "SMA_200", "Daily_Trend",
]


class GoldTradingPipeline:
    """
    Responsible for the data half of the pipeline:
        Download โ’ Feature engineering โ’ Labeling โ’ X/y assembly

    The `prepare()` method returns (X, y, df_merged) with no side-effects,
    so it can be called once and the outputs cached by OptunaOptimiser for
    reuse across all trials.

    Training and simulation are delegated to ModelTrainer and TradingSimulator.
    """

    def __init__(self, cfg: Config = CFG):
        self.cfg      = cfg
        self.ingestion = DataIngestion(cfg)
        self.engineer  = FeatureEngineer(cfg)

    def prepare(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Run the full data pipeline and return (X, y, df_merged).

        NOTE: labels depend on tp_multiplier, sl_multiplier, label_horizon โ€”
        all of which are Optuna search variables.  Therefore `prepare()` must
        be called with the correct cfg already patched before invocation.
        This is handled by OptunaOptimiser._build_dataset() which calls
        prepare() inside each trial with the trial's cfg.
        """
        log.info("โ”€โ”€ Data preparation โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€")
        df_daily  = self.ingestion.get_daily()
        df_hourly = self.ingestion.get_hourly()

        df_daily_feats  = self.engineer.build_daily_features(df_daily)
        df_hourly_feats = self.engineer.build_hourly_features(df_hourly)
        df_merged       = self.engineer.merge_timeframes(df_hourly_feats,
                                                          df_daily_feats)

        labeler = TargetLabeler(self.cfg)
        y       = labeler.label(df_merged)

        # Trim tail rows whose labels are undefined
        df_merged = df_merged.iloc[: -self.cfg.label_horizon]
        y         = y.iloc[: -self.cfg.label_horizon]

        # Feature selection + NaN guard
        cols = [c for c in FEATURE_COLS if c in df_merged.columns]
        X    = df_merged[cols].copy()
        ok   = X.notna().all(axis=1) & y.notna()
        X, y, df_merged = X[ok], y[ok], df_merged[ok]

        log.info(f"  Dataset ready: {len(X)} rows ร— {len(cols)} features")
        return X, y, df_merged


# โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•
#  ยง8  OPTUNA OPTIMISER  (new in v3)
# โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•

class OptunaOptimiser:
    """
    Orchestrates the full three-phase workflow:

    Phase 1 โ€” Data cache
        Download data once and cache the raw OHLCV DataFrames.
        Feature engineering + labeling must run inside each trial because
        labels depend on tp/sl multipliers (Optuna search variables).

    Phase 2 โ€” Optuna study (n_trials)
        Each trial:
          a. Patches cfg with trial suggestions.
          b. Re-runs feature engineering + labeling with new cfg.
          c. Trains DNN + XGB on TRAIN partition.
          d. Simulates on VAL partition.
          e. Returns objective score (Net P&L with no-trade penalty).
        The TEST partition is never touched.

    Phase 3 โ€” Final evaluation
        a. Injects best_params into cfg.
        b. Re-labels entire dataset with winning tp/sl/horizon.
        c. Trains DNN + XGB on TRAIN + VAL combined.
        d. Runs definitive simulation on TEST only.
        e. Prints full stats + saves results CSV.

    Leakage guarantees
    โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
    โ€ข Data is downloaded once; feature engineering is cfg-dependent so it
      runs anew per trial but always from the same raw OHLCV source.
    โ€ข The StandardScaler in ModelTrainer is always fit on the training
      partition only, never the val or test partition.
    โ€ข TEST data is completely sequestered during all 30 trials โ€” it is only
      touched once in Phase 3 after the study has completed.
    โ€ข Optuna's TPE sampler explores the search space based solely on val
      performance, with no feedback from test data.
    """

    def __init__(self, cfg: Config = CFG):
        self.cfg          = cfg
        self._df_daily    = None   # Raw OHLCV cache โ€” fetched once
        self._df_hourly   = None

    # โ”€โ”€ Phase 1: cache raw data โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€

    def _cache_raw_data(self) -> None:
        """Download raw OHLCV once; reused across all trials."""
        log.info("Caching raw OHLCV data (one-time download) โ€ฆ")
        ingestion       = DataIngestion(self.cfg)
        self._df_daily  = ingestion.get_daily()
        self._df_hourly = ingestion.get_hourly()
        log.info("  Raw data cached โ“")

    # โ”€โ”€ Trial dataset builder โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€

    def _build_dataset(self, trial_cfg: Config
                        ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Run feature engineering + labeling for a specific trial config.
        Uses the cached raw OHLCV โ€” no re-download.
        """
        eng = FeatureEngineer(trial_cfg)
        df_daily_feats  = eng.build_daily_features(self._df_daily)
        df_hourly_feats = eng.build_hourly_features(self._df_hourly)
        df_merged       = eng.merge_timeframes(df_hourly_feats, df_daily_feats)

        labeler = TargetLabeler(trial_cfg)
        y       = labeler.label(df_merged)

        df_merged = df_merged.iloc[: -trial_cfg.label_horizon]
        y         = y.iloc[: -trial_cfg.label_horizon]

        cols = [c for c in FEATURE_COLS if c in df_merged.columns]
        X    = df_merged[cols].copy()
        ok   = X.notna().all(axis=1) & y.notna()
        return X[ok], y[ok], df_merged[ok]

    # โ”€โ”€ Optuna objective โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€

    def objective(self, trial: optuna.Trial) -> float:
        """
        Single Optuna trial.

        Search space (5 parameters):
        โ”โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”ฌโ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”ฌโ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”
        โ” Parameter          โ” Type      โ” Range / Rationale                โ”
        โ”โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”ผโ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”ผโ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”ค
        โ” tp_multiplier      โ” float     โ” [1.5, 5.0] โ€” wide enough to      โ”
        โ”                    โ”           โ” include scalp (1.5) and swing    โ”
        โ”                    โ”           โ” (5.0) styles                     โ”
        โ” sl_multiplier      โ” float     โ” [0.5, 2.5] โ€” tight (0.5ร—ATR)    โ”
        โ”                    โ”           โ” to generous (2.5ร—ATR)            โ”
        โ” trailing_atr_mult  โ” float     โ” [1.0, 3.5] โ€” must be โฅ 1 to     โ”
        โ”                    โ”           โ” leave breathing room             โ”
        โ” label_horizon      โ” int       โ” [24, 120] โ€” 1 day to 5 days of  โ”
        โ”                    โ”           โ” hourly bars to resolve TP/SL     โ”
        โ” signal_threshold   โ” float     โ” [0.50, 0.75] โ€” lower bound is   โ”
        โ”                    โ”           โ” 0.50 (coin flip) so Optuna can  โ”
        โ”                    โ”           โ” discover if strict filtering     โ”
        โ”                    โ”           โ” genuinely helps                  โ”
        โ””โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”ดโ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”ดโ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”

        Validation strategy:
        โ€ข Train on TRAIN partition (70 % of data).
        โ€ข Evaluate objective on VAL partition (15 %) ONLY.
        โ€ข TEST partition (15 %) never seen.
        """
        # โ”€โ”€ 1. Sample hyperparameters โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
        tp_mult      = trial.suggest_float("tp_multiplier",    1.5, 5.0,  step=0.1)
        sl_mult      = trial.suggest_float("sl_multiplier",    0.5, 2.5,  step=0.1)
        trail_mult   = trial.suggest_float("trailing_atr_mult",1.0, 3.5,  step=0.1)
        horizon      = trial.suggest_int  ("label_horizon",    24,  120,  step=12)
        threshold    = trial.suggest_float("signal_threshold", 0.50, 0.75, step=0.01)

        # โ”€โ”€ 2. Build trial-specific config (copy to avoid mutation) โ”€โ”€โ”€โ”€โ”€โ”€โ”€
        trial_cfg = copy.deepcopy(self.cfg)
        trial_cfg.tp_multiplier    = tp_mult
        trial_cfg.sl_multiplier    = sl_mult
        trial_cfg.trailing_atr_mult = trail_mult
        trial_cfg.label_horizon    = horizon
        trial_cfg.signal_threshold = threshold

        # โ”€โ”€ 3. Build dataset with trial labels โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
        try:
            X, y, df_merged = self._build_dataset(trial_cfg)
        except Exception as e:
            log.debug(f"  Trial {trial.number}: dataset build failed โ€” {e}")
            return -1_000.0

        n  = len(X)
        t1 = int(n * trial_cfg.train_ratio)
        t2 = int(n * (trial_cfg.train_ratio + trial_cfg.val_ratio))

        if t1 < 100 or (t2 - t1) < 20:
            log.debug(f"  Trial {trial.number}: insufficient rows โ’ prune")
            raise optuna.exceptions.TrialPruned()

        # โ”€โ”€ 4. Train on TRAIN partition only โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
        trainer = ModelTrainer(trial_cfg)
        try:
            trainer.train(X, y, final=False)
        except Exception as e:
            log.debug(f"  Trial {trial.number}: training failed โ€” {e}")
            return -1_000.0

        # โ”€โ”€ 5. Generate probabilities for VAL partition โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
        X_val    = X.iloc[t1:t2]
        df_val   = df_merged.iloc[t1:t2]
        p_dnn_v, p_xgb_v = trainer.predict_proba_df(X_val)

        # โ”€โ”€ 6. Simulate on VAL partition (TEST untouched) โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
        sim = TradingSimulator(trial_cfg)
        results = sim.run(df_val, p_dnn_v, p_xgb_v)

        # โ”€โ”€ 7. Compute objective โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
        score = TradingSimulator.compute_objective(results)

        n_trades = len(results) if not results.empty else 0
        log.info(
            f"  Trial {trial.number:>3d} | "
            f"TPร—{tp_mult:.1f} SLร—{sl_mult:.1f} "
            f"Trailร—{trail_mult:.1f} H={horizon:3d} "
            f"Thr={threshold:.2f} | "
            f"Trades={n_trades:>4d} | "
            f"Score={score:>+8.3f}"
        )
        return score

    # โ”€โ”€ Phase 2: run the study โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€

    def run_optimization(self) -> Dict:
        """
        Runs `n_trials` Optuna trials using TPE (Bayesian) sampling,
        evaluating each trial on the VAL partition only.

        Returns the best_params dictionary.
        """
        if self._df_daily is None:
            self._cache_raw_data()

        log.info("โ•" * 62)
        log.info(f"  OPTUNA OPTIMISATION โ€” {self.cfg.n_trials} trials")
        log.info("  Objective partition : VAL only (TEST sequestered)")
        log.info("  Objective metric    : Net P&L (USD, no-trade penalty = -10000)")
        log.info("โ•" * 62)

        sampler = TPESampler(seed=self.cfg.optuna_seed)
        study   = optuna.create_study(
            direction   = "maximize",
            sampler     = sampler,
            study_name  = "gold_ensemble_v3",
        )
        study.optimize(
            self.objective,
            n_trials        = self.cfg.n_trials,
            show_progress_bar = False,
        )

        log.info("\n" + "โ•" * 62)
        log.info("  OPTIMISATION COMPLETE")
        log.info(f"  Best trial  : #{study.best_trial.number}")
        log.info(f"  Best score  : {study.best_value:.4f}")
        log.info("  Best params :")
        for k, v in study.best_params.items():
            log.info(f"    {k:<22s} = {v}")
        log.info("โ•" * 62)

        return study.best_params

    # โ”€โ”€ Phase 3: inject best params + final run โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€

    def apply_params(self, best_params: Dict) -> None:
        """
        Overwrites the relevant Config fields with Optuna's winning values.
        Called between Phase 2 and Phase 3.
        """
        for k, v in best_params.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
                log.info(f"  Config.{k:<22s} โ {v}")
            else:
                log.warning(f"  Unknown config key from Optuna: {k}")

    def run_final(self) -> pd.DataFrame:
        """
        Phase 3 โ€” trains on TRAIN+VAL with best params, evaluates on TEST.

        Steps:
          1. Rebuild dataset with best labeling params.
          2. Train DNN + XGBoost on rows [0 โ€ฆ t2)  (TRAIN + VAL).
          3. Simulate on rows [t2 โ€ฆ end)            (TEST only, first look).
          4. Print full performance stats + save CSV.
        """
        log.info("\n" + "โ•" * 62)
        log.info("  FINAL EVALUATION โ€” unseen TEST partition")
        log.info("  Models trained on: TRAIN + VAL combined")
        log.info("โ•" * 62)

        # Re-build dataset with optimal labeling parameters
        X, y, df_merged = self._build_dataset(self.cfg)

        # Train on TRAIN+VAL (final=True inside ModelTrainer)
        self.cfg.epochs = self.cfg.final_epochs    # Restore full epoch budget
        trainer = ModelTrainer(self.cfg)
        trainer.train(X, y, final=True)

        # Slice TEST partition
        n  = len(X)
        t2 = int(n * (self.cfg.train_ratio + self.cfg.val_ratio))
        X_test  = X.iloc[t2:]
        y_test  = y.iloc[t2:]
        df_test = df_merged.iloc[t2:]

        log.info(f"  Test partition: {len(df_test)} bars "
                 f"[{df_test.index[0].date()} โ’ {df_test.index[-1].date()}]")

        # Model evaluation (classification metrics on TEST)
        trainer.evaluate(X_test, y_test, label="test")

        # Simulation on TEST
        p_dnn, p_xgb = trainer.predict_proba_df(X_test)
        n_dnn  = (p_dnn > self.cfg.signal_threshold).sum()
        n_xgb  = (p_xgb > self.cfg.signal_threshold).sum()
        avg_p  = (p_dnn + p_xgb) / 2.0
        n_soft = (avg_p > self.cfg.signal_threshold).sum()
        log.info(f"  Signals -> DNN:{n_dnn}  XGB:{n_xgb}  Soft-agreed:{n_soft}")

        sim     = TradingSimulator(self.cfg)
        results = sim.run(df_test, p_dnn, p_xgb)
        sim.print_stats(results, label="FINAL TEST")

        if not results.empty:
            fname = "gold_ensemble_v3_final.csv"
            results.to_csv(fname, index=False)
            log.info(f"  Results saved โ’ {fname}")

        return results

    # โ”€โ”€ Master entry point โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€

    def run(self) -> pd.DataFrame:
        """
        Convenience wrapper that runs all three phases in sequence:
            Phase 1 โ’ cache data
            Phase 2 โ’ Optuna optimisation (val-set only)
            Phase 3 โ’ final evaluation    (test-set only)
        """
        # Phase 1: cache raw data once
        self._cache_raw_data()

        # Phase 2: optimise on val partition
        best_params = self.run_optimization()

        # Inject winning parameters into cfg
        log.info("\nโ”€โ”€ Injecting best parameters into Config โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€")
        self.apply_params(best_params)

        # Phase 3: re-train on train+val, evaluate on test
        results = self.run_final()

        log.info("\n" + "โ•" * 62)
        log.info("  PIPELINE v3 COMPLETE")
        log.info("โ•" * 62)
        return results


# โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•
#  ENTRY POINT
# โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•

def main() -> None:
    optimiser = OptunaOptimiser(cfg=CFG)
    trade_results = optimiser.run()

    if trade_results is not None and not trade_results.empty:
        print("\n-- First 10 Trades (Optuna Best Params) --")
        print(trade_results.head(10).to_string(index=False))
        trade_results.to_csv("gold_optuna_backtest.csv", index=False)
        print("Results saved -> gold_optuna_backtest.csv")
    else:
        print("No trades generated by best params.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("Pipeline crashed")
        raise

