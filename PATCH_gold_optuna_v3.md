# Gold Optuna v3 Hotfix (Drop-in Patch)

ใช้แพตช์นี้กับไฟล์สคริปต์เดิมของคุณ (ที่มีคลาส `Config`, `DataIngestion`, และ entrypoint `if __name__ == "__main__":`) เพื่อแก้อาการ "รันแล้วเงียบ/ไม่มีอะไรออกมา" โดยไม่ต้องรื้อ architecture เดิม

## 1) แก้ Logging ให้บังคับโชว์เสมอ

แทนบล็อก `logging.basicConfig(...)` เดิมด้วย:

```python
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,  # สำคัญ: บังคับ reset handler เดิม (Jupyter/IDE ชอบกิน log)
)
log = logging.getLogger(__name__)
```

## 2) เพิ่มพาธไฟล์แบบยืดหยุ่นใน Config

เพิ่ม field ต่อไปนี้ใน `Config`:

```python
from typing import List

hourly_csv_candidates: List[str] = field(default_factory=lambda: [
    "xauusd_1h.csv",     # ชื่อเดิมในโค้ด
    "XAU_1h_data.csv",   # ชื่อจาก Kaggle ชุดที่คุณเปิดอยู่
    "XAUUSD_1h.csv",
])
daily_csv_candidates: List[str] = field(default_factory=lambda: [
    "XAU_1d_data.csv",   # fallback ถ้า yfinance ล่ม
    "xauusd_1d.csv",
])
```

## 3) แทนที่ทั้งคลาส DataIngestion ด้วยเวอร์ชันนี้

```python
from pathlib import Path

class DataIngestion:
    """
    Robust loader:
    - Daily: yfinance ก่อน, fallback ไป local CSV ได้
    - Hourly: auto-detect filename + delimiter + timestamp/ohlcv columns
    """

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg

    @staticmethod
    def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
        # clean ชื่อคอลัมน์: ลบ # / ช่องว่าง / lower-case
        clean = {
            c: c.strip().lower().replace("#", "").replace(" ", "_")
            for c in df.columns
        }
        d = df.rename(columns=clean).copy()

        mapping = {
            "open": "Open", "high": "High", "low": "Low", "close": "Close",
            "volume": "Volume", "tick_volume": "Volume", "vol": "Volume",
            "datetime": "Datetime", "date": "Datetime", "time": "Datetime",
            "timestamp": "Datetime", "local_time": "Datetime", "gmt_time": "Datetime",
        }
        for src, dst in mapping.items():
            if src in d.columns and dst not in d.columns:
                d.rename(columns={src: dst}, inplace=True)
        return d

    @staticmethod
    def _read_csv_auto(path: Path) -> pd.DataFrame:
        # sep=None ให้ pandas sniff delimiter อัตโนมัติ (รองรับ , ; \t)
        return pd.read_csv(path, sep=None, engine="python")

    def _find_existing_file(self, candidates) -> Path:
        for name in candidates:
            p = Path(name)
            if p.exists() and p.is_file():
                return p

        # fallback: หาไฟล์ที่มี 1h/1d ในชื่อ
        all_csv = list(Path(".").glob("*.csv"))
        if not all_csv:
            raise FileNotFoundError("No CSV file found in current directory.")
        for p in all_csv:
            n = p.name.lower()
            if "1h" in n or "hour" in n:
                return p
        return all_csv[0]

    def _postprocess_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        d = self._normalise_cols(df)

        if "Datetime" not in d.columns:
            # เผื่อชื่อคอลัมน์แปลก ให้ลองหา column ที่ parse datetime ได้
            for c in d.columns:
                parsed = pd.to_datetime(d[c], errors="coerce", dayfirst=True)
                if parsed.notna().mean() > 0.8:
                    d["Datetime"] = parsed
                    break
        else:
            d["Datetime"] = pd.to_datetime(d["Datetime"], errors="coerce", dayfirst=True)

        required = ["Datetime", "Open", "High", "Low", "Close"]
        missing = [c for c in required if c not in d.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing} | got: {d.columns.tolist()}")

        d = d.dropna(subset=["Datetime"]).copy()
        d.set_index("Datetime", inplace=True)
        if d.index.tz is not None:
            d.index = d.index.tz_localize(None)

        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")

        d = d.sort_index()
        d = d[~d.index.duplicated(keep="last")]
        return d

    def get_daily(self) -> pd.DataFrame:
        # 1) Try yfinance
        try:
            log.info(f"Downloading {self.cfg.ticker} daily data from yfinance ...")
            d = yf.download(
                self.cfg.ticker,
                start=self.cfg.daily_start,
                end=self.cfg.end_date,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            d.index = pd.to_datetime(d.index)
            if d.index.tz is not None:
                d.index = d.index.tz_localize(None)
            d = d.sort_index()
            if not d.empty:
                log.info(f"  Daily(yf) -> {len(d)} rows [{d.index[0]} -> {d.index[-1]}]")
                return d
            log.warning("yfinance returned empty daily data. Fallback to local daily CSV.")
        except Exception as e:
            log.warning(f"yfinance daily download failed: {e}. Fallback to local daily CSV.")

        # 2) Fallback to local daily csv
        p = self._find_existing_file(self.cfg.daily_csv_candidates)
        log.info(f"Loading daily from local CSV: {p}")
        d = self._postprocess_ohlcv(self._read_csv_auto(p))
        log.info(f"  Daily(local) -> {len(d)} rows [{d.index[0]} -> {d.index[-1]}]")
        return d

    def get_hourly(self) -> pd.DataFrame:
        p = self._find_existing_file(self.cfg.hourly_csv_candidates)
        log.info(f"Loading hourly from local CSV: {p}")
        h = self._postprocess_ohlcv(self._read_csv_auto(p))
        log.info(f"  Hourly -> {len(h)} rows [{h.index[0]} -> {h.index[-1]}]")
        return h
```

## 4) แก้ entrypoint ให้ไม่เงียบ (เห็น traceback ชัด)

แทนท้ายไฟล์เดิมด้วย:

```python
def main():
    log.info("Starting Gold Optuna pipeline v3 ...")
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
```

## 5) Optional: ลดเวลา debug รอบแรก

ใน `Config` ตั้งชั่วคราว:

```python
n_trials = 5
epochs = 5
final_epochs = 10
```

เพื่อยืนยันว่า flow วิ่งครบก่อน แล้วค่อยขยับกลับค่าเดิม
