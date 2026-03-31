import pandas as pd
from pathlib import Path

candidates = [
    'xauusd_1h.csv',
    'XAU_1h_data.csv',
    'XAUUSD_1h.csv',
    'XAU_1d_data.csv',
]

found = []
for c in candidates:
    p = Path(c)
    if p.exists():
        found.append(p)

if not found:
    found = list(Path('.').glob('*.csv'))

if not found:
    raise SystemExit('No CSV files found in current directory.')

for p in found:
    print(f'\n=== {p} ===')
    df = pd.read_csv(p, sep=None, engine='python', nrows=5)
    print('Columns:', df.columns.tolist())
    print(df.head(2).to_string(index=False))
