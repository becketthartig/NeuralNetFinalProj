import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import re

# simple script to graph all of the columns in our daily_features CSVs
# useful for visualization of correlations and just data in general

CSV_PATH = "daily_features_normalized.csv" 
OUT_DIR  = Path("plots")
OUT_DIR.mkdir(exist_ok=True)


df = pd.read_csv(CSV_PATH, parse_dates=[0])  
df = df.sort_values(df.columns[0])
date_col = df.columns[0]
dates = df[date_col]

for c in df.columns[1:]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

pdf_path = OUT_DIR / "all_columns.pdf"
with PdfPages(pdf_path) as pdf:
    for col in df.columns[1:]:
        y = df[col]
        if y.dropna().empty:
            continue  

        plt.figure(figsize=(10, 4))
        plt.plot(dates, y, marker="o", linewidth=1)
        plt.title(col)
        plt.xlabel("Date")
        plt.ylabel(col)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.gcf().autofmt_xdate()

        png_path = OUT_DIR / f"{safe_name(col)}.png"
        plt.savefig(png_path, dpi=150)
        pdf.savefig()
        plt.close()

