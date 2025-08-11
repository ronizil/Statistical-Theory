import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import silhouette_samples
import warnings

# Suppress seaborn palette warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# === PDF saving ===
OUTDIR = Path("figures_pdf")
OUTDIR.mkdir(exist_ok=True)

def save_pdf(name: str, dpi: int = 300, close: bool = True) -> None:
    """
    Saves the current matplotlib figure to PDF with a given name.

    Parameters:
        name (str): Filename (without extension)
        dpi (int or float): Resolution for the saved figure
        close (bool): Whether to close the figure after saving
    """
    path = OUTDIR / f"{name}.pdf"
    dpi = int(dpi)  # ensure dpi is int, even if passed as float or np.float64
    plt.savefig(path, format="pdf", dpi=dpi,
                bbox_inches='tight', pad_inches=0)
    print(f"✓ PDF saved to → {path}")
    if close:
        plt.close()
