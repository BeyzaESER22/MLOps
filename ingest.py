import pandas as pd
import os

def load_data(filepath):
    """Veriyi yükler. Dosya yoksa hata fırlatır."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {filepath}")
    return pd.read_csv(filepath)
