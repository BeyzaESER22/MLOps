import pandas as pd

def clean_data(df):
    """Eksik verileri temizler."""
    # Basit temizlik: Eksik satırları at
    return df.dropna()

def balance_data(df):
    """Veri dengeleme (Basitleştirilmiş: Olduğu gibi döndürür)"""
    # Gerçek projede burada SMOTE vs. olurdu
    return df
