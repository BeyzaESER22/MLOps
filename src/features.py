import pandas as pd
from sklearn.feature_extraction import FeatureHasher

def apply_feature_cross(df):
    """Feature Cross uygular (Örnek: Kategori ve Seviye birleşimi)"""
    if 'Category' in df.columns and 'Course_Level' in df.columns:
        df['Category_Level_Cross'] = df['Category'].astype(str) + '_' + df['Course_Level'].astype(str)
    return df

def apply_hashing(df, col_name, n_features=50):
    """Hashing Trick uygular (High Cardinality için)"""
    if col_name not in df.columns:
        return df
    
    hasher = FeatureHasher(n_features=n_features, input_type='string')
    # Sütunu string'e çevir
    hashed_features = hasher.transform(df[col_name].astype(str).apply(lambda x: [x])).toarray()
    
    # Yeni sütun isimleri
    feature_names = [f'hashed_{col_name}_{i}' for i in range(n_features)]
    hashed_df = pd.DataFrame(hashed_features, columns=feature_names, index=df.index)
    
    # Orijinal sütunu at ve yenileri ekle
    df = pd.concat([df.drop(columns=[col_name]), hashed_df], axis=1)
    return df
