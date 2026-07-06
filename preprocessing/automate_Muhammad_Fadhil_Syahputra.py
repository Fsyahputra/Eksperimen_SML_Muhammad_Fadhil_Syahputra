import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def automate_preprocessing(input_path='../titanic.csv', output_path='titanic_preprocessed.csv'):
    print("=== Memulai Otomatisasi Preprocessing Dataset Titanic ===")
    
    # 1. Memuat Dataset
    print(f"[1/8] Memuat dataset dari '{input_path}'...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File {input_path} tidak ditemukan!")
        return None
    print(f"   - Dataset dimuat dengan bentuk: {df.shape}")
    
    # 2. Menghapus Data Duplikat
    print("[2/8] Memeriksa data duplikat...")
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates()
        print(f"   - Berhasil menghapus {duplicate_count} baris duplikat.")
    else:
        print("   - Tidak ditemukan data duplikat.")
        
    # 3. Menangani Data Kosong (Missing Values)
    print("[3/8] Menangani missing values...")
    # Imputasi Embarked dengan modus ('S')
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Imputasi Age menggunakan median berdasarkan Pclass dan Sex
    df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
    
    # Penanganan Cabin dengan membuat fitur indikator biner 'Has_Cabin'
    df['Has_Cabin'] = df['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)
    print("   - Imputasi selesai (Embarked, Age, Cabin).")

    # 4. Feature Engineering
    print("[4/8] Melakukan Feature Engineering...")
    # Menggabungkan SibSp dan Parch menjadi FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Menentukan apakah penumpang bepergian sendiri
    df['IsAlone'] = df['FamilySize'].apply(lambda x: 1 if x == 1 else 0)
    print("   - Fitur baru berhasil dibuat: 'FamilySize', 'IsAlone'.")

    # 5. Binning (Pengelompokan Data)
    print("[5/8] Melakukan Binning pada kolom Age...")
    age_bins = [0, 12, 18, 35, 60, 120]
    age_labels = ['Child', 'Teenager', 'Young Adult', 'Adult', 'Elderly']
    df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
    print("   - Kategori usia berhasil dibuat: 'AgeGroup'.")

    # 6. Deteksi dan Penanganan Outlier (Log Transformation)
    print("[6/8] Menangani outlier pada kolom Fare dengan Log Transformation...")
    df['Fare_Log'] = np.log1p(df['Fare'])
    print("   - Fitur 'Fare_Log' berhasil dibuat.")

    # 7. Encoding Data Kategorikal
    print("[7/8] Melakukan encoding variabel kategorikal...")
    # Label encoding untuk Sex (male -> 0, female -> 1)
    df['Sex_Code'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # One-hot encoding untuk Embarked & AgeGroup
    df = pd.get_dummies(df, columns=['Embarked', 'AgeGroup'], drop_first=True, dtype=int)
    print("   - Encoding selesai.")

    # 8. Normalisasi & Standarisasi Fitur
    print("[8/8] Melakukan standarisasi fitur numerik...")
    scaler = StandardScaler()
    numerical_features = ['Age', 'Fare_Log', 'FamilySize']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    # Hapus kolom yang tidak lagi dibutuhkan untuk pemodelan
    drop_cols = ['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Fare', 'SibSp', 'Parch']
    df_model = df.drop(columns=drop_cols)
    print(f"   - Fitur-fitur yang tidak diperlukan telah dihapus. Sisa kolom: {df_model.shape[1]}")
    
    # Menyimpan data bersih
    df_model.to_csv(output_path, index=False)
    print(f"\n>>> Sukses! Data hasil preprocessing disimpan ke '{output_path}'")
    print(f"Shape data akhir: {df_model.shape}\n")
    return df_model

if __name__ == '__main__':
    automate_preprocessing()
