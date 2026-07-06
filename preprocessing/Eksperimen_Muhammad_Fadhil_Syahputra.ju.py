# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     name: python
# ---

# %% [markdown]
# # **1. Perkenalan Dataset**

# %% [markdown]
# Tahap pertama, Anda harus mencari dan menggunakan dataset dengan ketentuan sebagai berikut:
#
# 1. **Sumber Dataset**:  
#   Dataset yang saya gunakan dalam eksperimen ini merupakan dataset titanic yang saya dapatkan dari public repositories github. Dataset ini berisi informasi tentang penumpang kapal Titanic, termasuk fitur-fitur seperti usia, jenis kelamin, kelas tiket, dan status kelangsungan hidup. Berikut link repositori dataset yang saya gunakan: [Titanic Dataset](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv) 


# %% [markdown]
# # **2. Import Library**
# Pada tahap ini, Anda perlu mengimpor beberapa pustaka (library) Python yang dibutuhkan untuk analisis data dan pembangunan model machine learning atau deep learning.
# %%
import pandas as pd
import numpy as np

# %% [markdown]
# # **3. Memuat Dataset**
# %% [markdown]
# Pada tahap ini, Anda perlu memuat dataset ke dalam notebook. Jika dataset dalam format CSV, Anda bisa menggunakan pustaka pandas untuk membacanya. Pastikan untuk mengecek beberapa baris awal dataset untuk memahami strukturnya dan memastikan data telah dimuat dengan benar.

# Jika dataset berada di Google Drive, pastikan Anda menghubungkan Google Drive ke Colab terlebih dahulu. Setelah dataset berhasil dimuat, langkah berikutnya adalah memeriksa kesesuaian data dan siap untuk dianalisis lebih lanjut.
#
# Jika dataset berupa unstructured data, silakan sesuaikan dengan format seperti kelas Machine Learning Pengembangan atau Machine Learning Terapan


# %%
df = pd.read_csv('../titanic.csv')
df.head()


# %% [markdown]
# # **4. Exploratory Data Analysis (EDA)**
#
# Pada tahap ini, Anda akan melakukan **Exploratory Data Analysis (EDA)** untuk memahami karakteristik dataset.
# Tujuan dari EDA adalah untuk memperoleh wawasan awal yang mendalam mengenai data dan menentukan langkah selanjutnya dalam analisis atau pemodelan.

# %% [markdown]
# ### 4.1 Statistik Deskriptif dan Informasi Umum
# %%
# Melihat ringkasan informasi dataset
print("--- Info Dataset ---")
df.info()

print("\n--- Statistik Deskriptif (Numerik) ---")
print(df.describe())

print("\n--- Statistik Deskriptif (Kategorikal) ---")
print(df.describe(include=['O']))

# %% [markdown]
# ### 4.2 Analisis Missing Values
# %%
missing_data = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({'Total Missing': missing_data, 'Percentage (%)': missing_percent})
print(missing_df[missing_df['Total Missing'] > 0])

# %% [markdown]
# ### 4.3 Visualisasi Distribusi dan Hubungan Fitur dengan Kelangsungan Hidup (Survived)
# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")

# Buat figure dengan beberapa subplot
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Distribusi Kelas Penumpang (Pclass) vs Kelangsungan Hidup
sns.countplot(data=df, x='Pclass', hue='Survived', ax=axes[0, 0], palette='Set2')
axes[0, 0].set_title('Tingkat Kelangsungan Hidup berdasarkan Kelas (Pclass)')
axes[0, 0].set_xlabel('Kelas Penumpang')
axes[0, 0].set_ylabel('Jumlah Penumpang')

# 2. Distribusi Jenis Kelamin (Sex) vs Kelangsungan Hidup
sns.countplot(data=df, x='Sex', hue='Survived', ax=axes[0, 1], palette='Set1')
axes[0, 1].set_title('Tingkat Kelangsungan Hidup berdasarkan Jenis Kelamin')
axes[0, 1].set_xlabel('Jenis Kelamin')
axes[0, 1].set_ylabel('Jumlah Penumpang')

# 3. Distribusi Pelabuhan Embarkasi (Embarked) vs Kelangsungan Hidup
sns.countplot(data=df, x='Embarked', hue='Survived', ax=axes[0, 2], palette='Pastel1')
axes[0, 2].set_title('Tingkat Kelangsungan Hidup berdasarkan Embarked')
axes[0, 2].set_xlabel('Pelabuhan Embarkasi')
axes[0, 2].set_ylabel('Jumlah Penumpang')

# 4. Distribusi Usia (Age) vs Kelangsungan Hidup
sns.histplot(data=df, x='Age', hue='Survived', kde=True, ax=axes[1, 0], multiple='stack', palette='muted')
axes[1, 0].set_title('Distribusi Usia vs Kelangsungan Hidup')
axes[1, 0].set_xlabel('Usia')

# 5. Distribusi Tarif (Fare) vs Kelangsungan Hidup
sns.histplot(data=df, x='Fare', hue='Survived', kde=True, ax=axes[1, 1], multiple='stack', palette='coolwarm')
axes[1, 1].set_title('Distribusi Tarif (Fare) vs Kelangsungan Hidup')
axes[1, 1].set_xlabel('Tarif')
axes[1, 1].set_xlim(0, 300) # Batasi x-axis untuk visualisasi yang lebih baik karena ada outlier tarif tinggi

# 6. Heatmap Korelasi Fitur Numerik
numerical_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1, 2], cbar=True)
axes[1, 2].set_title('Matriks Korelasi Fitur Numerik')

plt.tight_layout()
plt.show()

# %% [markdown]
# # **5. Data Preprocessing**

# %% [markdown]
# Pada tahap ini, data preprocessing adalah langkah penting untuk memastikan kualitas data sebelum digunakan dalam model machine learning.
#
# Jika Anda menggunakan data teks, data mentah sering kali mengandung nilai kosong, duplikasi, atau rentang nilai yang tidak konsisten, yang dapat memengaruhi kinerja model. Oleh karena itu, proses ini bertujuan untuk membersihkan dan mempersiapkan data agar analisis berjalan optimal.
#
# Berikut adalah tahapan-tahapan yang bisa dilakukan, tetapi **tidak terbatas** pada:
# 1. Menghapus atau Menangani Data Kosong (Missing Values)
# 2. Menghapus Data Duplikat
# 3. Normalisasi atau Standarisasi Fitur
# 4. Deteksi dan Penanganan Outlier
# 5. Encoding Data Kategorikal
# 6. Binning (Pengelompokan Data)
#
# Cukup sesuaikan dengan karakteristik data yang kamu gunakan yah. Khususnya ketika kami menggunakan data tidak terstruktur.

# %% [markdown]
# ### 5.1 Menghapus Data Duplikat
# %%
duplicate_count = df.duplicated().sum()
print(f"Jumlah data duplikat: {duplicate_count}")
if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Data duplikat telah dihapus.")

# %% [markdown]
# ### 5.2 Menangani Missing Values
# %%
print("Missing values sebelum penanganan:")
print(df.isnull().sum())

# 1. Imputasi Embarked: Isi 2 nilai kosong dengan modus ('S')
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# 2. Imputasi Age: Isi nilai kosong menggunakan median berdasarkan Pclass dan Sex
df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))

# 3. Penanganan Cabin: Karena terlalu banyak kosong (77%), kita buat fitur indikator biner 'Has_Cabin'
df['Has_Cabin'] = df['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)

print("\nMissing values setelah penanganan:")
print(df.isnull().sum()[['Embarked', 'Age', 'Cabin']])

# %% [markdown]
# ### 5.3 Feature Engineering & Binning (Pengelompokan Data)
# %%
# 1. Menggabungkan SibSp dan Parch menjadi FamilySize
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# 2. Menentukan apakah penumpang bepergian sendiri
df['IsAlone'] = df['FamilySize'].apply(lambda x: 1 if x == 1 else 0)

# 3. Binning pada Age (Pengelompokan Umur)
age_bins = [0, 12, 18, 35, 60, 120]
age_labels = ['Child', 'Teenager', 'Young Adult', 'Adult', 'Elderly']
df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)

print("\nHasil Binning Umur (AgeGroup):")
print(df['AgeGroup'].value_counts())

# %% [markdown]
# ### 5.4 Penanganan Outlier pada Tarif (Fare)
# %%
# Kita gunakan visualisasi boxplot sebelum dan sesudah transformasi log
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(data=df, y='Fare', ax=axes[0], color='salmon')
axes[0].set_title('Distribusi Fare Sebelum Log Transformation')

# Log transformation untuk menangani skewness dan outlier
df['Fare_Log'] = np.log1p(df['Fare'])

sns.boxplot(data=df, y='Fare_Log', ax=axes[1], color='skyblue')
axes[1].set_title('Distribusi Fare Setelah Log Transformation')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.5 Encoding Data Kategorikal

# %%
df.columns 
# %%
# 1. Label encoding untuk Sex (male -> 0, female -> 1)
df['Sex_Code'] = df['Sex'].map({'male': 0, 'female': 1})

# 2. One-hot encoding untuk Embarked & AgeGroup
df = pd.get_dummies(df, columns=['Embarked', 'AgeGroup'], drop_first=True, dtype=int)

print("\nKolom setelah encoding:")
print(df.columns.tolist())

# %% [markdown]
# ### 5.6 Normalisasi / Standarisasi Fitur
# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fitur numerik yang akan distandarisasi
numerical_features = ['Age', 'Fare_Log', 'FamilySize']

# Lakukan standarisasi
df_scaled = df.copy()
df_scaled[numerical_features] = scaler.fit_transform(df[numerical_features])

print("\nHasil standarisasi beberapa baris pertama:")
print(df_scaled[numerical_features].head())

# %% [markdown]
# ### 5.7 Menyimpan Data Hasil Preprocessing
# %%
# Hapus kolom yang tidak lagi dibutuhkan untuk pemodelan
drop_cols = ['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Fare', 'SibSp', 'Parch']
df_model = df_scaled.drop(columns=drop_cols)

print("Dataframe siap digunakan untuk pemodelan:")
print(df_model.head())

# Simpan data yang sudah bersih ke file csv baru
df_model.to_csv('titanic_preprocessed.csv', index=False)
print("\nData preprocessed telah disimpan ke 'titanic_preprocessed.csv'")

