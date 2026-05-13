# LINEAR REGRESSION — Implementasi Step by Step
Dataset: California Housing (built-in sklearn, tidak perlu download)

#### ---- STEP 0: Import semua yang diperlukan ----
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size']  = 11

####  ---- STEP 1: Load dan Kenali Dataset ----
print("=" * 50)
print("STEP 1: MENGENAL DATASET")
print("=" * 50)

housing = fetch_california_housing()

####  Buat DataFrame agar lebih mudah dilihat
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['HouseValue'] = housing.target  # target: median harga rumah (x$100.000)

print("\n📌 Deskripsi Dataset:")
print(housing.DESCR[:800])  # tampilkan deskripsi singkat

print("\n📌 5 baris pertama:")
print(df.head())

print("\n📌 Statistik Deskriptif:")
print(df.describe().round(2))

print("\n📌 Apakah ada nilai kosong (missing values)?")
print(df.isnull().sum())

####  Penjelasan fitur:
fitur_info = {
    'MedInc'    : 'Median pendapatan rumah tangga di blok (dalam $10.000)',
    'HouseAge'  : 'Median usia rumah di blok (tahun)',
    'AveRooms'  : 'Rata-rata jumlah kamar per rumah tangga',
    'AveBedrms' : 'Rata-rata jumlah kamar tidur per rumah tangga',
    'Population': 'Populasi di blok tersebut',
    'AveOccup'  : 'Rata-rata jumlah penghuni per rumah tangga',
    'Latitude'  : 'Lintang geografis',
    'Longitude' : 'Bujur geografis',
    'HouseValue': 'TARGET: Median harga rumah (dalam $100.000)'
}
print("\n📌 Penjelasan Fitur:")
for f, d in fitur_info.items():
    print(f"  {f:12s}: {d}")


####  ---- STEP 2: Eksplorasi Visual ----
print("\n" + "=" * 50)
print("STEP 2: EKSPLORASI DATA (EDA)")
print("=" * 50)

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(df.columns):
    axes[i].hist(df[col], bins=40, color='steelblue', alpha=0.7, edgecolor='white')
    axes[i].set_title(f'Distribusi: {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frekuensi')

plt.suptitle('Distribusi Semua Variabel', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('01_distribusi.png', bbox_inches='tight')
plt.show()
print("✅ Plot distribusi tersimpan: 01_distribusi.png")

####  Heatmap Korelasi
plt.figure(figsize=(10, 8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))  # sembunyikan segitiga atas (duplikat)
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, linewidths=0.5,
            annot_kws={'size': 9})
plt.title('Heatmap Korelasi Antar Variabel\n(+1 = korelasi positif sempurna, -1 = negatif sempurna)',
          fontsize=12)
plt.tight_layout()
plt.savefig('02_heatmap_korelasi.png', bbox_inches='tight')
plt.show()
print("✅ Plot korelasi tersimpan: 02_heatmap_korelasi.png")

####  Scatter plot: fitur terpenting vs target
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
for i, col in enumerate(housing.feature_names):
    axes[i].scatter(df[col], df['HouseValue'], alpha=0.1, s=5, color='navy')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('HouseValue')
    axes[i].set_title(f'{col} vs HouseValue')
plt.suptitle('Hubungan Setiap Fitur dengan Target', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('03_scatter_fitur_target.png', bbox_inches='tight')
plt.show()


####  ---- STEP 3: Persiapan Data ----
print("\n" + "=" * 50)
print("STEP 3: PERSIAPAN DATA")
print("=" * 50)

X = df.drop('HouseValue', axis=1)
y = df['HouseValue']

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Total data    : {len(df)} baris")
print(f"Data Training : {len(X_train)} baris ({len(X_train)/len(df)*100:.0f}%)")
print(f"Data Testing  : {len(X_test)} baris  ({len(X_test)/len(df)*100:.0f}%)")

####  SCALING — kenapa penting?
print("\n📌 Kenapa perlu scaling?")
print("Sebelum scaling:")
print(X_train.describe().loc['mean'].round(2))
#### MedInc ~ 3.87, Population ~ 1400 → sangat berbeda skalanya!
#### Model bisa "bias" ke fitur dengan angka besar

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # ← PENTING: hanya transform, BUKAN fit_transform!

####  Kenapa test pakai .transform() bukan .fit_transform()?
####  Karena .fit() menghitung mean dan std dari data.
####  Kalau kita fit di test set, kita "bocorkan" informasi test ke model → data leakage!
print("\nSetelah scaling (semua fitur punya mean ≈ 0 dan std ≈ 1):")
print(pd.DataFrame(X_train_scaled, columns=housing.feature_names).describe().loc[['mean','std']].round(3))


####  ---- STEP 4: Training Model ----
print("\n" + "=" * 50)
print("STEP 4: TRAINING LINEAR REGRESSION")
print("=" * 50)

model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)
print("✅ Model berhasil dilatih!")

####  Lihat parameter yang dipelajari
print(f"\nIntercept (β₀): {model_lr.intercept_:.4f}")
print("\nKoefisien tiap fitur:")
koef_df = pd.DataFrame({
    'Fitur'      : housing.feature_names,
    'Koefisien'  : model_lr.coef_,
    'Abs_Koef'   : np.abs(model_lr.coef_)
}).sort_values('Koefisien', ascending=False)
print(koef_df.to_string(index=False))

print("""
📖 Cara baca koefisien:
  Koef positif → fitur ini meningkatkan harga
  Koef negatif → fitur ini menurunkan harga
  Koef besar   → fitur ini pengaruhnya kuat
  (ingat: karena sudah di-scaling, koefisien bisa langsung dibandingkan)
""")


####  ---- STEP 5: Prediksi & Evaluasi ----
print("\n" + "=" * 50)
print("STEP 5: EVALUASI MODEL")
print("=" * 50)

y_pred_lr = model_lr.predict(X_test_scaled)

mse  = mean_squared_error(y_test, y_pred_lr)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred_lr)
r2   = r2_score(y_test, y_pred_lr)

print(f"""
┌─────────────────────────────────────────────┐
│           HASIL EVALUASI LINEAR REGRESSION  │
├────────┬────────────┬───────────────────────┤
│ Metrik │   Nilai    │    Interpretasi        │
├────────┼────────────┼───────────────────────┤
│  MSE   │ {mse:>10.4f} │ Rata² kuadrat error   │
│  RMSE  │ {rmse:>10.4f} │ ±${rmse*100:.0f}k dari harga nyata │
│  MAE   │ {mae:>10.4f} │ ±${mae*100:.0f}k dari harga nyata │
│  R²    │ {r2:>10.4f} │ {r2*100:.1f}% variansi dijelaskan│
└────────┴────────────┴───────────────────────┘

📖 Interpretasi R²={r2:.4f}:
  Model ini mampu menjelaskan {r2*100:.1f}% dari variasi harga rumah.
  Sisanya {(1-r2)*100:.1f}% disebabkan faktor lain yang tidak ada di dataset.
""")


####  ---- STEP 6: Visualisasi Evaluasi ----
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

####  Plot 1: Aktual vs Prediksi
axes[0].scatter(y_test, y_pred_lr, alpha=0.2, s=8, color='steelblue')
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', lw=2, label='Prediksi Sempurna')
axes[0].set_xlabel('Nilai Aktual (x$100k)')
axes[0].set_ylabel('Nilai Prediksi (x$100k)')
axes[0].set_title(f'Aktual vs Prediksi\nR² = {r2:.4f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

####  Plot 2: Residual vs Prediksi (cek homoskedastisitas)
residuals = y_test - y_pred_lr
axes[1].scatter(y_pred_lr, residuals, alpha=0.2, s=8, color='coral')
axes[1].axhline(0, color='black', lw=1.5, linestyle='--')
axes[1].set_xlabel('Nilai Prediksi')
axes[1].set_ylabel('Residual (Aktual - Prediksi)')
axes[1].set_title('Residual Plot\n(Ideal: tersebar acak di sekitar 0)')
axes[1].grid(True, alpha=0.3)

####  Plot 3: Distribusi Residual (cek normalitas)
axes[2].hist(residuals, bins=50, color='mediumseagreen', alpha=0.7, edgecolor='white')
axes[2].axvline(0, color='red', lw=2, linestyle='--')
axes[2].set_xlabel('Residual')
axes[2].set_ylabel('Frekuensi')
axes[2].set_title(f'Distribusi Residual\n(Ideal: simetris di sekitar 0)')
axes[2].grid(True, alpha=0.3)

plt.suptitle('Diagnostik Model Linear Regression', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('04_evaluasi_linear_regression.png', bbox_inches='tight')
plt.show()
print("✅ Plot evaluasi tersimpan: 04_evaluasi_linear_regression.png")

####  ---- STEP 7: Visualisasi Koefisien ----
plt.figure(figsize=(9, 5))
colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in koef_df['Koefisien']]
bars = plt.barh(koef_df['Fitur'], koef_df['Koefisien'], color=colors, alpha=0.85)
plt.axvline(0, color='black', lw=1)
plt.xlabel('Nilai Koefisien (setelah scaling)')
plt.title('Koefisien Linear Regression\n(Hijau = pengaruh positif, Merah = negatif)')
plt.grid(axis='x', alpha=0.3)
for bar, val in zip(bars, koef_df['Koefisien']):
    plt.text(val + 0.01 if val >= 0 else val - 0.01,
             bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center',
             ha='left' if val >= 0 else 'right', fontsize=9)
plt.tight_layout()
plt.savefig('05_koefisien_lr.png', bbox_inches='tight')
plt.show()
