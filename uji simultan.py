import pandas as pd
import statsmodels.api as sm
from scipy import stats

df = pd.read_csv('heart_full.csv')
# Membaca dataset

# Analisis dengan 10 variabel numerik menggunakan Python
X_full = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak','slope','ca','thal','target']]
y_full = df['exang']

# Menambahkan konstanta untuk termasuk intercept
X_full = sm.add_constant(X_full)

# Membuat model regresi
model_full = sm.OLS(y_full, X_full).fit()

# Menampilkan hasil regresi
print("Hasil Regresi untuk 10 Variabel Numerik:")
print(model_full.summary())

# Perhitungan manual uji F
k = len(X_full.columns) - 1  # jumlah variabel prediktor
n = len(y_full)  # jumlah observasi
ssr = model_full.ess  # sum of squares regression
sse = model_full.ssr  # sum of squares residual
df_reg = k  # derajat kebebasan regresi
df_res = n - k - 1  # derajat kebebasan residual

# Menghitung mean square regression dan mean square residual
msr = ssr / df_reg
mse = sse / df_res

# Menghitung nilai uji F dan p-value
F_statistic = msr / mse
p_value = stats.f.sf(F_statistic, df_reg, df_res)

# Menampilkan hasil uji F
print(f"\nUji Simultan Regresi (F-Test):")
print(f"F-Statistic: {F_statistic:.4f}")
print(f"P-Value: {p_value:.4f}")

# Uji hipotesis
alpha = 0.05
if p_value < alpha:
    print("Model regresi secara keseluruhan signifikan pada tingkat signifikansi 0.05")
else:
    print("Model regresi secara keseluruhan tidak signifikan pada tingkat signifikansi 0.05")
