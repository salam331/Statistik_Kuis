import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Membaca dataset
df = pd.read_csv('heart_full.csv')

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

# Mengekstrak koefisien regresi
coefficients = model_full.params

# Menampilkan koefisien regresi untuk setiap variabel
print("\nKoefisien Regresi:")
for i, coef in enumerate(coefficients):
    if i == 0:
        print(f"Intercept: {coef:.4f}")
    else:
        print(f"X{i}: {coef:.4f}")