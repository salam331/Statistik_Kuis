import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Membaca dataset
df = pd.read_csv('heart_full.csv')

# 1. Analisis dengan 5 variabel numerik menggunakan Python
X_sample = df[['sex', 'age', 'restecg', 'trestbps', 'chol']]
y_sample = df['exang']

# Menambahkan konstanta untuk termasuk intercept
X_sample = sm.add_constant(X_sample)

# Membuat model regresi
model_sample = sm.OLS(y_sample, X_sample).fit()

# Menampilkan hasil regresi
print("Hasil Regresi untuk 5 Variabel Numerik:")
print(model_sample.summary())

# 2. Analisis dengan 10 variabel numerik menggunakan Python
X_full = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak','slope','ca','thal','target']]
y_full = df['exang']

# Menambahkan konstanta untuk termasuk intercept
X_full = sm.add_constant(X_full)

# Membuat model regresi
model_full = sm.OLS(y_full, X_full).fit()

# Menampilkan hasil regresi
print("\nHasil Regresi untuk 10 Variabel Numerik:")
print(model_full.summary())

# 3. Uji Parsial (t-test) untuk semua variabel pada data 100 sample
t_test = model_full.t_test(np.eye(len(model_full.params)))
t_stat = t_test.tvalue
p_values = t_test.pvalue

print("\nUji Parsial (t-test) untuk semua variabel pada data 100 sample:")
for i, var in enumerate(model_full.params.index):
    print(f"{var}: T-Stat: {t_stat[i]}, P-Value: {p_values[i]}")
    if p_values[i] < 0.05:
        print(f"Variabel '{var}' signifikan secara parsial.")
    else:
        print(f"Variabel '{var}' tidak signifikan secara parsial.")



# 4. Uji Simultan Regresi (F-test) untuk model dengan 10 variabel numerik
f_test = model_full.f_test("age = sex = cp = trestbps = chol = fbs = restecg = thalach = exang = oldpeak = slope = ca = thal = target = 0")
f_stat = f_test.statistic
f_p_value = f_test.pvalue

print(f"\nUji Simultan Regresi (F-test) untuk model dengan 10 variabel numerik:")
print(f"F-Stat: {f_stat}, P-Value: {f_p_value}")
if f_p_value < 0.05:
    print("Model regresi secara keseluruhan signifikan.")
else:
    print("Model regresi secara keseluruhan tidak signifikan.")



# 5. Uji Kebaikan Model menggunakan R-squared
r_squared = model_full.rsquared
print(f"\nR-Squared (Koefisien Determinasi) untuk model dengan 10 variabel numerik:")
print(f"R-Squared: {r_squared}")
