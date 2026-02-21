import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# 1. Cargar archivo EDF
raw = mne.io.read_raw_edf("Subject00_1.edf", preload=True)

# 2. Obtener datos en formato numpy
data = raw.get_data()

# 3. Transponer para sklearn
X = data.T

print("Forma de los datos:", X.shape)

# 4. Aplicar ICA
ica = FastICA(n_components=5, random_state=42)
S = ica.fit_transform(X)

# 5. Graficar componentes
plt.figure(figsize=(12,8))

for i in range(5):
    plt.subplot(5,1,i+1)
    plt.plot(S[:, i])
    plt.title(f"Componente Independiente {i+1}")

plt.tight_layout()
plt.show()