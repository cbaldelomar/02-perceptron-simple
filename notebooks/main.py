# %%
import sys

sys.path.append("../libs")

import neuronas
import importlib
import pandas as pd
import matplotlib.pyplot as plt

importlib.reload(neuronas)

from neuronas import NeuronaPerceptron

# %%
%matplotlib inline

df_casas = pd.read_csv("../data/houses1_dataset.csv")

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

ax1.hist(df_casas.Area)
ax2.hist(df_casas.Precio)

#colores = df_casas['Decision'].map({'Comprar': 'blue', 'No_Comprar': 'red'})
ax3.scatter(df_casas.Area, df_casas.Precio) #, c=colores

fig.get_tight_layout()
plt.show()

# %%
del fig, ax1, ax2, ax3
import gc
gc.collect()

# %%
df_casas['Precio_'] = (df_casas.Precio - min(df_casas.Precio)) / (max(df_casas.Precio) - min(df_casas.Precio))

df_casas['Area_'] = (df_casas.Area - min(df_casas.Area)) / (max(df_casas.Area) - min(df_casas.Area))

df_casas['Decision_'] = df_casas.Decision.map({'Comprar': 1, 'No_Comprar': 0})
# %%

neurona = NeuronaPerceptron(entradas=2, salida=1)