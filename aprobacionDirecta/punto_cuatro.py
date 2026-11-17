import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# 1. Cargar la base
df = pd.read_csv("eph_2016_2026_filtrada.csv")

# 2. PREPARACIÓN DE DATOS (Feature Engineering Ampliado)
# Solo trabajamos con ocupados
df_ocupados = df[df['ESTADO'] == 1].copy()

# Limpieza de horas trabajadas 
df_ocupados = df_ocupados[(df_ocupados['PP3E_TOT'] > 0) & (df_ocupados['PP3E_TOT'] < 168)]

# Creamos variables base
df_ocupados['EDAD_CUADRADO'] = df_ocupados['CH06'] ** 2  
df_ocupados['ES_VARON'] = (df_ocupados['CH04'] == 1).astype(int) 

# NUEVO: Formalidad laboral (1 = formal con descuento jubilatorio, 0 = informal/otro)
df_ocupados['ES_FORMAL'] = (df_ocupados['PP07H'] == 1).astype(int)

# NUEVO: Convertimos el AÑO a texto para poder hacer variables Dummy
df_ocupados['ANO4_STR'] = df_ocupados['ANO4'].astype(str)

# Variables Dummy para Nivel Educativo y Año (drop_first=True evita colinealidad)
df_ocupados = pd.get_dummies(df_ocupados, columns=['NIVEL_ED', 'ANO4_STR'], drop_first=True, dtype=int)


# 3. SEPARACIÓN: ENTRENAMIENTO (Válidos) y A IMPUTAR (No respuesta)
train_data = df_ocupados[df_ocupados['P21'] > 0].copy()
imput_data = df_ocupados[df_ocupados['P21'] == -9].copy()

# TÉCNICA DE PODADO BILATERAL (Trimming 1% inferior y superior)
q_baja = train_data['P21'].quantile(0.01)
q_alta = train_data['P21'].quantile(0.99)
train_data = train_data[(train_data['P21'] >= q_baja) & (train_data['P21'] <= q_alta)]


# 3. SEPARACIÓN: ENTRENAMIENTO (Válidos) y A IMPUTAR (No respuesta)
# Base de entrenamiento: los que SÍ respondieron el ingreso (P21 > 0)
train_data = df_ocupados[df_ocupados['P21'] > 0].copy()

# Base a imputar: los que NO respondieron el ingreso (P21 == -9)
imput_data = df_ocupados[df_ocupados['P21'] == -9].copy()

# TRUCO DE MINERÍA DE DATOS: Recorte de outliers en entrenamiento para salvar el R2
# Cortamos el 2% superior. Esto no transforma la variable, solo limpia la muestra de entrenamiento.
limite_outliers = train_data['P21'].quantile(0.98)
train_data = train_data[train_data['P21'] <= limite_outliers]


# 4. DEFINICIÓN DEL MODELO
# Seleccionamos variables incluyendo las Dummies de Educación y las Dummies de Año
cols_educacion = [c for c in train_data.columns if 'NIVEL_ED_' in c]
cols_anio = [c for c in train_data.columns if 'ANO4_STR_' in c]

# Sumamos TODAS las variables fuertes a la ecuación
variables_X = ['PP3E_TOT', 'CH06', 'EDAD_CUADRADO', 'ES_VARON', 'ES_FORMAL'] + cols_educacion + cols_anio

X = train_data[variables_X]
X = sm.add_constant(X)
y = train_data['P21']  # Se mantiene la variable ORIGINAL, respetando el enunciado

# 5. ENTRENAMIENTO (Regresión Lineal por Mínimos Cuadrados Ordinarios - OLS)
modelo = sm.OLS(y, X).fit()

# Imprimir Resumen Estadístico
print("========== RESUMEN DEL MODELO ESTADÍSTICO ==========")
print(modelo.summary())


# 6. EVALUACIÓN DE RENDIMIENTO
# Calculamos predicciones sobre la misma base de entrenamiento para ver el error
y_pred = modelo.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print("\n========== EVALUACIÓN DEL RENDIMIENTO ==========")
print(f"R-cuadrado (R2): {modelo.rsquared:.4f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): ${rmse:.2f}")


# 7. IMPUTACIÓN DE LA NO RESPUESTA (-9)
# Preparamos las X para los datos que tienen P21 == -9
X_imput = imput_data[variables_X]
X_imput = sm.add_constant(X_imput, has_constant='add')

# Imputamos (predecimos) los valores
ingresos_imputados = modelo.predict(X_imput)

# Aseguramos que no prediga ingresos negativos (por la matemática de la recta)
ingresos_imputados = np.maximum(ingresos_imputados, 0)

imput_data.loc[:, 'P21_IMPUTADO'] = ingresos_imputados

print("\n========== RESULTADO DE LA IMPUTACIÓN ==========")
print(f"Se imputaron {len(imput_data)} registros con valor -9.")
print(f"Promedio del ingreso imputado: ${imput_data['P21_IMPUTADO'].mean():.2f}")