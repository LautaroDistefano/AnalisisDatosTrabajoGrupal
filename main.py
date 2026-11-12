import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar la base limpia
df = pd.read_csv("eph_2016_2026_filtrada.csv")

print("--- 1. ANÁLISIS DE NO RESPUESTA ---")
# En la EPH, el código -9 en P21 (Ingreso principal) o P47T (Ingreso total) significa "No sabe / No contesta".
# Solo nos interesa medir la no respuesta en las personas que TRABAJAN (Ocupados: ESTADO == 1).
ocupados = df[df['ESTADO'] == 1].copy()
total_ocupados = len(ocupados)

no_respuesta_p21 = len(ocupados[ocupados['P21'] == -9])
porcentaje_nr = (no_respuesta_p21 / total_ocupados) * 100

print(f"Total de Ocupados en la muestra: {total_ocupados}")
print(f"Ocupados que no declararon ingresos (valor -9): {no_respuesta_p21}")
print(f"Tasa de No Respuesta en P21: {porcentaje_nr:.2f}%\n")


print("--- 2. ANÁLISIS UNIVARIADO (Medidas de Tendencia Central y Dispersión) ---")
# Filtramos los valores inválidos de edad e ingresos para no distorsionar las medias
edades_validas = df[df['CH06'] >= 0]['CH06']
horas_validas = ocupados[(ocupados['PP3E_TOT'] >= 0) & (ocupados['PP3E_TOT'] != 999)]['PP3E_TOT']

print("Estadísticas descriptivas de Edad (CH06):")
print(edades_validas.describe().round(2))

print("\nEstadísticas descriptivas de Horas Trabajadas (PP3E_TOT):")
print(horas_validas.describe().round(2))


print("\n--- 3. DETECCIÓN DE VALORES ATÍPICOS (OUTLIERS) ---")
# Analizamos los ingresos estrictamente mayores a 0 para ver su distribución real
ingresos_validos = ocupados[ocupados['P21'] > 0]['P21']

# Aplicamos el criterio estadístico del Rango Intercuartílico (RIQ)
Q1 = ingresos_validos.quantile(0.25)
Q3 = ingresos_validos.quantile(0.75)
RIQ = Q3 - Q1

# Límite superior para detectar atípicos (Q3 + 1.5 * RIQ)
limite_superior = Q3 + 1.5 * RIQ
outliers = ingresos_validos[ingresos_validos > limite_superior]
porcentaje_outliers = (len(outliers) / len(ingresos_validos)) * 100

print(f"Mediana de ingresos válidos nominales: ${ingresos_validos.median():.2f}")
print(f"Límite superior (umbral de atípicos): ${limite_superior:.2f}")
print(f"Cantidad de registros atípicos (extremo superior): {len(outliers)} ({porcentaje_outliers:.2f}%)")

# Comparativa por aglomerado
print("\n--- COMPARATIVA POR AGLOMERADO ---")
# 1. No respuesta por aglomerado
nr_aglo = ocupados.groupby('AGLOMERADO').apply(
    lambda x: (len(x[x['P21'] == -9]) / len(x)) * 100
).rename("Tasa No Respuesta (%)")
print(nr_aglo.round(2))

# 2. Mediana de ingresos válidos por aglomerado
ingresos_aglo = ocupados[ocupados['P21'] > 0].groupby('AGLOMERADO')['P21'].median().rename("Mediana Ingreso Nominal")
print("\n", ingresos_aglo)

# --- VISUALIZACIÓN ---
# Gráfico 1: Boxplot de ingresos para visualizar los outliers
plt.figure(figsize=(10, 5))
sns.boxplot(x=ingresos_validos, color='skyblue')
plt.title("Distribución del Ingreso Principal (P21) y Valores Atípicos")
plt.xlabel("Ingreso Nominal ($)")
plt.ticklabel_format(style='plain', axis='x') # Evita la notación científica (1e6)
plt.tight_layout()
plt.show()

# Gráfico 2: Histograma de Edades
plt.figure(figsize=(10, 5))
sns.histplot(edades_validas, bins=20, kde=True, color='coral')
plt.title("Distribución de Edades en la Muestra")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()