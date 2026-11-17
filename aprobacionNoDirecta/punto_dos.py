import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Omitir warnings visuales de Seaborn/Pandas
warnings.filterwarnings('ignore')

# 1. Cargar la base unificada
df = pd.read_csv("eph_2016_2026_filtrada.csv")

# 2. RECODIFICACIÓN DE VARIABLES (Feature Engineering)
# Sexo
df['SEXO'] = df['CH04'].map({1: 'Varón', 2: 'Mujer'})

# Nivel Educativo
dicc_educ = {
    1: '1_Primaria Incompleta', 2: '2_Primaria Completa',
    3: '3_Secundaria Incompleta', 4: '4_Secundaria Completa',
    5: '5_Superior Incompleta', 6: '6_Superior Completa', 7: '7_Sin Instrucción'
}
df['EDUCACION'] = df['NIVEL_ED'].map(dicc_educ)

# Calificación Ocupacional (CNO - PP04D_COD)
# El 5to dígito indica la calificación. Rellenamos con ceros a la izquierda.
df['PP04D_COD'] = df['PP04D_COD'].fillna(0).astype(int).astype(str).str.zfill(5)
df['CALIFICACION'] = df['PP04D_COD'].str[4].map({
    '1': 'Profesional', '2': 'Técnico', 
    '3': 'Operativo', '4': 'No calificado'
})

# Grupos Etarios
df['GRUPO_EDAD'] = pd.cut(df['CH06'], bins=[15, 29, 59, 100], labels=['Jóvenes (15-29)', 'Adultos (30-59)', 'Mayores (60+)'])

# Mapeo de Aglomerados (Ajustá los nombres según los que hayas elegido: 32 CABA, 2 La Plata)
dicc_aglo = {32: 'CABA', 2: 'Gran La Plata', 33: 'Partidos GBA', 13: 'Gran Córdoba'}
df['NOMBRE_AGLO'] = df['AGLOMERADO'].map(dicc_aglo)


# 3. AJUSTE POR INFLACIÓN (Deflactación con datos reales INDEC)
# Índice de Precios al Consumidor (IPC INDEC) Nivel General. Base Diciembre 2016 = 100
ipc_real = {
    2016: 100.0,
    2017: 124.8,
    2018: 184.2,
    2019: 283.3,
    2020: 385.6,
    2021: 581.8,
    2022: 1133.4,
    2023: 3529.5,
    2024: 7687.3,
    2025: 10108.7,
    2026: 12514.1
}

# Convertimos a DataFrame para hacer el cruce (merge)
df_ipc = pd.DataFrame(list(ipc_real.items()), columns=['ANO4', 'INDICE_PRECIOS'])
df = df.merge(df_ipc, on='ANO4', how='left')

# Fórmula de deflactación: (Ingreso Nominal / IPC del período) * IPC Base
ipc_base_2026 = 12514.1
df['P21_REAL'] = (df['P21'] / df['INDICE_PRECIOS']) * ipc_base_2026


# 4. EVOLUCIÓN HISTÓRICA DE LAS TASAS DE MERCADO DE TRABAJO (Usando Ponderadores)
def calcular_tasas(grupo):
    pob_total = grupo['PONDERA'].sum()
    ocupados = grupo.loc[grupo['ESTADO'] == 1, 'PONDERA'].sum()
    desocupados = grupo.loc[grupo['ESTADO'] == 2, 'PONDERA'].sum()
    pea = ocupados + desocupados # Población Económicamente Activa
    
    tasa_actividad = (pea / pob_total) * 100 if pob_total > 0 else 0
    tasa_empleo = (ocupados / pob_total) * 100 if pob_total > 0 else 0
    tasa_desocupacion = (desocupados / pea) * 100 if pea > 0 else 0
    
    return pd.Series({
        'Tasa_Actividad': tasa_actividad,
        'Tasa_Empleo': tasa_empleo,
        'Tasa_Desocupacion': tasa_desocupacion
    })

tasas_historicas = df.groupby(['ANO4', 'NOMBRE_AGLO']).apply(calcular_tasas).reset_index()


# 5. EVOLUCIÓN DE MEDIDAS DE TENDENCIA Y POSICIÓN (Ingreso Real)
# Filtramos solo ocupados con ingresos válidos mayores a 0
ocupados_ingresos = df[(df['ESTADO'] == 1) & (df['P21_REAL'] > 0)]

ingresos_historicos = ocupados_ingresos.groupby(['ANO4', 'NOMBRE_AGLO'])['P21_REAL'].agg(
    Media='mean',
    Mediana='median', # Q2
    Q1=lambda x: x.quantile(0.25),
    Q3=lambda x: x.quantile(0.75)
).reset_index()


# 6. ANÁLISIS MULTIVARIADO CRUZADO (Desocupación por Sexo y Año)
desoc_sexo = df.groupby(['ANO4', 'SEXO']).apply(calcular_tasas).reset_index()


# 7. TABLA DE RAMA DE ACTIVIDAD (Requisito PP04B_COD)
print("--- MEDIANA DEL INGRESO REAL POR RAMA DE ACTIVIDAD (Corte General) ---")
# Agrupamos por los primeros 2 dígitos de la rama de actividad para simplificar
ocupados_ingresos['RAMA_AGRUPADA'] = ocupados_ingresos['PP04B_COD'].fillna(0).astype(int).astype(str).str[:2]
ingreso_por_rama = ocupados_ingresos.groupby('RAMA_AGRUPADA')['P21_REAL'].median().sort_values(ascending=False).head(5)
print("Top 5 Ramas con mejor Mediana Salarial Real:\n", ingreso_por_rama)


# ==========================================
# SECCIÓN DE VISUALIZACIÓN DE DATOS (Plots)
# ==========================================
sns.set_theme(style="whitegrid")

# Gráfico A: Evolución de la Tasa de Desocupación por Aglomerado
plt.figure(figsize=(10, 5))
sns.lineplot(data=tasas_historicas, x='ANO4', y='Tasa_Desocupacion', hue='NOMBRE_AGLO', marker='o')
plt.title('Evolución de la Tasa de Desocupación (2016-2026)')
plt.ylabel('Tasa de Desocupación (%)')
plt.xlabel('Año')
plt.xticks(range(2016, 2027))
plt.tight_layout()
plt.show()

# Gráfico B: Evolución de la Mediana del Ingreso REAL
plt.figure(figsize=(10, 5))
sns.lineplot(data=ingresos_historicos, x='ANO4', y='Mediana', hue='NOMBRE_AGLO', marker='s')

# Agregamos sombra entre Q1 y Q3 para medir desigualdad salarial
aglo_principal = df['NOMBRE_AGLO'].dropna().unique()[0]
data_sombra = ingresos_historicos[ingresos_historicos['NOMBRE_AGLO'] == aglo_principal]
plt.fill_between(data_sombra['ANO4'], data_sombra['Q1'], data_sombra['Q3'], alpha=0.2, label=f'Rango Intercuartílico ({aglo_principal})')

plt.title('Evolución del Ingreso Principal Real (Mediana a precios constantes 2026)')
plt.ylabel('Ingreso Real ($)')
plt.xlabel('Año')
plt.legend()
plt.tight_layout()
plt.show()

# Gráfico C: Análisis Multivariado - Ingreso por Calificación y Sexo
plt.figure(figsize=(10, 6))
# Usamos errorbar=None en lugar de ci=None (actualización reciente de seaborn)
sns.barplot(data=ocupados_ingresos, x='CALIFICACION', y='P21_REAL', hue='SEXO', estimator=np.median, errorbar=None, order=['Profesional', 'Técnico', 'Operativo', 'No calificado'])
plt.title('Mediana del Ingreso Real por Calificación Ocupacional y Sexo')
plt.ylabel('Ingreso Real ($)')
plt.xlabel('Calificación Ocupacional')
plt.tight_layout()
plt.show()