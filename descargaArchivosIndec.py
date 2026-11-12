import pandas as pd
import pyeph
import time

# 1. Configuración de parámetros
years = range(2016, 2027)
quarters = [1, 2, 3, 4]

# Códigos de los aglomerados elegidos (Ejemplo: 33 = GBA, 13 = Gran Córdoba)
# Cámbialos según lo que defina tu grupo
aglomerados_elegidos = [33, 13] 

# Variables estrictamente necesarias para optimizar RAM
cols_to_keep = [
    'ANO4', 'TRIMESTRE', 'AGLOMERADO', 'ESTADO', 
    'P21', 'P47T', 'CH04', 'CH06', 'NIVEL_ED', 
    'PP04B_COD', 'PP04D_COD', 'PP3E_TOT', 'PP07H', 
    'PONDERA', 'PONDIIO'
]

dfs = []

# 2. Bucle de descarga
for year in years:
    for q in quarters:
        # Los trimestres 3 y 4 de 2026 no existen aún, los saltamos
        if year == 2026 and q > 2:
            continue
            
        print(f"Descargando EPH: {year} - Trimestre {q}...")
        
        try:
            # Descarga de la base usando la librería pyeph
            df_trim = pyeph.obtener(data="eph", ano=year, periodo=q, tipo_base='individual')
            
            # Filtrado geográfico inmediato
            df_filt = df_trim[df_trim['AGLOMERADO'].isin(aglomerados_elegidos)].copy()
            
            # Filtramos solo las columnas que necesitamos y que realmente existan en esa base
            cols_presentes = [c for c in cols_to_keep if c in df_filt.columns]
            df_filt = df_filt[cols_presentes]
            
            dfs.append(df_filt)
            
            # Pausa para no saturar el servidor del INDEC
            time.sleep(1) 
            
        except Exception as e:
            print(f"Error o trimestre no disponible ({year} T{q}): {e}")

# 3. Consolidación
print("\nConcatenando los trimestres...")
df_final = pd.concat(dfs, ignore_index=True)

# Armonizar tipos de datos numéricos
cols_numericas = ['P21', 'P47T', 'CH06', 'PP3E_TOT', 'PONDERA', 'PONDIIO', 'ESTADO']
for col in cols_numericas:
    if col in df_final.columns:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

print(f"\n¡Descarga finalizada! Filas totales: {len(df_final)}")
print(df_final.head())

# Descomentá la siguiente línea para guardar la base lista en tu PC
# df_final.to_csv("eph_2016_2026_filtrada.csv", index=False)