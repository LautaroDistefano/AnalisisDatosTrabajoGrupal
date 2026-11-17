import os
import pandas as pd

carpeta_bases = "./Bases_EPH"
aglomerados_elegidos = [32, 2]
cols_to_keep = [
    'ANO4', 'TRIMESTRE', 'AGLOMERADO', 'ESTADO', 
    'P21', 'P47T', 'CH04', 'CH06', 'NIVEL_ED', 
    'PP04B_COD', 'PP04D_COD', 'PP3E_TOT', 'PP07H', 
    'PONDERA', 'PONDIIO'
]

dfs = []
archivos = [f for f in os.listdir(carpeta_bases) if f.endswith(('.txt', '.csv')) and 'individual' in f.lower()]
print(f"Se encontraron {len(archivos)} bases para procesar.\n")

for archivo in archivos:
    ruta_completa = os.path.join(carpeta_bases, archivo)
    print(f"Procesando: {archivo}")
    
    try:
        # 1. Detección de separador dinámica leyendo la primera línea del archivo
        with open(ruta_completa, 'r', encoding='utf-8', errors='ignore') as f:
            primera_linea = f.readline()
            separador = ';' if ';' in primera_linea else ','
            
        # 2. Leemos con Pandas usando el separador correcto detectado
        df = pd.read_csv(ruta_completa, sep=separador, low_memory=False)
        
        # Pasar nombres de columnas a mayúsculas
        df.columns = df.columns.str.upper()
        
        # Filtrar por aglomerado
        if 'AGLOMERADO' in df.columns:
            df = df[df['AGLOMERADO'].isin(aglomerados_elegidos)]
            
        # Quedarnos solo con las columnas necesarias
        cols_presentes = [c for c in cols_to_keep if c in df.columns]
        df = df[cols_presentes]
        
        dfs.append(df)
        
    except Exception as e:
        print(f"Error crítico al leer {archivo}: {e}")

print("\nUnificando todas las bases...")
df_final = pd.concat(dfs, ignore_index=True)

cols_numericas = ['P21', 'P47T', 'CH06', 'PP3E_TOT', 'PONDERA', 'PONDIIO', 'ESTADO', 'NIVEL_ED']
for col in cols_numericas:
    if col in df_final.columns:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

print(f"\n¡Proceso finalizado! Filas totales: {len(df_final)}")
df_final.to_csv("eph_2016_2026_filtrada.csv", index=False)
print("Archivo 'eph_2016_2026_filtrada.csv' creado con éxito.")