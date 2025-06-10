import json
import pandas as pd
from datetime import datetime

def extract_data_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = []
    
    # Iterar por años, meses y registros
    for year, months in data.items():
        for month, entries in months.items():
            for entry in entries:
                try:
                    # Extraer datos principales
                    fecha = datetime.strptime(entry['fecha'], '%Y-%m-%d %H:%M:%S')
                    datos = entry['datos']
                    pred = datos.get('prediccion', {})
                    
                    # Extraer datos de plantas (contar unidades en avería/mantenimiento)
                    plantas = datos.get('plantas', {})
                    averias = plantas.get('averia', [])
                    mantenimientos = plantas.get('mantenimiento', [])
                    
                    record = {
                        'fecha': fecha,
                        'year': year,
                        'month': month,
                        'disponibilidad': pred.get('disponibilidad'),
                        'demanda_maxima': pred.get('demanda_maxima'),
                        'afectacion': pred.get('afectacion'),
                        'deficit': pred.get('deficit'),
                        'respaldo': 1 if pred.get('respaldo') else 0,
                        'horario_pico': pred.get('horario_pico', ''),
                        'unidades_averia': sum(len(unit['unidades']) for unit in averias if unit['unidades']),
                        'unidades_mantenimiento': sum(len(unit['unidades']) for unit in mantenimientos if unit['unidades']),
                        'limitacion_termica': plantas.get('limitacion_termica', {}).get('mw_afectados'),
                        'motores_impacto': datos.get('distribuida', {}).get('motores_con_problemas', {}).get('impacto_mw')
                    }
                    
                    records.append(record)
                except Exception as e:
                    print(f"Error procesando entrada: {e}")
                    continue
    
    return pd.DataFrame(records)

def clean_and_save(df, output_path):
    # Limpieza básica
    df = df.dropna(subset=['deficit'])  # Eliminar filas sin target
    df['deficit'] = df['deficit'].fillna(0)  # Asumir 0 si hay NaN pero se conservó
    
    # Codificar variables categóricas
    df['horario_pico'] = df['horario_pico'].astype('category').cat.codes
    
    # Ordenar por fecha
    df = df.sort_values('fecha').reset_index(drop=True)
    
    # Guardar a CSV
    df.to_csv(output_path, index=False)
    print(f"Datos guardados en {output_path} con {len(df)} registros")
    return df

if __name__ == "__main__":
    # Configuración
    INPUT_JSON = "datos_electricos_organizados.json"
    OUTPUT_CSV = "datos_electricos.csv"
    
    # Procesamiento
    print("Procesando datos...")
    df = extract_data_from_json(INPUT_JSON)
    clean_and_save(df, OUTPUT_CSV)
    
    # Mostrar preview
    print("\nPreview de los datos:")
    print(df.head())