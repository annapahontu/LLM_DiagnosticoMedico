import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#Seleccionamos la carpeta donde se encuentran los archivos CSV
home_dir = os.path.expanduser('~')
ruta_descargas = os.path.join(home_dir, 'Downloads')
if not os.path.exists(ruta_descargas):
    ruta_descargas = os.path.join(home_dir, 'Descargas')

archivos = {
    'base': 'metricas_modelo_base_completo.csv',
    'ft_07': 'metricas_modelo_finetuned_completo (1).csv',
    'ft_05': 'metricas_modelo_finetuned_completo_temp05.csv'
}

rutas = {k: os.path.join(ruta_descargas, v) for k, v in archivos.items()}

for clave, ruta in rutas.items():
    if not os.path.exists(ruta):
        print(f"Error: No se encontró el archivo '{archivos[clave]}' en {ruta_descargas}")
        exit()

#Leemos los csv y preparamos los datos para las gráficas
df_base = pd.read_csv(rutas['base'])
df_ft_07 = pd.read_csv(rutas['ft_07'])
df_ft_05 = pd.read_csv(rutas['ft_05'])

df_base['Modelo'] = 'Base (Llama 3)'
df_ft_07['Modelo'] = 'Fine-Tuned (Temp 0.7)'
df_ft_05['Modelo'] = 'Fine-Tuned (Temp 0.5)'

df_completo = pd.concat([df_base, df_ft_07, df_ft_05], ignore_index=True)
df_completo['Caso_Clinico'] = df_completo['ID_Case'].apply(lambda x: x.split('_Attempt_')[0])


#Generamos las tablas de resultados promedio para cada modelo
for modelo in df_completo['Modelo'].unique():
    df = df_completo[df_completo['Modelo'] == modelo]
    
    bertscore = df['Metrica_BERTScore_F1'].mean()
    keyword_acc = df['Metrica_Acierto_PalabrasClave'].mean() * 100
    rouge_l = df['Metrica_ROUGE_L_F1'].mean()
    struct_acc = df['Metrica_Acierto_Estructurado'].mean() * 100
    format_rate = df['Metrica_Formato_Correcto'].mean() * 100
    
    tasa_alucinacion_principal = (1 - df['Metrica_Acierto_PalabrasClave'].mean()) * 100 
    media_alt_diag = df['Metrica_Num_Alt_Diagnosticos'].mean()
    pct_con_alucinacion = (df['Metrica_Num_Alt_Diagnosticos'] > 0).mean() * 100

    resultados_dicc = {
        '  1. BERTScore (Semántica) F1': f"{bertscore:.4f}",
        '  2. Acierto Palabras Clave': f"{keyword_acc:.2f}%",
        '  3. ROUGE-L F1 (Media)': f"{rouge_l:.4f}",
        '  4. Precisión Estructurada': f"{struct_acc:.2f}%",
        '  5. Tasa Formato Correcto': f"{format_rate:.2f}%",
        '  6. Tasa Alucinación Diag. Principal': f"{tasa_alucinacion_principal:.2f}%",
        '  7. Media Diagnósticos Alternativos': f"{media_alt_diag:.2f}",
        '  8. Respuestas con ≥1 Diag. Alt.': f"{pct_con_alucinacion:.2f}%"
    }
    
    df_tabla = pd.DataFrame(list(resultados_dicc.items()), columns=['Métrica Médica Evaluada', 'Valor Obtenido'])

    fig, ax = plt.subplots(figsize=(10, 4.5)) 
    ax.axis('off') 
    
    tabla = ax.table(
        cellText=df_tabla.values, 
        colLabels=df_tabla.columns, 
        loc='center',
        colWidths=[0.7, 0.3], 
        bbox=[0.05, 0.05, 0.9, 0.85] 
    )
    
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(12) 
    
    
    for (row, col), cell in tabla.get_celld().items():
        cell.set_edgecolor('#a0a0a0') # Bordes grises sutiles
        
        
        if row == 0:
            cell.set_facecolor('#e6e6e6')
            cell.set_text_props(weight='bold', ha='center') 
        else:
            if col == 0:
                cell.set_text_props(ha='left') 
            else:
                cell.set_text_props(ha='center') 
                
    plt.title(f'Resultados Promedio: {modelo}', fontweight="bold", fontsize=15, y=0.95)
    
    nombre_archivo = f"tabla_medias_{modelo.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')}.png"
    ruta_guardado = os.path.join(ruta_descargas, nombre_archivo)
    
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    plt.close() 

print(f"Tablas completas guardadas en: {ruta_descargas}!")

#Generamos las gráficas para cada métrica, comparando los modelos entre sí y guardándolas en PNG
df_comp_base_ft = df_completo[df_completo['Modelo'].isin(['Base (Llama 3)', 'Fine-Tuned (Temp 0.7)'])]
df_comp_temps = df_completo[df_completo['Modelo'].isin(['Fine-Tuned (Temp 0.7)', 'Fine-Tuned (Temp 0.5)'])]

sns.set_theme(style="whitegrid", palette="muted")

# --- GRÁFICA 1: BERTScore ---
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_comp_base_ft, x='Caso_Clinico', y='Metrica_BERTScore_F1', hue='Modelo', showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
plt.title('Estabilidad de la Precisión Semántica (BERTScore F1)', fontsize=14, fontweight='bold')
plt.ylabel('BERTScore F1', fontsize=12)
plt.xlabel('')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(ruta_descargas, "grafica_1_bertscore.png"), dpi=300) 
plt.close()

# --- GRÁFICA 2: Formato Correcto ---
plt.figure(figsize=(12, 6))
sns.barplot(data=df_comp_base_ft, x='Caso_Clinico', y='Metrica_Formato_Correcto', hue='Modelo', errorbar=None)
plt.title('Tasa de Éxito en la Adherencia al Formato Estructurado', fontsize=14, fontweight='bold')
plt.ylabel('Tasa de Éxito (0 a 1)', fontsize=12)
plt.xlabel('')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(ruta_descargas, "grafica_2_formato.png"), dpi=300)
plt.close()

# --- GRÁFICA 3: Alucinaciones ---
plt.figure(figsize=(12, 6))
sns.barplot(data=df_comp_base_ft, x='Caso_Clinico', y='Metrica_Num_Alt_Diagnosticos', hue='Modelo', errorbar='sd', capsize=0.1)
plt.title('Promedio de Alucinaciones (Diagnósticos Extra Propuestos) por Caso', fontsize=14, fontweight='bold')
plt.ylabel('Promedio de Diagnósticos Extra', fontsize=12)
plt.xlabel('')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(ruta_descargas, "grafica_3_alucinaciones_barras.png"), dpi=300)
plt.close()

# --- GRÁFICA 4: IMPACTO DE LA TEMPERATURA ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
medias_temp = df_comp_temps.groupby('Modelo')[['Metrica_Num_Alt_Diagnosticos', 'Metrica_Formato_Correcto']].mean().reset_index()

sns.barplot(data=medias_temp, x='Modelo', y='Metrica_Num_Alt_Diagnosticos', ax=axes[0], palette=['#ff9999', '#66b3ff'])
axes[0].set_title('Reducción de Alucinaciones por Temperatura', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Promedio Global de Diagnósticos Extra', fontsize=11)
axes[0].set_xlabel('')
for i in axes[0].containers: axes[0].bar_label(i, padding=3, fmt='%.3f')

sns.barplot(data=medias_temp, x='Modelo', y='Metrica_Formato_Correcto', ax=axes[1], palette=['#99ff99', '#ffcc99'])
axes[1].set_title('Estabilidad del Formato por Temperatura', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Tasa Global de Formato Correcto', fontsize=11)
axes[1].set_xlabel('')
for i in axes[1].containers: axes[1].bar_label(i, padding=3, fmt='%.3f')

plt.suptitle('Comparativa de Modelos Fine-Tuned: Impacto de la Temperatura (0.7 vs 0.5)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(ruta_descargas, "grafica_4_impacto_temperatura.png"), dpi=300)
plt.close()
