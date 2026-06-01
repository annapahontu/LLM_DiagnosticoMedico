# Análisis técnico de LLMs como herramientas de apoyo al diagnóstico médico

## Descripción del Proyecto
Este repositorio contiene el código y los recursos desarrollados para el trabajo de la asignatura **Programación Avanzada en Bioinformática** (Grado en Ingeniería de la Salud, Universidad de Málaga).

El objetivo principal es estudiar la viabilidad de utilizar Grandes Modelos de Lenguaje (LLMs), específicamente **Llama 3**, como herramientas de apoyo al diagnóstico médico. Para ello, realizamos una comparativa práctica entre:
1. El modelo base (Llama 3 en su estado original).
2. Un modelo ajustado (*fine-tuned*) con datos médicos, basándonos en la metodología del repositorio [Shekswess/LLM-Medical-Finetuning](https://github.com/Shekswess/LLM-Medical-Finetuning).

Buscamos analizar cómo este entrenamiento especializado mejora el procesamiento de la información clínica, evaluando su capacidad de acierto y la reducción de "alucinaciones" (generación de datos falsos), un factor de riesgo crítico en entornos de salud.

## Estructura del Repositorio
El código principal del proyecto se organiza dentro de la carpeta `src/`, la cual contiene los siguientes elementos:

* **`Pruebas_Llama3_Base.ipynb`**: Notebook de Google Colab utilizado para evaluar el comportamiento del modelo Llama 3 sin modificar. Aquí introducimos casos clínicos simulados para registrar cómo responde el modelo base por defecto.
* **`Llama3_Medical_Finetuning.ipynb`**: Notebook donde ejecutamos el ajuste fino (*fine-tuning*) del modelo. Empleamos técnicas eficientes como PEFT (LoRA/QLoRA), cuantización a 4 bits y la librería `unsloth` para que el entrenamiento sea viable en recursos limitados.
* **`Evaluacion_Metricas.ipynb`**: Notebook dedicado a la comparación estructurada de ambos modelos. Se analizan las diferencias en las respuestas, documentando las mejoras obtenidas y evaluando la mitigación de alucinaciones.
* **`graficas.py`**: Script de Python que toma los datos extraídos de la evaluación y genera las gráficas comparativas que ilustrarán el informe final del trabajo.

## Requisitos y Ejecución
Para reproducir el código de este proyecto, se requiere:
- Un entorno compatible con **Jupyter Notebooks** (se recomienda **Google Colab**).
- Aceleración por hardware (**GPU**) para poder cargar y entrenar los modelos sin problemas de memoria (OOM).
- Credenciales de *Hugging Face* para la descarga de los modelos de Llama 3.

> **Aviso Importante:** Los casos clínicos y los diagnósticos generados en esta demostración son exclusivamente pruebas conceptuales de software. Un LLM es una herramienta informática y **en ningún caso** sustituye el diagnóstico, consejo o tratamiento de un profesional médico cualificado.
