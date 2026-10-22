# Tablero Saber 11 · puntaje global por departamento

Tablero interactivo en Streamlit del proyecto **Predicción del puntaje global de la prueba Saber 11 a partir de variables socioeconómicas y del colegio**
(Maestría en Inteligencia Artificial, Universidad de La Salle, curso Ciencia de Datos). Autor: Andrés Giovanny Rubiano Muñoz.

Los cuadernos del proyecto (planteamiento y EDA, preparación y modelos de referencia, modelado y evaluación) están en
[RubianoAndy/saber11-project](https://github.com/RubianoAndy/saber11-project). Este repositorio contiene solo el tablero, que se despliega en Streamlit Community Cloud.

## Vistas

| Vista | Contenido |
| --- | --- |
| **Panorama** | Tarjetas KPI, puntaje promedio por área, distribución por nivel de desempeño, tabla por departamento, *treemap* naturaleza × jornada y mapa de Colombia. |
| **Brechas** | Brechas privado/oficial, con/sin internet y de género; puntaje por edad, educación de la madre, jornada y estrato. |
| **Predicción** | Formulario con el perfil del estudiante → puntaje global estimado (Gradient Boosting), rango probable, nivel de desempeño con probabilidades (SVM) y contribuciones SHAP. |
| **Segmentos** | Clústeres de K-means por nivel (K = 3) y por perfil relativo de áreas (K = 2), perfiles y clasificador de un estudiante nuevo. |
| **Modelos** | Comparación de los modelos de regresión y clasificación, importancia de variables y ficha de los datos. |

Los filtros de la barra lateral (departamento, naturaleza, jornada, género) afectan las vistas de panorama y brechas.

## Cómo funciona

El tablero **no depende de archivos binarios**: al arrancar descarga la muestra pública de Kaggle
([`andyrubiano/saber-11-muestra-2022-4`](https://www.kaggle.com/datasets/andyrubiano/saber-11-muestra-2022-4), 1.000 estudiantes del periodo 2022-4,
derivada de los datos abiertos del ICFES, CC BY-SA 4.0), aplica la misma preparación del cuaderno 02 y entrena los modelos finales del cuaderno 03 con los
hiperparámetros elegidos allí por validación cruzada. Todo queda en caché; el arranque en frío tarda unos 10 segundos. Las métricas que muestra la vista
*Modelos* se calculan en vivo sobre la misma partición de prueba de los cuadernos y coinciden con ellas (MAE 34,3, R² 0,286, F1 macro 0,498, AUC 0,680).

```
app.py                 interfaz (Streamlit + Plotly)
src/saber11.py         carga desde Kaggle, preparación, entrenamiento, predicción y segmentación
.streamlit/config.toml tema (azul institucional y amarillo)
assets/images/         logo y foto del autor
requirements.txt       dependencias con versiones exactas
```

## Ejecutar localmente

```bash
python -m venv .venv
source .venv/Scripts/activate      # en Windows con Git Bash; en Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue

En [share.streamlit.io](https://share.streamlit.io): *Create app* → repositorio `RubianoAndy/saber11-dashboard`, rama `main`, archivo principal `app.py`.
No requiere secretos: el dataset de Kaggle es público.

## Licencia de los datos

ICFES, *Resultados únicos Saber 11*, [datos.gov.co](https://www.datos.gov.co/Educaci-n/Resultados-nicos-Saber-11/kgxf-xxbe), licencia CC BY-SA 4.0.
