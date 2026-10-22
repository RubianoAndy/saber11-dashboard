<div align="center">
    <img src="assets/images/Logo.png" width="250" alt="Logo Universidad de La Salle">
</div>

# Tablero interactivo · Predicción del puntaje global de la prueba Saber 11

## 📋 Información General

<div align="center">
    <img src="assets/images/author/Andy Rubiano.png" width="200" alt="Foto de Andrés Giovanny Rubiano Muñoz" style="border-radius: 10px;">
</div>

| Aspecto | Detalles |
|--------|----------|
| **Autor** | Andrés Giovanny Rubiano Muñoz "Andy Rubiano" |
| **Correo** | arubiano67@unisalle.edu.co |
| **Asignatura** | Ciencia de Datos |
| **Actividad** | Proyecto Integrador de Ciencia de Datos · Fase 4 (comunicación y despliegue) |
| **Programa** | Maestría en Inteligencia Artificial |
| **Universidad** | Universidad de La Salle |
| **Líneas de Trabajo** | Visualización de Datos, Despliegue de Modelos, Interpretabilidad |
| **Año** | 2026 |
| **Estado** | Tablero desplegado en [saber11-dashboard.streamlit.app](https://saber11-dashboard.streamlit.app/) · Sustentación pendiente |

---

## 🎯 Descripción del Proyecto

Este repositorio contiene el **tablero interactivo** del proyecto *Predicción del puntaje global de la prueba Saber 11 a partir de variables socioeconómicas y del colegio*, cuyos cuadernos de análisis, preparación y modelado viven en [RubianoAndy/saber11-project](https://github.com/RubianoAndy/saber11-project). El tablero es la pieza de comunicación de la fase 4: pone en manos de una secretaría de educación, un rector o un investigador los hallazgos de las tres fases anteriores, con filtros, indicadores, predicción en vivo y segmentación de estudiantes.

Se separa del repositorio principal por una razón práctica: el despliegue en Streamlit Community Cloud queda **liviano** (no arrastra los cuadernos ni PyTorch) y el tablero es **autónomo**: al arrancar descarga la muestra pública de Kaggle, aplica la misma preparación del cuaderno 02 y entrena los modelos finales del cuaderno 03 con los hiperparámetros elegidos allí por validación cruzada. No depende de ningún archivo binario y las métricas que muestra se calculan en vivo sobre la misma partición de prueba de los cuadernos.

### Objetivos Principales

- Comunicar las brechas del puntaje global (naturaleza y jornada del colegio, acceso a internet, educación de los padres, edad, geografía) con filtros por departamento, naturaleza, jornada y género.
- Ofrecer una **predicción en vivo** del puntaje global y del nivel de desempeño de un estudiante a partir de su contexto, con una explicación de qué suma y qué resta (valores SHAP) y un rango de error honesto.
- Mostrar los **segmentos de estudiantes** hallados por K-means y permitir ubicar a un estudiante nuevo en ellos.
- Presentar la comparación de modelos y la ficha de los datos para la sustentación.

---

## 📚 Estructura del Repositorio

```
.
├── README.md                     # Este archivo
├── app.py                        # Interfaz del tablero (Streamlit + Plotly): cinco vistas y filtros
├── requirements.txt              # Dependencias con versiones exactas (las que usa Streamlit Community Cloud)
├── .gitignore                    # Excluye .venv/, __pycache__/ y secretos de Streamlit
├── .streamlit/
│   └── config.toml               # Tema: azul institucional (#0B2545) y amarillo (#FFC000)
├── assets/
│   └── images/
│       ├── Logo.png                  # Logo institucional (fondo claro)
│       ├── UnisalleDarkLogoV1.png    # Logo en blanco para la barra lateral azul
│       └── author/                   # Fotografía del autor
└── src/
    └── saber11.py                # Carga desde Kaggle, preparación (cuaderno 02), modelos (cuaderno 03), predicción y segmentación
```

---

## 🧪 Metodología

### Vistas del tablero

| Vista | Contenido |
|---|---|
| **Panorama** | Tarjetas KPI (estudiantes, puntaje promedio, mínimo y máximo), puntaje promedio por área, distribución por nivel de desempeño, tabla resumen por departamento, *treemap* naturaleza × jornada coloreado por puntaje y mapa de Colombia con el promedio por departamento. |
| **Brechas** | Brechas privado − oficial, con − sin internet y hombres − mujeres, extraedad; puntaje por naturaleza × área, por edad, por educación de la madre, por jornada y por estrato × internet. |
| **Predicción** | Formulario con las variables del estudiante, el hogar y el colegio → puntaje global estimado por **Gradient Boosting**, rango probable (± MAE), percentil en la muestra, nivel de desempeño con probabilidades por **SVM** y gráfico de contribuciones **SHAP** de esa predicción. |
| **Segmentos** | Clústeres de **K-means** por nivel (puntajes por área + edad, K = 3) y por perfil relativo (áreas menos el promedio del estudiante, K = 2) en componentes principales, perfil de cada segmento y clasificador de un estudiante nuevo con su radar de fortalezas. |
| **Modelos** | Métricas en vivo del Gradient Boosting y la SVM, tablas comparativas de los seis regresores y los cuatro clasificadores del cuaderno 03, importancia de variables y ficha de los datos. |

Los filtros de la barra lateral (departamento, naturaleza, jornada y género) afectan las vistas de panorama y brechas; la predicción y los segmentos usan siempre los modelos entrenados con toda la muestra.

### Datos y modelos

| Aspecto | Valor |
|---|---|
| Fuente original | ICFES · [datos.gov.co](https://www.datos.gov.co/Educaci-n/Resultados-nicos-Saber-11/kgxf-xxbe), conjunto `kgxf-xxbe`, periodo 2022-4, CC BY-SA 4.0 |
| Muestra utilizada | Kaggle · [`andyrubiano/saber-11-muestra-2022-4`](https://www.kaggle.com/datasets/andyrubiano/saber-11-muestra-2022-4) · 1.000 estudiantes por muestreo estratificado (naturaleza × área × tercil de puntaje, semilla 42) de los 532.792 evaluados |
| Preparación (cuaderno 02) | Faltantes del cuestionario como categoría *No respondió* más indicador, edad winsorizada en el percentil 99, extraedad, educación de los padres en escala ordinal 0 a 9, personas por cuarto, estrato agrupado, seis regiones, interacción naturaleza × área |
| Partición | 70 / 15 / 15 estratificada por nivel de desempeño (700 / 150 / 150), semilla 42; preprocesadores ajustados solo en entrenamiento |
| Regresión | `GradientBoostingRegressor` (100 árboles, profundidad 2, tasa 0,1, hojas ≥ 5): mejor modelo por validación cruzada en el cuaderno 03 |
| Clasificación | `SVC` con núcleo RBF (`C` = 0,3), `probability=True` para las probabilidades y el AUC |
| Segmentación | `KMeans` (K = 3 sobre puntajes por área y edad con escalado robusto; K = 2 sobre el perfil relativo) |
| Explicación | `shap.TreeExplainer` sobre el Gradient Boosting (aditividad exacta) |
| Sin fuga de información | Los puntajes por área no se usan para predecir el global (lo determinan por fórmula); solo entran en la segmentación |

---

## 📊 Resultados

Las métricas que el tablero calcula al arrancar coinciden con las del cuaderno 03 porque usa la misma muestra, la misma partición y los mismos hiperparámetros:

| Modelo | Métrica en prueba (150 estudiantes) |
|---|---|
| Gradient Boosting (regresión) | **MAE 34,3 · RMSE 43,3 · R² 0,286** · 18,7 % mejor que predecir la media (42,2) |
| SVM RBF (nivel Bajo / Medio / Alto) | **F1 macro 0,498 · exactitud 0,500 · AUC-ROC 0,680** · la clase *Medio* es la difícil (F1 0,38) |
| K-means por nivel | 3 segmentos: alto desempeño (423, media 298), bajo a medio (557, media 214) y adultos de jornada nocturna y sabatina (20, 39 años, media 190); silueta 0,38 |
| K-means por perfil | 2 perfiles de 500: fuerte en lectura crítica y débil en inglés vs. fuerte en inglés y débil en sociales; silueta 0,19 |
| Variables más importantes | Edad, educación de la madre y del padre, región, internet en el hogar y jornada |

> ⚠️ **Lectura honesta.** El contexto socioeconómico y escolar explica cerca del 30 % de la varianza del puntaje; el error típico de ±34 puntos se muestra siempre junto a la predicción. El tablero describe desigualdades estructurales y no debe usarse para decidir sobre estudiantes individuales.

---

## ⚙️ Requisitos

> ℹ️ **No requiere credenciales.** El dataset de Kaggle es público y se descarga con `kagglehub` sin cuenta. El arranque en frío (descarga y entrenamiento) tarda unos 10 segundos y queda en caché.

Para ejecutarlo localmente se usa un entorno virtual con Python 3.12. Desde Git Bash, en la raíz del repositorio:

```bash
py -3.12 -m venv .venv
```

```bash
source .venv/Scripts/activate
```

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

| Etapa | Dependencias |
|---|---|
| Interfaz | `streamlit` |
| Descarga de datos | `kagglehub` |
| Carga y preparación | `pandas`, `numpy`, `pyarrow` |
| Modelos y explicación | `scikit-learn`, `shap` |
| Gráficos | `plotly` |

### Despliegue en Streamlit Community Cloud

1. Entrar a [share.streamlit.io](https://share.streamlit.io) con la cuenta de GitHub.
2. *Create app* → *Deploy a public app from GitHub*.
3. Repositorio `RubianoAndy/saber11-dashboard`, rama `main`, archivo principal `app.py`.
4. *Deploy*. Cada `git push` posterior redespliega la aplicación. La app está publicada en [saber11-dashboard.streamlit.app](https://saber11-dashboard.streamlit.app/).

---

## 📋 Estado del Proyecto

### ✅ Tablero · [saber11-dashboard.streamlit.app](https://saber11-dashboard.streamlit.app/)

- ✅ **Cinco vistas** (panorama, brechas, predicción, segmentos, modelos) con filtros globales e identidad gráfica institucional
- ✅ **Predicción en vivo** con rango de error, nivel de desempeño y explicación SHAP por estudiante
- ✅ **Segmentación** por nivel y por perfil relativo, con ubicación de un estudiante nuevo
- ✅ **Autonomía**: datos desde Kaggle y modelos entrenados al arrancar; métricas verificadas contra el cuaderno 03
- ✅ Probado localmente con Python 3.12 y las versiones fijadas en `requirements.txt`

### ⏳ Pendiente

- ⏳ Sustentación (28 nov. 2026)

---

## 🔑 Palabras Clave

`Streamlit` · `Tablero` · `Saber 11` · `ICFES` · `Gradient Boosting` · `SVM` · `K-means` · `SHAP` · `Plotly` · `Despliegue de Modelos` · `Datos Abiertos` · `scikit-learn`

---

## 🔗 Recursos

- [Aplicación publicada](https://saber11-dashboard.streamlit.app/)
- [Repositorio principal del proyecto (cuadernos)](https://github.com/RubianoAndy/saber11-project)
- [Dataset — Resultados únicos Saber 11 (datos.gov.co)](https://www.datos.gov.co/Educaci-n/Resultados-nicos-Saber-11/kgxf-xxbe)
- [Muestra utilizada — Saber 11 Colombia - Muestra 2022-4 (Kaggle)](https://www.kaggle.com/datasets/andyrubiano/saber-11-muestra-2022-4)
- [Streamlit — Documentación](https://docs.streamlit.io/)
- [Streamlit Community Cloud](https://share.streamlit.io)
- [Plotly Express — API reference](https://plotly.com/python-api-reference/plotly.express.html)
- [SHAP — Documentación](https://shap.readthedocs.io/)
- [kagglehub](https://github.com/Kaggle/kagglehub)
- [ICFES — Prueba Saber 11](https://www.icfes.gov.co/evaluaciones-icfes/saber-11/)

---

## 📧 Contacto

**Andrés Giovanny Rubiano Muñoz**
Maestría en Inteligencia Artificial · Universidad de La Salle
arubiano67@unisalle.edu.co

---

## 📄 Derechos Reservados

© 2026 Andrés Giovanny Rubiano Muñoz (Andy Rubiano). Todos los derechos reservados.

Este trabajo académico y su contenido —investigación, código, metodologías y documentación— son propiedad intelectual conjunta de:

- **Andrés Giovanny Rubiano Muñoz** (Andy Rubiano) — Autor
- **Universidad de La Salle** — Institución académica

Los datos utilizados pertenecen al ICFES y se distribuyen bajo licencia CC BY-SA 4.0 a través del Portal Nacional de Datos Abiertos de Colombia. El uso, reproducción o distribución del resto del trabajo requiere autorización previa escrita de los titulares de derechos.

---

<div align="center">
  Universidad de La Salle | Bogotá D. C., Colombia
</div>
