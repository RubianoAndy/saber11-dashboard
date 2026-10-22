# -*- coding: utf-8 -*-
"""Carga, preparación y modelos del tablero Saber 11.

Reproduce las decisiones de los cuadernos del proyecto saber11-project:
  - cuaderno 02: limpieza, variables derivadas, partición 70/15/15 y preprocesadores;
  - cuaderno 03: hiperparámetros elegidos por validación cruzada (Gradient Boosting, SVM, K-means).
El tablero entrena al arrancar con la muestra pública de Kaggle, de modo que no depende de archivos binarios.
"""
from __future__ import annotations

from dataclasses import dataclass

import kagglehub
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score,
                             root_mean_squared_error, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVC

DATASET = "andyrubiano/saber-11-muestra-2022-4"
ARCHIVO = "saber11_muestra_1000.csv"
RANDOM_STATE = 42
FECHA_PRUEBA = pd.Timestamp("2022-09-04")
CLASES = ["Bajo", "Medio", "Alto"]

AREA_COLS = ["punt_matematicas", "punt_lectura_critica", "punt_sociales_ciudadanas", "punt_c_naturales", "punt_ingles"]
AREA_NOMBRES = {"punt_matematicas": "Matemáticas", "punt_lectura_critica": "Lectura crítica", "punt_sociales_ciudadanas": "Sociales y ciudadanas",
                "punt_c_naturales": "Ciencias naturales", "punt_ingles": "Inglés"}
SCORE_COLS = AREA_COLS + ["punt_global"]

NUM_COLS = ["edad", "educ_madre_ord", "educ_padre_ord", "personas_por_cuarto"]
BIN_COLS = ["extraedad", "cuestionario_incompleto", "educ_madre_desconocida", "educ_padre_desconocida", "genero_F"]
NOM_COLS = ["fami_estratovivienda", "fami_personashogar", "fami_cuartoshogar", "fami_tieneinternet", "fami_tienecomputador", "fami_tieneautomovil",
            "fami_tienelavadora", "cole_naturaleza", "cole_area_ubicacion", "naturaleza_area", "cole_jornada", "cole_caracter", "cole_genero",
            "cole_sede_principal", "region"]
FEATURES = NUM_COLS + BIN_COLS + NOM_COLS

EDUCACION = {"Ninguno": 0, "Primaria incompleta": 1, "Primaria completa": 2, "Secundaria (Bachillerato) incompleta": 3,
             "Secundaria (Bachillerato) completa": 4, "Técnica o tecnológica incompleta": 5, "Técnica o tecnológica completa": 6,
             "Educación profesional incompleta": 7, "Educación profesional completa": 8, "Postgrado": 9}
PERSONAS = {"1 a 2": 1.5, "3 a 4": 3.5, "5 a 6": 5.5, "7 a 8": 7.5, "9 o más": 9.5}
CUARTOS = {"Uno": 1, "Dos": 2, "Tres": 3, "Cuatro": 4, "Cinco": 5, "Seis o mas": 6}
ESTRATO_AGRUPADO = {"Estrato 4": "Estrato 4 a 6", "Estrato 5": "Estrato 4 a 6", "Estrato 6": "Estrato 4 a 6"}

REGIONES = {"BOGOTÁ": "Bogotá",
            **{d: "Andina" for d in ["ANTIOQUIA", "BOYACA", "CALDAS", "CUNDINAMARCA", "HUILA", "NORTE SANTANDER", "QUINDIO", "RISARALDA", "SANTANDER", "TOLIMA"]},
            **{d: "Caribe" for d in ["ATLANTICO", "BOLIVAR", "CESAR", "CORDOBA", "LA GUAJIRA", "MAGDALENA", "SUCRE", "SAN ANDRES"]},
            **{d: "Pacífica" for d in ["CAUCA", "CHOCO", "NARIÑO", "VALLE"]},
            **{d: "Orinoquía" for d in ["ARAUCA", "CASANARE", "META", "VICHADA"]},
            **{d: "Amazonía" for d in ["AMAZONAS", "CAQUETA", "GUAINIA", "GUAVIARE", "PUTUMAYO", "VAUPES"]}}

# Coordenadas aproximadas de la capital de cada departamento, para el mapa.
COORDENADAS = {"AMAZONAS": (-4.20, -69.94), "ANTIOQUIA": (6.25, -75.56), "ARAUCA": (7.08, -70.76), "ATLANTICO": (10.96, -74.80), "BOGOTÁ": (4.71, -74.07),
               "BOLIVAR": (10.40, -75.51), "BOYACA": (5.54, -73.36), "CALDAS": (5.07, -75.52), "CAQUETA": (1.61, -75.61), "CASANARE": (5.34, -72.40),
               "CAUCA": (2.44, -76.61), "CESAR": (10.46, -73.25), "CHOCO": (5.69, -76.66), "CORDOBA": (8.75, -75.88), "CUNDINAMARCA": (4.82, -74.36),
               "GUAINIA": (3.87, -67.92), "GUAVIARE": (2.57, -72.64), "HUILA": (2.93, -75.28), "LA GUAJIRA": (11.54, -72.91), "MAGDALENA": (11.24, -74.20),
               "META": (4.14, -73.63), "NARIÑO": (1.21, -77.28), "NORTE SANTANDER": (7.89, -72.50), "PUTUMAYO": (1.15, -76.65), "QUINDIO": (4.53, -75.68),
               "RISARALDA": (4.81, -75.69), "SAN ANDRES": (12.58, -81.70), "SANTANDER": (7.12, -73.12), "SUCRE": (9.30, -75.40), "TOLIMA": (4.44, -75.24),
               "VALLE": (3.45, -76.53), "VAUPES": (1.25, -70.23), "VICHADA": (6.19, -67.49)}

# Hiperparámetros elegidos por validación cruzada en el cuaderno 03.
PARAMS_GB = dict(n_estimators=100, learning_rate=0.1, max_depth=2, min_samples_leaf=5, subsample=1.0, random_state=RANDOM_STATE)
PARAMS_SVM = dict(kernel="rbf", C=0.3, gamma="scale", probability=True, random_state=RANDOM_STATE)
K_NIVEL, K_PERFIL = 3, 2


# ----------------------------------------------------------------------------- datos
def cargar_muestra() -> pd.DataFrame:
    """Descarga la muestra pública desde Kaggle (1.000 estudiantes, periodo 2022-4)."""
    ruta = kagglehub.dataset_download(DATASET, path=ARCHIVO)
    df = pd.read_csv(ruta, dtype=str)
    for c in SCORE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df.drop_duplicates().set_index("estu_consecutivo")


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la limpieza y las variables derivadas del cuaderno 02 a la muestra completa."""
    d = df.copy()
    fami = [c for c in d.columns if c.startswith("fami_")]
    d["cuestionario_incompleto"] = d[fami].isna().any(axis=1).astype(int)
    for c in fami:
        d[c] = d[c].fillna("No respondió")
    d["cole_caracter"] = d["cole_caracter"].fillna("No reportado")
    nac = pd.to_datetime(d["estu_fechanacimiento"], format="%d/%m/%Y", errors="coerce")
    d["edad_original"] = ((FECHA_PRUEBA - nac).dt.days / 365.25).round(1)
    p99 = float(d["edad_original"].quantile(0.99))
    d["edad"] = d["edad_original"].clip(upper=p99)
    d["extraedad"] = (d["edad_original"] >= 19).astype(int)
    d = _derivadas(d)
    cortes = d["punt_global"].quantile([1 / 3, 2 / 3]).to_numpy()
    d["nivel_desempeno"] = pd.cut(d["punt_global"], bins=[-np.inf, cortes[0], cortes[1], np.inf], labels=CLASES).astype(str)
    d["region"] = d["cole_depto_ubicacion"].map(REGIONES)
    d.attrs["p99_edad"], d.attrs["cortes"] = p99, cortes
    return d


def _derivadas(d: pd.DataFrame) -> pd.DataFrame:
    for p, col in [("madre", "fami_educacionmadre"), ("padre", "fami_educacionpadre")]:
        o = d[col].map(EDUCACION)
        d[f"educ_{p}_ord"] = o.astype(float)
        d[f"educ_{p}_desconocida"] = o.isna().astype(int)
    d["personas_por_cuarto"] = (d["fami_personashogar"].map(PERSONAS) / d["fami_cuartoshogar"].map(CUARTOS)).astype(float)
    d["genero_F"] = (d["estu_genero"] == "F").astype(int)
    d["fami_estratovivienda"] = d["fami_estratovivienda"].replace(ESTRATO_AGRUPADO)
    d["naturaleza_area"] = d["cole_naturaleza"] + " · " + d["cole_area_ubicacion"]
    return d


def fila_estudiante(valores: dict, p99_edad: float) -> pd.DataFrame:
    """Construye la fila de variables explicativas de un estudiante a partir del formulario del tablero."""
    d = pd.DataFrame([valores])
    fami = [c for c in d.columns if c.startswith("fami_")]
    d["cuestionario_incompleto"] = int((d[fami] == "No respondió").any(axis=1).iloc[0])
    d["edad_original"] = float(valores["edad"])
    d["edad"] = min(float(valores["edad"]), p99_edad)
    d["extraedad"] = int(float(valores["edad"]) >= 19)
    d = _derivadas(d)
    d["region"] = d["cole_depto_ubicacion"].map(REGIONES)
    return d[FEATURES]


# ----------------------------------------------------------------------------- modelos
@dataclass
class Modelos:
    datos: pd.DataFrame
    particion: pd.Series
    prep_lineal: ColumnTransformer
    prep_arboles: ColumnTransformer
    gb: GradientBoostingRegressor
    svm: SVC
    kmeans_nivel: KMeans
    kmeans_perfil: KMeans
    scaler_nivel: RobustScaler
    scaler_perfil: StandardScaler
    pca_nivel: PCA
    pca_perfil: PCA
    metricas_reg: dict
    metricas_clf: dict
    siluetas: dict
    importancias: pd.Series


def entrenar(d: pd.DataFrame) -> Modelos:
    """Partición 70/15/15 estratificada, preprocesadores ajustados en entrenamiento y modelos finales del cuaderno 03."""
    pos = np.arange(len(d))
    est = d["nivel_desempeno"].to_numpy()
    p_tr, p_tmp = train_test_split(pos, test_size=0.30, random_state=RANDOM_STATE, stratify=est)
    p_va, p_te = train_test_split(p_tmp, test_size=0.50, random_state=RANDOM_STATE, stratify=est[p_tmp])
    particion = pd.Series("train", index=d.index, name="particion")
    particion.iloc[p_va] = "val"
    particion.iloc[p_te] = "test"

    X_tr, y_tr, c_tr = d.loc[particion == "train", FEATURES], d.loc[particion == "train", "punt_global"], d.loc[particion == "train", "nivel_desempeno"]
    X_te, y_te, c_te = d.loc[particion == "test", FEATURES], d.loc[particion == "test", "punt_global"], d.loc[particion == "test", "nivel_desempeno"]

    prep_lineal = ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM_COLS),
         ("bin", "passthrough", BIN_COLS),
         ("nom", OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False, min_frequency=5, dtype=np.float32), NOM_COLS)],
        verbose_feature_names_out=False).set_output(transform="pandas").fit(X_tr)
    prep_arboles = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), NUM_COLS), ("bin", "passthrough", BIN_COLS),
         ("nom", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32), NOM_COLS)],
        verbose_feature_names_out=False).set_output(transform="pandas").fit(X_tr)

    gb = GradientBoostingRegressor(**PARAMS_GB).fit(prep_arboles.transform(X_tr), y_tr)
    svm = SVC(**PARAMS_SVM).fit(prep_lineal.transform(X_tr), c_tr)

    pred = gb.predict(prep_arboles.transform(X_te))
    metricas_reg = {"MAE": mean_absolute_error(y_te, pred), "RMSE": root_mean_squared_error(y_te, pred), "R2": r2_score(y_te, pred),
                    "MAE_media": mean_absolute_error(y_te, np.full(len(y_te), y_tr.mean()))}
    Xl_te = prep_lineal.transform(X_te)
    metricas_clf = {"exactitud": accuracy_score(c_te, svm.predict(Xl_te)), "F1 macro": f1_score(c_te, svm.predict(Xl_te), average="macro"),
                    "AUC-ROC": roc_auc_score(c_te, svm.predict_proba(Xl_te), multi_class="ovr", average="macro", labels=sorted(CLASES))}

    # Segmentación sobre las 1.000 filas (no hay objetivo que filtrar).
    Xc = d[AREA_COLS + ["edad"]].astype(float)
    Xc = Xc.fillna(Xc.median())
    scaler_nivel = RobustScaler().fit(Xc)
    Z_nivel = scaler_nivel.transform(Xc)
    perfil = d[AREA_COLS].astype(float)
    perfil = perfil.fillna(perfil.median())
    rel = perfil.sub(perfil.mean(axis=1), axis=0)
    scaler_perfil = StandardScaler().fit(rel)
    Z_perfil = scaler_perfil.transform(rel)
    kmeans_nivel = KMeans(n_clusters=K_NIVEL, n_init=20, random_state=RANDOM_STATE).fit(Z_nivel)
    kmeans_perfil = KMeans(n_clusters=K_PERFIL, n_init=20, random_state=RANDOM_STATE).fit(Z_perfil)
    siluetas = {"nivel": silhouette_score(Z_nivel, kmeans_nivel.labels_), "perfil": silhouette_score(Z_perfil, kmeans_perfil.labels_)}
    pca_nivel = PCA(n_components=2, random_state=RANDOM_STATE).fit(Z_nivel)
    pca_perfil = PCA(n_components=2, random_state=RANDOM_STATE).fit(Z_perfil)

    datos = d.copy()
    datos["cluster_nivel"] = kmeans_nivel.labels_
    datos["cluster_perfil"] = kmeans_perfil.labels_
    datos[["pc1_nivel", "pc2_nivel"]] = pca_nivel.transform(Z_nivel)
    datos[["pc1_perfil", "pc2_perfil"]] = pca_perfil.transform(Z_perfil)
    importancias = pd.Series(gb.feature_importances_, index=prep_arboles.get_feature_names_out()).sort_values(ascending=False)

    return Modelos(datos, particion, prep_lineal, prep_arboles, gb, svm, kmeans_nivel, kmeans_perfil, scaler_nivel, scaler_perfil,
                   pca_nivel, pca_perfil, metricas_reg, metricas_clf, siluetas, importancias)


def predecir(m: Modelos, fila: pd.DataFrame) -> dict:
    """Puntaje global estimado, nivel de desempeño con probabilidades y contribuciones SHAP de un estudiante."""
    import shap

    Xa = m.prep_arboles.transform(fila)
    Xl = m.prep_lineal.transform(fila)
    puntaje = float(m.gb.predict(Xa)[0])
    proba = pd.Series(m.svm.predict_proba(Xl)[0], index=m.svm.classes_).reindex(CLASES)
    sv = shap.TreeExplainer(m.gb)(Xa)
    contrib = pd.Series(sv.values[0], index=Xa.columns)
    return {"puntaje": puntaje, "nivel": str(m.svm.predict(Xl)[0]), "proba": proba, "base": float(sv.base_values[0]), "contrib": contrib}


def segmento_de(m: Modelos, areas: dict, edad: float) -> dict:
    """Clúster de nivel y de perfil relativo de un estudiante dado sus cinco puntajes por área y su edad."""
    x = pd.DataFrame([{**{a: float(areas[a]) for a in AREA_COLS}, "edad": min(edad, float(m.datos["edad"].max()))}])
    nivel = int(m.kmeans_nivel.predict(m.scaler_nivel.transform(x[AREA_COLS + ["edad"]]))[0])
    rel = x[AREA_COLS].sub(x[AREA_COLS].mean(axis=1), axis=0)
    perfil = int(m.kmeans_perfil.predict(m.scaler_perfil.transform(rel))[0])
    return {"nivel": nivel, "perfil": perfil}
