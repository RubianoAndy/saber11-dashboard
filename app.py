# -*- coding: utf-8 -*-
"""Tablero Saber 11 · Universidad de La Salle · Andrés Giovanny Rubiano Muñoz.

Streamlit con cinco vistas: panorama, brechas, predicción en vivo, segmentos y modelos.
Los datos se descargan de Kaggle y los modelos se entrenan al arrancar (ver src/saber11.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import saber11 as s11

st.set_page_config(page_title="Saber 11 · Tablero", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

NAVY, YELLOW, BLUE, ORANGE, GRIS = "#0B2545", "#FFC000", "#2A78D6", "#EB6834", "#52514E"
PALETA = [BLUE, "#1F3A93", ORANGE, "#9467BD", "#E377C2", "#8C564B", YELLOW, "#2CA02C"]
ROOT = Path(__file__).parent

# ----------------------------------------------------------------------------- estilo (sidebar azul, botón activo amarillo, tarjetas blancas)
st.markdown(f"""
<style>
[data-testid="stSidebar"] {{ background: {NAVY}; }}
[data-testid="stSidebar"] > div:first-child {{ min-height: 100vh; }}
[data-testid="stSidebar"] * {{ color: #FFFFFF; }}
[data-testid="stSidebar"] hr {{ border-color: {YELLOW}; opacity: .8; }}
[data-testid="stSidebar"] .stButton {{ width: 100%; }}
[data-testid="stSidebar"] .stButton > button {{ width: 100%; display: block; background: {NAVY}; color: #FFF; border: 1px solid #2a4f86; font-weight: 700; border-radius: 0; }}
[data-testid="stSidebar"] .stButton > button:hover {{ border-color: {YELLOW}; color: {YELLOW}; }}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {{ background: {YELLOW}; color: {NAVY}; border-color: {YELLOW}; }}
[data-testid="stSidebar"] .stButton > button[kind="primary"] * {{ color: {NAVY}; }}
[data-testid="stSidebar"] [data-baseweb="select"] > div, [data-testid="stSidebar"] [data-baseweb="tag"] {{ background: #0f3366; border-color: #2a4f86; border-radius: 0; }}
[data-testid="stSidebar"] [data-baseweb="select"] svg {{ fill: #FFF; }}
[data-testid="stSidebar"] [data-testid="stImage"] img {{ border-radius: 0; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; }}
h1.titulo {{ color: {NAVY}; font-size: 2rem; font-weight: 800; letter-spacing: .5px; margin: 0; text-transform: uppercase; }}
p.autor {{ color: {GRIS}; font-weight: 700; margin: 0 0 .8rem 0; }}
.kpi {{ background: #FFF; border-radius: 0; padding: 14px 18px; box-shadow: 0 1px 4px rgba(0,0,0,.12); display: flex; align-items: center; gap: 14px; height: 92px; }}
.kpi .icono {{ font-size: 2rem; }}
.kpi .valor {{ font-size: 1.55rem; font-weight: 800; color: #111; line-height: 1.1; }}
.kpi .etiqueta {{ font-size: .95rem; font-weight: 700; color: {NAVY}; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background: #FFF; border-radius: 0; box-shadow: 0 1px 4px rgba(0,0,0,.12); }}
[data-testid="stVerticalBlockBorderWrapper"] > div {{ border-radius: 0; }}
h3.panel {{ color: {NAVY}; font-size: 1rem; font-weight: 800; margin: 0 0 4px 0; }}
.resultado {{ background: #FFF; border-left: 8px solid {YELLOW}; border-radius: 0; padding: 16px 20px; box-shadow: 0 1px 4px rgba(0,0,0,.12); }}
.resultado .grande {{ font-size: 2.6rem; font-weight: 800; color: {NAVY}; line-height: 1; }}
[data-testid="stDataFrame"], [data-baseweb="select"] > div, [data-baseweb="input"] > div, .stButton > button {{ border-radius: 0 !important; }}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#222", size=12),
                     legend=dict(title=None), xaxis=dict(gridcolor="#EEE"), yaxis=dict(gridcolor="#EEE"))


def panel(titulo):
    st.markdown(f'<h3 class="panel">{titulo}</h3>', unsafe_allow_html=True)


def kpi(col, icono, valor, etiqueta):
    col.markdown(f'<div class="kpi"><div class="icono">{icono}</div><div><div class="valor">{valor}</div><div class="etiqueta">{etiqueta}</div></div></div>',
                 unsafe_allow_html=True)


def fig(f, alto=320):
    f.update_layout(height=alto, **PLOTLY_LAYOUT)
    st.plotly_chart(f, width="stretch", config={"displayModeBar": False})


def fmt(x, dec=1):
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ----------------------------------------------------------------------------- datos y modelos (una sola vez por servidor)
@st.cache_data(show_spinner="Descargando la muestra desde Kaggle ...")
def datos():
    return s11.preparar(s11.cargar_muestra())


@st.cache_resource(show_spinner="Entrenando los modelos (Gradient Boosting, SVM y K-means) ...")
def modelos():
    return s11.entrenar(datos())


df = datos()
m = modelos()
D = m.datos

# ----------------------------------------------------------------------------- barra lateral: navegación y filtros
VISTAS = ["Panorama - vista 1", "Brechas - vista 2", "Predicción - vista 3", "Segmentos", "Modelos"]
if "vista" not in st.session_state:
    st.session_state.vista = VISTAS[0]

with st.sidebar:
    _, logo, _ = st.columns([1, 5, 1])
    logo.image(str(ROOT / "assets" / "images" / "UnisalleDarkLogoV1.png"), width="stretch")
    st.write("")
    for v in VISTAS:
        if st.button(v, type="primary" if st.session_state.vista == v else "secondary", key=f"nav_{v}", width="stretch"):
            st.session_state.vista = v
            st.rerun()
    st.markdown("---")
    st.markdown("**Departamento**")
    deptos = sorted(D["cole_depto_ubicacion"].unique())
    sel_deptos = st.multiselect("Departamento", deptos, default=[], placeholder="Todos", label_visibility="collapsed")
    st.markdown("**Naturaleza del colegio**")
    sel_nat = st.multiselect("Naturaleza", ["OFICIAL", "NO OFICIAL"], default=[], placeholder="Todas", label_visibility="collapsed")
    st.markdown("**Jornada**")
    sel_jor = st.multiselect("Jornada", sorted(D["cole_jornada"].unique()), default=[], placeholder="Todas", label_visibility="collapsed")
    st.markdown("**Género del estudiante**")
    sel_gen = st.radio("Género", ["Todos", "Femenino", "Masculino"], label_visibility="collapsed")
    st.markdown("---")
    c1, c2 = st.columns([1, 2])
    c1.image(str(ROOT / "assets" / "images" / "author" / "Andy Rubiano.png"), width="stretch")
    c2.markdown("**Andy Rubiano**  \nMaestría en IA  \nCiencia de Datos")

F = D.copy()
if sel_deptos:
    F = F[F["cole_depto_ubicacion"].isin(sel_deptos)]
if sel_nat:
    F = F[F["cole_naturaleza"].isin(sel_nat)]
if sel_jor:
    F = F[F["cole_jornada"].isin(sel_jor)]
if sel_gen != "Todos":
    F = F[F["estu_genero"] == ("F" if sel_gen == "Femenino" else "M")]

st.markdown('<h1 class="titulo">Puntaje global Saber 11 por departamento</h1><p class="autor">Por Andy Rubiano · ICFES 2022-4 · muestra representativa de 1.000 estudiantes</p>',
            unsafe_allow_html=True)
if F.empty:
    st.warning("Ningún estudiante cumple los filtros seleccionados.")
    st.stop()

# ============================================================================= vista 1 · panorama
if st.session_state.vista == VISTAS[0]:
    k = st.columns(4)
    kpi(k[0], "🎓", fmt(len(F), 0), "Estudiantes")
    kpi(k[1], "📊", fmt(F["punt_global"].mean()), "Puntaje global promedio")
    kpi(k[2], "📉", fmt(F["punt_global"].min(), 0), "Puntaje mínimo")
    kpi(k[3], "📈", fmt(F["punt_global"].max(), 0), "Puntaje máximo")
    st.write("")
    a, b = st.columns([1.3, 1])
    with a, st.container(border=True):
        panel("Puntaje promedio por área")
        areas = F[s11.AREA_COLS].mean().rename(s11.AREA_NOMBRES).sort_values(ascending=False)
        f = px.bar(x=areas.index, y=areas.values, text=[fmt(v) for v in areas.values], color_discrete_sequence=[YELLOW])
        f.update_traces(textposition="outside")
        f.update_layout(xaxis_title=None, yaxis_title="Puntaje (0 a 100)", yaxis_range=[0, 75])
        fig(f)
    with b, st.container(border=True):
        panel("Porcentaje de estudiantes por nivel de desempeño")
        niv = F["nivel_desempeno"].value_counts().reindex(s11.CLASES).fillna(0)
        f = px.pie(names=niv.index, values=niv.values, hole=0, color=niv.index, color_discrete_map={"Bajo": ORANGE, "Medio": "#1F3A93", "Alto": BLUE})
        f.update_traces(textinfo="percent", textfont_size=13, sort=False)
        f.update_layout(legend=dict(title="Nivel", orientation="v"))
        fig(f)
    a, b, c = st.columns([1.35, 1, 1])
    with a, st.container(border=True):
        panel("Resumen por departamento")
        g = F.groupby("cole_depto_ubicacion").agg(Estudiantes=("punt_global", "size"), Promedio=("punt_global", "mean"),
                                                    Alto=("nivel_desempeno", lambda s: (s == "Alto").mean() * 100)).sort_values("Promedio", ascending=False)
        g.index.name = "Departamento"
        tot = pd.DataFrame({"Estudiantes": [len(F)], "Promedio": [F["punt_global"].mean()], "Alto": [(F["nivel_desempeno"] == "Alto").mean() * 100]}, index=["Total"])
        st.dataframe(pd.concat([g, tot]).rename(columns={"Alto": "% nivel Alto"}).style.format({"Promedio": "{:.1f}", "% nivel Alto": "{:.1f} %"}),
                     height=330, width="stretch")
    with b, st.container(border=True):
        panel("Estudiantes por naturaleza y jornada")
        t = F.groupby(["cole_naturaleza", "cole_jornada"]).agg(n=("punt_global", "size"), promedio=("punt_global", "mean")).reset_index()
        f = px.treemap(t, path=["cole_naturaleza", "cole_jornada"], values="n", color="promedio", color_continuous_scale=["#EB6834", "#F3F4F6", BLUE],
                       color_continuous_midpoint=D["punt_global"].mean(), hover_data={"promedio": ":.1f"})
        f.update_traces(textinfo="label+value", marker_line_width=1.5, marker_line_color="white")
        f.update_layout(coloraxis_colorbar=dict(title="Promedio", thickness=10, len=.8))
        fig(f, 340)
    with c, st.container(border=True):
        panel("Puntaje promedio en Colombia")
        geo = F.groupby("cole_depto_ubicacion").agg(n=("punt_global", "size"), promedio=("punt_global", "mean")).reset_index()
        geo["lat"] = geo["cole_depto_ubicacion"].map(lambda d: s11.COORDENADAS[d][0])
        geo["lon"] = geo["cole_depto_ubicacion"].map(lambda d: s11.COORDENADAS[d][1])
        f = px.scatter_map(geo, lat="lat", lon="lon", size="n", color="promedio", hover_name="cole_depto_ubicacion", size_max=38, zoom=3.9,
                           center={"lat": 5.4, "lon": -73.6}, color_continuous_scale=["#EB6834", "#F3F4F6", BLUE], color_continuous_midpoint=D["punt_global"].mean(),
                           hover_data={"lat": False, "lon": False, "n": True, "promedio": ":.1f"}, map_style="carto-positron")
        f.update_layout(coloraxis_colorbar=dict(title="Promedio", thickness=10, len=.8))
        fig(f, 340)

# ============================================================================= vista 2 · brechas
elif st.session_state.vista == VISTAS[1]:
    k = st.columns(4)
    g_nat = F.groupby("cole_naturaleza")["punt_global"].mean()
    brecha_nat = g_nat.get("NO OFICIAL", np.nan) - g_nat.get("OFICIAL", np.nan)
    g_int = F.groupby("fami_tieneinternet")["punt_global"].mean()
    g_gen = F.groupby("estu_genero")["punt_global"].mean()
    kpi(k[0], "🏫", (fmt(brecha_nat) if pd.notna(brecha_nat) else "—") + " pts", "Brecha privado − oficial")
    kpi(k[1], "🌐", (fmt(g_int.get("Si", np.nan) - g_int.get("No", np.nan)) if {"Si", "No"} <= set(g_int.index) else "—") + " pts", "Brecha con internet − sin internet")
    kpi(k[2], "⚧", (fmt(g_gen.get("M", np.nan) - g_gen.get("F", np.nan)) if {"M", "F"} <= set(g_gen.index) else "—") + " pts", "Brecha hombres − mujeres")
    kpi(k[3], "🎂", fmt((F["extraedad"] == 1).mean() * 100) + " %", "Estudiantes en extraedad (≥ 19 años)")
    st.write("")
    a, b = st.columns(2)
    with a, st.container(border=True):
        panel("Puntaje promedio por naturaleza y área del colegio")
        g = F.groupby(["cole_naturaleza", "cole_area_ubicacion"]).agg(promedio=("punt_global", "mean"), n=("punt_global", "size")).reset_index()
        f = px.bar(g, x="cole_naturaleza", y="promedio", color="cole_area_ubicacion", barmode="group", text=g["promedio"].map(fmt),
                   color_discrete_sequence=[BLUE, YELLOW], hover_data={"n": True})
        f.update_traces(textposition="outside")
        f.update_layout(xaxis_title=None, yaxis_title="Puntaje global", yaxis_range=[0, 330], legend_title="Área")
        fig(f)
    with b, st.container(border=True):
        panel("Puntaje promedio por edad (la caída después de los 18 años)")
        e = F.assign(edad_grupo=F["edad_original"].clip(upper=30).round()).groupby("edad_grupo").agg(promedio=("punt_global", "mean"), n=("punt_global", "size")).reset_index()
        e = e[e["n"] >= 3]
        f = px.line(e, x="edad_grupo", y="promedio", markers=True, color_discrete_sequence=[BLUE], hover_data={"n": True})
        f.update_layout(xaxis_title="Edad en años (30 agrupa a los mayores)", yaxis_title="Puntaje global")
        fig(f)
    a, b, c = st.columns(3)
    with a, st.container(border=True):
        panel("Educación de la madre")
        orden = list(s11.EDUCACION) + ["No sabe", "No Aplica", "No respondió"]
        g = F.groupby("fami_educacionmadre")["punt_global"].agg(["mean", "size"]).reindex([o for o in orden if o in F["fami_educacionmadre"].unique()])
        f = px.bar(y=g.index, x=g["mean"], orientation="h", color_discrete_sequence=[BLUE], text=g["mean"].map(fmt), hover_data={"n": g["size"]})
        f.update_traces(textposition="outside")
        f.update_layout(yaxis_title=None, xaxis_title="Puntaje global", xaxis_range=[0, 330], yaxis=dict(autorange="reversed"))
        fig(f, 380)
    with b, st.container(border=True):
        panel("Jornada del colegio")
        g = F.groupby("cole_jornada")["punt_global"].agg(["mean", "size"]).sort_values("mean", ascending=False)
        f = px.bar(x=g.index, y=g["mean"], color_discrete_sequence=[YELLOW], text=g["mean"].map(fmt), hover_data={"n": g["size"]})
        f.update_traces(textposition="outside")
        f.update_layout(xaxis_title=None, yaxis_title="Puntaje global", yaxis_range=[0, 330])
        fig(f, 380)
    with c, st.container(border=True):
        panel("Estrato de la vivienda y acceso a internet")
        orden = ["Sin Estrato", "Estrato 1", "Estrato 2", "Estrato 3", "Estrato 4 a 6", "No respondió"]
        g = F.groupby(["fami_estratovivienda", "fami_tieneinternet"])["punt_global"].mean().reset_index()
        g = g[g["fami_tieneinternet"].isin(["Si", "No"])]
        f = px.bar(g, x="fami_estratovivienda", y="punt_global", color="fami_tieneinternet", barmode="group", category_orders={"fami_estratovivienda": orden},
                   color_discrete_map={"Si": BLUE, "No": ORANGE})
        f.update_layout(xaxis_title=None, yaxis_title="Puntaje global", yaxis_range=[0, 330], legend_title="Internet")
        fig(f, 380)

# ============================================================================= vista 3 · predicción en vivo
elif st.session_state.vista == VISTAS[2]:
    st.markdown("Completa el perfil del estudiante y el modelo estima su puntaje global (Gradient Boosting) y su nivel de desempeño (SVM). "
                "Los modelos usan solo variables conocidas **antes** de la prueba; el error típico del puntaje es de ±" + fmt(m.metricas_reg["MAE"], 0) + " puntos.")
    izq, der = st.columns([1.15, 1])
    with izq, st.container(border=True):
        panel("Perfil del estudiante")
        c1, c2, c3 = st.columns(3)
        edad = c1.number_input("Edad (años)", 14.0, 80.0, 17.0, 0.5)
        genero = c2.selectbox("Género", ["F", "M"], format_func=lambda g: "Femenino" if g == "F" else "Masculino")
        depto = c3.selectbox("Departamento del colegio", sorted(D["cole_depto_ubicacion"].unique()), index=sorted(D["cole_depto_ubicacion"].unique()).index("BOGOTÁ"))
        c1, c2, c3 = st.columns(3)
        naturaleza = c1.selectbox("Naturaleza del colegio", ["OFICIAL", "NO OFICIAL"])
        area = c2.selectbox("Área del colegio", ["URBANO", "RURAL"])
        jornada = c3.selectbox("Jornada", ["MAÑANA", "TARDE", "COMPLETA", "UNICA", "NOCHE", "SABATINA"])
        c1, c2, c3 = st.columns(3)
        caracter = c1.selectbox("Carácter", ["ACADÉMICO", "TÉCNICO/ACADÉMICO", "TÉCNICO", "NO APLICA", "No reportado"])
        cole_genero = c2.selectbox("Colegio", ["MIXTO", "FEMENINO", "MASCULINO"])
        sede = c3.selectbox("Sede principal", ["S", "N"], format_func=lambda x: "Sí" if x == "S" else "No")
        st.markdown("**Hogar**")
        c1, c2, c3 = st.columns(3)
        estrato = c1.selectbox("Estrato", ["Estrato 1", "Estrato 2", "Estrato 3", "Estrato 4", "Estrato 5", "Estrato 6", "Sin Estrato", "No respondió"], index=1)
        personas = c2.selectbox("Personas en el hogar", list(s11.PERSONAS) + ["No respondió"], index=1)
        cuartos = c3.selectbox("Cuartos para dormir", list(s11.CUARTOS) + ["No respondió"], index=2)
        c1, c2 = st.columns(2)
        educ_m = c1.selectbox("Educación de la madre", list(s11.EDUCACION) + ["No sabe", "No respondió"], index=4)
        educ_p = c2.selectbox("Educación del padre", list(s11.EDUCACION) + ["No sabe", "No respondió"], index=4)
        c1, c2, c3, c4 = st.columns(4)
        internet = c1.selectbox("Internet", ["Si", "No"])
        computador = c2.selectbox("Computador", ["Si", "No"])
        automovil = c3.selectbox("Automóvil", ["Si", "No"], index=1)
        lavadora = c4.selectbox("Lavadora", ["Si", "No"])
    valores = {"edad": edad, "estu_genero": genero, "cole_depto_ubicacion": depto, "cole_naturaleza": naturaleza, "cole_area_ubicacion": area,
               "cole_jornada": jornada, "cole_caracter": caracter, "cole_genero": cole_genero, "cole_sede_principal": sede,
               "fami_estratovivienda": estrato, "fami_personashogar": personas, "fami_cuartoshogar": cuartos, "fami_educacionmadre": educ_m,
               "fami_educacionpadre": educ_p, "fami_tieneinternet": internet, "fami_tienecomputador": computador, "fami_tieneautomovil": automovil,
               "fami_tienelavadora": lavadora}
    r = s11.predecir(m, s11.fila_estudiante(valores, df.attrs["p99_edad"]))
    with der:
        mae = m.metricas_reg["MAE"]
        pct = (D["punt_global"] < r["puntaje"]).mean() * 100
        st.markdown(f'<div class="resultado"><div style="color:{GRIS};font-weight:700">Puntaje global estimado</div><div class="grande">{r["puntaje"]:.0f}</div>'
                    f'<div>rango probable {r["puntaje"] - mae:.0f} a {r["puntaje"] + mae:.0f} · supera al {pct:.0f} % de la muestra</div>'
                    f'<div style="margin-top:8px"><b>Nivel de desempeño (SVM):</b> {r["nivel"]} &nbsp;·&nbsp; '
                    + " · ".join(f"{k} {v * 100:.0f} %" for k, v in r["proba"].items()) + "</div></div>", unsafe_allow_html=True)
        st.write("")
        with st.container(border=True):
            panel("Qué suma y qué resta en esta predicción (valores SHAP)")
            c = r["contrib"].reindex(r["contrib"].abs().sort_values(ascending=False).index).head(10)
            etiquetas = {"edad": "Edad", "educ_madre_ord": "Educación de la madre", "educ_padre_ord": "Educación del padre", "personas_por_cuarto": "Personas por cuarto",
                         "extraedad": "Extraedad", "cuestionario_incompleto": "Cuestionario incompleto", "educ_madre_desconocida": "Educ. madre desconocida",
                         "educ_padre_desconocida": "Educ. padre desconocida", "genero_F": "Género", "fami_estratovivienda": "Estrato", "fami_personashogar": "Personas en el hogar",
                         "fami_cuartoshogar": "Cuartos", "fami_tieneinternet": "Internet", "fami_tienecomputador": "Computador", "fami_tieneautomovil": "Automóvil",
                         "fami_tienelavadora": "Lavadora", "cole_naturaleza": "Naturaleza del colegio", "cole_area_ubicacion": "Área del colegio",
                         "naturaleza_area": "Naturaleza × área", "cole_jornada": "Jornada", "cole_caracter": "Carácter", "cole_genero": "Colegio (género)",
                         "cole_sede_principal": "Sede principal", "region": "Región"}
            f = go.Figure(go.Bar(x=c.values[::-1], y=[etiquetas.get(i, i) for i in c.index[::-1]], orientation="h",
                                 marker_color=[BLUE if v > 0 else ORANGE for v in c.values[::-1]], text=[f"{v:+.1f}" for v in c.values[::-1]], textposition="outside"))
            f.update_layout(xaxis_title=f"Contribución en puntos (base del modelo: {r['base']:.0f})", yaxis_title=None)
            fig(f, 360)

# ============================================================================= vista 4 · segmentos
elif st.session_state.vista == VISTAS[3]:
    NOMBRES_NIVEL = {}
    prof = D.groupby("cluster_nivel").agg(n=("punt_global", "size"), promedio=("punt_global", "mean"), edad=("edad_original", "mean"))
    orden = prof.sort_values("promedio", ascending=False).index.tolist()
    for i, c in enumerate(orden):
        NOMBRES_NIVEL[c] = ("Adultos de jornada nocturna y sabatina" if prof.loc[c, "edad"] > 25 else ("Alto desempeño" if i == 0 else "Desempeño bajo a medio"))
    rel = D[s11.AREA_COLS].sub(D[s11.AREA_COLS].mean(axis=1), axis=0).groupby(D["cluster_perfil"]).mean()
    NOMBRES_PERFIL = {c: f"Fuerte en {s11.AREA_NOMBRES[rel.loc[c].idxmax()].lower()}, débil en {s11.AREA_NOMBRES[rel.loc[c].idxmin()].lower()}" for c in rel.index}

    k = st.columns(4)
    kpi(k[0], "🧩", str(s11.K_NIVEL), "Segmentos por nivel (K-means)")
    kpi(k[1], "📐", fmt(m.siluetas["nivel"], 2), "Silueta de la segmentación por nivel")
    kpi(k[2], "🎯", str(s11.K_PERFIL), "Perfiles relativos (áreas − promedio)")
    kpi(k[3], "📐", fmt(m.siluetas["perfil"], 2), "Silueta de los perfiles")
    st.write("")
    a, b = st.columns(2)
    with a, st.container(border=True):
        panel("Segmentos por nivel: cinco puntajes por área y edad (componentes principales)")
        P = D.assign(segmento=D["cluster_nivel"].map(NOMBRES_NIVEL))
        f = px.scatter(P, x="pc1_nivel", y="pc2_nivel", color="segmento", color_discrete_sequence=[BLUE, ORANGE, "#1F3A93"], opacity=.7,
                       hover_data={"pc1_nivel": False, "pc2_nivel": False, "punt_global": True, "edad_original": True, "cole_jornada": True})
        f.update_traces(marker_size=7)
        f.update_layout(xaxis_title="Componente principal 1", yaxis_title="Componente principal 2", legend=dict(orientation="h", y=-0.2))
        fig(f, 360)
    with b, st.container(border=True):
        panel("Perfiles relativos: fortalezas por área descontado el nivel (componentes principales)")
        P = D.assign(perfil=D["cluster_perfil"].map(NOMBRES_PERFIL))
        f = px.scatter(P, x="pc1_perfil", y="pc2_perfil", color="perfil", color_discrete_sequence=[BLUE, ORANGE], opacity=.7,
                       hover_data={"pc1_perfil": False, "pc2_perfil": False, "punt_global": True})
        f.update_traces(marker_size=7)
        f.update_layout(xaxis_title="Componente principal 1", yaxis_title="Componente principal 2", legend=dict(orientation="h", y=-0.2))
        fig(f, 360)
    a, b = st.columns([1.4, 1])
    with a, st.container(border=True):
        panel("Perfil de cada segmento por nivel")
        g = D.groupby("cluster_nivel")
        t = pd.DataFrame({"Estudiantes": g.size(), "Puntaje global": g["punt_global"].mean(), "Edad": g["edad_original"].mean(),
                          **{s11.AREA_NOMBRES[a_]: g[a_].mean() for a_ in s11.AREA_COLS},
                          "% colegio oficial": g["cole_naturaleza"].apply(lambda x: (x == "OFICIAL").mean() * 100),
                          "% jornada nocturna o sabatina": g["cole_jornada"].apply(lambda x: x.isin(["NOCHE", "SABATINA"]).mean() * 100),
                          "% con internet": g["fami_tieneinternet"].apply(lambda x: (x == "Si").mean() * 100)})
        t.index = [NOMBRES_NIVEL[i] for i in t.index]
        st.dataframe(t.T.style.format("{:.1f}"), width="stretch", height=390)
    with b, st.container(border=True):
        panel("¿A qué segmento pertenece un estudiante?")
        c1, c2 = st.columns(2)
        vals = {}
        for i, a_ in enumerate(s11.AREA_COLS):
            vals[a_] = (c1 if i % 2 == 0 else c2).number_input(s11.AREA_NOMBRES[a_], 0, 100, int(D[a_].median()))
        ed = c2.number_input("Edad", 14, 80, 17)
        seg = s11.segmento_de(m, vals, ed)
        rel_u = pd.Series(vals).astype(float)
        rel_u = (rel_u - rel_u.mean()).rename(s11.AREA_NOMBRES)
        st.markdown(f"**Segmento por nivel:** {NOMBRES_NIVEL[seg['nivel']]}  \n**Perfil relativo:** {NOMBRES_PERFIL[seg['perfil']]}")
        f = go.Figure(go.Scatterpolar(r=list(rel_u.values) + [rel_u.values[0]], theta=list(rel_u.index) + [rel_u.index[0]], fill="toself", line_color=BLUE))
        f.update_layout(polar=dict(radialaxis=dict(title="puntos sobre el promedio propio")), showlegend=False)
        fig(f, 260)

# ============================================================================= vista 5 · modelos y datos
else:
    k = st.columns(4)
    kpi(k[0], "📏", fmt(m.metricas_reg["MAE"]) + " pts", "MAE en prueba · Gradient Boosting")
    kpi(k[1], "📈", fmt(m.metricas_reg["R2"], 3), "R² en prueba")
    kpi(k[2], "🎯", fmt(m.metricas_clf["F1 macro"], 3), "F1 macro en prueba · SVM")
    kpi(k[3], "📐", fmt(m.metricas_clf["AUC-ROC"], 3), "AUC-ROC (uno contra el resto)")
    st.write("")
    a, b = st.columns([1, 1])
    with a, st.container(border=True):
        panel("Comparación de modelos de regresión (prueba, 150 estudiantes)")
        comp = pd.DataFrame({"Modelo": ["Gradient Boosting", "Random Forest", "Perceptrón multicapa (PyTorch)", "ElasticNet", "Ridge", "Lasso", "Predecir la media"],
                             "MAE cv": [34.53, 34.62, 35.19, 34.76, 34.77, 34.93, np.nan], "MAE prueba": [34.33, 33.81, 34.37, 34.97, 35.00, 35.22, 42.20],
                             "RMSE prueba": [43.31, 42.45, 42.58, 42.75, 42.72, 43.27, 50.9], "R² prueba": [0.286, 0.314, 0.310, 0.304, 0.305, 0.287, 0.0]}).set_index("Modelo")
        st.dataframe(comp.style.format("{:.2f}", na_rep="—").format({"R² prueba": "{:.3f}"}), width="stretch", height=290)
        st.caption("Resultados del cuaderno 03; el Gradient Boosting (100 árboles de profundidad 2) es el mejor por validación cruzada y es el que usa el tablero. "
                   f"Entrenado aquí al arrancar: MAE {m.metricas_reg['MAE']:.2f}, RMSE {m.metricas_reg['RMSE']:.2f}, R² {m.metricas_reg['R2']:.3f}.")
    with b, st.container(border=True):
        panel("Comparación de clasificadores del nivel de desempeño (prueba)")
        comp = pd.DataFrame({"Modelo": ["SVM (RBF)", "KNN", "Regresión logística", "Random Forest", "Azar"],
                             "F1 cv": [0.508, 0.482, 0.507, 0.503, np.nan], "F1 macro prueba": [0.498, 0.502, 0.482, 0.463, 0.333],
                             "Exactitud": [0.500, 0.500, 0.480, 0.473, 0.333], "AUC-ROC": [0.680, 0.681, 0.664, 0.693, 0.500]}).set_index("Modelo")
        st.dataframe(comp.style.format("{:.3f}", na_rep="—"), width="stretch", height=240)
        st.caption("La clase Medio es la difícil (F1 0,38): los terciles cortan un puntaje continuo donde el contexto no distingue a los estudiantes vecinos. "
                   f"SVM entrenada aquí: F1 macro {m.metricas_clf['F1 macro']:.3f}, exactitud {m.metricas_clf['exactitud']:.3f}, AUC {m.metricas_clf['AUC-ROC']:.3f}.")
    a, b = st.columns([1, 1])
    with a, st.container(border=True):
        panel("Importancia de las variables en el Gradient Boosting")
        imp = m.importancias.head(12)
        nombres = {"edad": "Edad", "educ_madre_ord": "Educación de la madre", "educ_padre_ord": "Educación del padre", "region": "Región", "fami_tieneinternet": "Internet",
                   "cole_jornada": "Jornada", "cole_naturaleza": "Naturaleza del colegio", "genero_F": "Género", "fami_personashogar": "Personas en el hogar",
                   "fami_estratovivienda": "Estrato", "fami_cuartoshogar": "Cuartos", "fami_tienecomputador": "Computador", "personas_por_cuarto": "Personas por cuarto",
                   "naturaleza_area": "Naturaleza × área", "extraedad": "Extraedad", "cole_caracter": "Carácter", "fami_tieneautomovil": "Automóvil",
                   "cole_area_ubicacion": "Área del colegio", "fami_tienelavadora": "Lavadora", "cole_genero": "Colegio (género)"}
        f = px.bar(x=imp.values[::-1], y=[nombres.get(i, i) for i in imp.index[::-1]], orientation="h", color_discrete_sequence=[YELLOW], text=[f"{v:.3f}" for v in imp.values[::-1]])
        f.update_traces(textposition="outside")
        f.update_layout(xaxis_title="Reducción de impureza (suma = 1)", yaxis_title=None, xaxis_range=[0, imp.max() * 1.2])
        fig(f, 380)
    with b, st.container(border=True):
        panel("Sobre los datos y el método")
        st.markdown(f"""
- **Fuente:** ICFES, *Resultados únicos Saber 11*, periodo 2022-4, publicado en [datos.gov.co](https://www.datos.gov.co/Educaci-n/Resultados-nicos-Saber-11/kgxf-xxbe) (CC BY-SA 4.0).
- **Muestra:** 1.000 estudiantes extraídos por muestreo estratificado (naturaleza × área × tercil de puntaje, semilla 42) de los 532.792 evaluados; publicada en Kaggle como
  [`andyrubiano/saber-11-muestra-2022-4`](https://www.kaggle.com/datasets/{s11.DATASET}). El tablero la descarga al arrancar.
- **Preparación:** faltantes del cuestionario como categoría propia más un indicador, edad winsorizada en el percentil 99 ({df.attrs['p99_edad']:.0f} años), extraedad,
  educación de los padres en escala ordinal, personas por cuarto, estrato agrupado, seis regiones. Partición 70/15/15 estratificada; preprocesadores ajustados solo en entrenamiento.
- **Sin fuga de información:** los puntajes por área no se usan para predecir el global (lo determinan por fórmula); solo entran en la segmentación.
- **Niveles de desempeño:** terciles del puntaje global: Bajo ≤ {df.attrs['cortes'][0]:.0f} < Medio ≤ {df.attrs['cortes'][1]:.0f} < Alto.
- **Cuadernos del proyecto:** [github.com/RubianoAndy/saber11-project](https://github.com/RubianoAndy/saber11-project) · código del tablero:
  [github.com/RubianoAndy/saber11-dashboard](https://github.com/RubianoAndy/saber11-dashboard).
""")
