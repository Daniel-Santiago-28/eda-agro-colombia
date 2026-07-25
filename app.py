"""Extractor y EDA de cifras de entrevistas (Groq · Llama 3.3 70B)."""
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import streamlit as st
from groq import Groq

st.set_page_config(page_title="Entrevistas | Extractor + EDA", page_icon="📋", layout="wide")

MODEL_ID = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Paleta y estilo (consistentes con el resto de dashboards del curso)
# ---------------------------------------------------------------------------
CAT_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK_PRIMARY, INK_SECONDARY = "#fcfcfb", "#0b0b0b", "#52514e"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"

sns.set_style("whitegrid", {
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "grid.color": GRID,
    "axes.edgecolor": BASELINE, "text.color": INK_PRIMARY, "axes.labelcolor": INK_PRIMARY,
    "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY, "font.family": "sans-serif",
})
plt.rcParams["axes.grid.axis"] = "y"

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(family="Segoe UI, system-ui, sans-serif", color=INK_PRIMARY),
    xaxis=dict(gridcolor=GRID, zerolinecolor=BASELINE, linecolor=BASELINE),
    yaxis=dict(gridcolor=GRID, zerolinecolor=BASELINE, linecolor=BASELINE),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=50, l=10, r=10, b=10),
)


def style_fig(fig, **overrides):
    fig.update_layout(**{**PLOTLY_LAYOUT, **overrides})
    return fig


COLUMNS = ["variable", "valor", "unidad", "periodo", "contexto"]

EXTRACTION_SYSTEM_PROMPT = (
    "Eres un analista de datos que extrae cifras cuantitativas mencionadas en transcripciones de entrevistas "
    "u otros textos narrativos. Lee el texto del usuario y devuelve EXCLUSIVAMENTE un objeto JSON con esta "
    'forma exacta: {"datos": [{"variable": str, "valor": number, "unidad": str, "periodo": str|null, '
    '"contexto": str}, ...]}.\n'
    "Reglas:\n"
    "- 'variable': nombre corto y claro del indicador (ej. 'Ingresos anuales', 'Número de empleados').\n"
    "- 'valor': el número en formato numérico plano (sin símbolos de moneda ni separadores de miles).\n"
    "- 'unidad': la unidad del valor (ej. 'USD', 'COP', '%', 'personas', 'años'); usa '' si no aplica.\n"
    "- 'periodo': año, fecha o periodo asociado si el texto lo menciona (ej. '2023', 'Q1 2024'); usa null si "
    "no se menciona.\n"
    "- 'contexto': una frase muy breve (máximo 12 palabras) que da contexto a la cifra.\n"
    "- Extrae TODAS las cifras cuantificables relevantes del texto, una por objeto.\n"
    "- No inventes cifras que no estén en el texto. No incluyas texto fuera del JSON."
)


def _clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return text


def extract_interview_data(client: Groq, texto: str) -> pd.DataFrame:
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": texto},
        ],
        temperature=0.1,
        max_tokens=2000,
        response_format={"type": "json_object"},
    )
    payload = json.loads(_clean_json_text(response.choices[0].message.content))
    registros = payload.get("datos", [])

    data = pd.DataFrame(registros)
    for col in COLUMNS:
        if col not in data.columns:
            data[col] = None
    data = data[COLUMNS]
    data["valor"] = pd.to_numeric(data["valor"], errors="coerce")
    data = data.dropna(subset=["valor"]).reset_index(drop=True)
    return data


def extract_year(value) -> "int | None":
    if pd.isna(value):
        return None
    match = re.search(r"(19|20)\d{2}", str(value))
    return int(match.group()) if match else None


# ---------------------------------------------------------------------------
# Barra lateral: configuración
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuración")
groq_key = st.sidebar.text_input(
    "GROQ API Key", type="password",
    help="Tu clave solo se usa en esta sesión de navegador, no se guarda en disco.",
)
st.sidebar.caption(f"Modelo de extracción: `{MODEL_ID}` vía Groq")

if "extracted_df" not in st.session_state:
    st.session_state.extracted_df = None
if "extract_version" not in st.session_state:
    st.session_state.extract_version = 0

st.sidebar.divider()
if st.session_state.extracted_df is not None and st.sidebar.button("🗑️ Borrar datos extraídos"):
    st.session_state.extracted_df = None
    st.session_state.extract_version += 1
    st.rerun()

# ---------------------------------------------------------------------------
# Encabezado y entrada de texto
# ---------------------------------------------------------------------------
st.title("📋 Extractor y EDA de cifras de entrevistas")
st.caption(
    "Pega el texto de una entrevista (o sube un .txt), extrae automáticamente las cifras mencionadas con "
    "Llama 3.3 70B en Groq, y analízalas con un EDA interactivo."
)

fuente = st.radio("Fuente del texto", ["Escribir / pegar texto", "Subir archivo .txt"], horizontal=True)

if fuente == "Escribir / pegar texto":
    texto_input = st.text_area(
        "Texto de la entrevista",
        height=240,
        placeholder=(
            "Ej: En 2020 la empresa tenía 15 empleados y facturó $2.000.000 USD. Para 2023, la plantilla "
            "creció a 45 personas y los ingresos llegaron a $8.500.000 USD, con una satisfacción del "
            "cliente del 89%, frente al 72% de 2020..."
        ),
    )
else:
    uploaded_file = st.file_uploader("Sube un archivo .txt", type=["txt"])
    texto_input = uploaded_file.read().decode("utf-8", errors="ignore") if uploaded_file else ""
    if texto_input:
        with st.expander("Vista previa del texto cargado"):
            st.text(texto_input[:3000])

extract_clicked = st.button(
    "🔎 Extraer datos con IA", type="primary",
    disabled=not (groq_key and texto_input.strip()),
)
if not groq_key:
    st.info("Ingresa tu GROQ API Key en la barra lateral para habilitar la extracción.")

if extract_clicked:
    with st.spinner("Extrayendo cifras del texto..."):
        try:
            nuevo_df = extract_interview_data(Groq(api_key=groq_key), texto_input)
            if nuevo_df.empty:
                st.warning("No se encontraron cifras cuantificables en el texto proporcionado.")
            else:
                st.session_state.extracted_df = nuevo_df
                st.session_state.extract_version += 1
                st.success(f"Se extrajeron {len(nuevo_df)} cifras del texto.")
        except Exception as exc:  # noqa: BLE001 — se informa el error al usuario, no se rompe la app
            st.error(f"⚠️ Ocurrió un error al extraer los datos: {exc}")

# ---------------------------------------------------------------------------
# Tabla editable + EDA
# ---------------------------------------------------------------------------
if st.session_state.extracted_df is not None:
    st.divider()
    st.subheader("Datos extraídos (editable)")
    st.caption("Revisa y corrige la extracción si es necesario antes del análisis; puedes añadir o borrar filas.")

    edited_df = st.data_editor(
        st.session_state.extracted_df,
        num_rows="dynamic",
        width="stretch",
        column_config={"valor": st.column_config.NumberColumn("Valor", format="%.2f")},
        key=f"editor_datos_{st.session_state.extract_version}",
    )
    edited_df = edited_df.dropna(subset=["variable", "valor"]).copy()
    edited_df["unidad"] = edited_df["unidad"].fillna("").replace("", "N/A")

    st.download_button(
        "⬇️ Descargar datos extraídos (CSV)",
        edited_df.to_csv(index=False).encode("utf-8"),
        "cifras_entrevista.csv",
        "text/csv",
    )

    if edited_df.empty:
        st.warning("No hay datos válidos para analizar.")
        st.stop()

    edited_df["_anio"] = edited_df["periodo"].apply(extract_year)
    UNIDAD_COLORS = dict(zip(sorted(edited_df["unidad"].unique()), CAT_PALETTE))

    st.subheader("📊 EDA — Análisis exploratorio de las cifras extraídas")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cifras extraídas", len(edited_df))
    k2.metric("Variables únicas", edited_df["variable"].nunique())
    k3.metric("Unidades distintas", edited_df["unidad"].nunique())
    anios_validos = edited_df["_anio"].dropna()
    k4.metric(
        "Rango de periodos",
        f"{int(anios_validos.min())}–{int(anios_validos.max())}" if not anios_validos.empty else "No especificado",
    )

    st.markdown("**Estadísticas por unidad**")
    resumen = edited_df.groupby("unidad")["valor"].agg(["count", "mean", "min", "max"]).round(2)
    resumen.columns = ["Nº cifras", "Promedio", "Mínimo", "Máximo"]
    st.dataframe(resumen, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Comparación de cifras por variable**")
        fig = px.bar(
            edited_df.sort_values("valor"), x="valor", y="variable", color="unidad", orientation="h",
            color_discrete_map=UNIDAD_COLORS, hover_data=["periodo", "contexto"],
            labels={"valor": "Valor", "variable": "", "unidad": "Unidad"},
        )
        style_fig(fig, height=max(320, 40 * len(edited_df)), legend_title_text="Unidad")
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown("**Distribución de valores por unidad**")
        fig, ax = plt.subplots(figsize=(6, max(4, 0.35 * len(edited_df) + 2)))
        sns.boxplot(
            data=edited_df, x="unidad", y="valor", hue="unidad", palette=UNIDAD_COLORS,
            legend=False, showfliers=False, ax=ax,
        )
        sns.stripplot(data=edited_df, x="unidad", y="valor", color=INK_SECONDARY, size=6, alpha=0.7, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Valor")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        sns.despine(ax=ax, left=True)
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown("**Evolución en el tiempo (variables con más de un periodo)**")
    trend_df = edited_df.dropna(subset=["_anio"])
    multi_period_vars = trend_df.groupby("variable")["_anio"].nunique()
    multi_period_vars = multi_period_vars[multi_period_vars > 1].index
    if len(multi_period_vars) > 0:
        trend_df = trend_df[trend_df["variable"].isin(multi_period_vars)].sort_values("_anio")
        var_colors = dict(zip(sorted(trend_df["variable"].unique()), CAT_PALETTE))
        fig = px.line(
            trend_df, x="_anio", y="valor", color="variable", markers=True,
            color_discrete_map=var_colors,
            labels={"_anio": "Año", "valor": "Valor", "variable": "Variable"},
        )
        fig.update_xaxes(dtick=1)
        style_fig(fig, height=380, legend_title_text="Variable")
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption(
            "No se detectaron variables con cifras en más de un periodo, así que no hay una tendencia que graficar."
        )
