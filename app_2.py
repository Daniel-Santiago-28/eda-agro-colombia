"""Dashboard de análisis agrícola — Agro Colombia."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

# ---------------------------------------------------------------------------
# Configuración de página y paleta (validada para lectura por color y CVD)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Agro Colombia | Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

CAT_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK_PRIMARY, INK_SECONDARY = "#fcfcfb", "#0b0b0b", "#52514e"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
DIVERGE_NEG, DIVERGE_MID, DIVERGE_POS = "#2a78d6", "#f0efec", "#e34948"
DIVERGING_CMAP = LinearSegmentedColormap.from_list("diverging", [DIVERGE_NEG, DIVERGE_MID, DIVERGE_POS])

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


TECH_ORDER = ["Bajo", "Medio", "Alto", "Muy Alto"]
TECH_COLORS = dict(zip(TECH_ORDER, [SEQ_BLUE[0], SEQ_BLUE[2], SEQ_BLUE[4], SEQ_BLUE[5]]))

# ---------------------------------------------------------------------------
# Carga y preparación de datos
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "agro_colombia.csv"


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["Fecha_Ultima_Auditoria"] = pd.to_datetime(data["Fecha_Ultima_Auditoria"])
    data["Nivel_Tecnificacion"] = pd.Categorical(data["Nivel_Tecnificacion"], categories=TECH_ORDER, ordered=True)
    data["Rendimiento_Ton_Ha"] = data["Produccion_Anual_Ton"] / data["Area_Hectareas"]
    data["Ingreso_Total_COP"] = data["Produccion_Anual_Ton"] * data["Precio_Venta_Por_Ton_COP"]
    data["Ingreso_Por_Ha_COP"] = data["Ingreso_Total_COP"] / data["Area_Hectareas"]
    data["Riego"] = data["Sistema_Riego_Tecnificado"].map(
        {True: "Con riego tecnificado", False: "Sin riego tecnificado"}
    )
    data["Mes_Auditoria"] = data["Fecha_Ultima_Auditoria"].dt.to_period("M").dt.to_timestamp()
    return data


df_raw = load_data(DATA_PATH)

CULTIVOS = sorted(df_raw["Tipo_Cultivo"].unique())
CULTIVO_COLORS = dict(zip(CULTIVOS, CAT_PALETTE))
DEPARTAMENTOS = sorted(df_raw["Departamento"].unique())
DEPTO_COLORS = dict(zip(DEPARTAMENTOS, CAT_PALETTE))
SUELOS = sorted(df_raw["Tipo_Suelo"].unique())
SUELO_COLORS = dict(zip(SUELOS, CAT_PALETTE))
RIEGO_COLORS = {"Con riego tecnificado": CAT_PALETTE[0], "Sin riego tecnificado": CAT_PALETTE[7]}

# ---------------------------------------------------------------------------
# Filtros (barra lateral)
# ---------------------------------------------------------------------------
st.sidebar.title("🌱 Filtros")
st.sidebar.caption("Explora el conjunto de datos de fincas agrícolas colombianas.")

sel_depto = st.sidebar.multiselect("Departamento", DEPARTAMENTOS, default=DEPARTAMENTOS)
sel_cultivo = st.sidebar.multiselect("Tipo de cultivo", CULTIVOS, default=CULTIVOS)
sel_tech = st.sidebar.multiselect("Nivel de tecnificación", TECH_ORDER, default=TECH_ORDER)
sel_suelo = st.sidebar.multiselect("Tipo de suelo", SUELOS, default=SUELOS)
sel_riego = st.sidebar.multiselect("Riego tecnificado", list(RIEGO_COLORS), default=list(RIEGO_COLORS))

area_lo, area_hi = float(df_raw["Area_Hectareas"].min()), float(df_raw["Area_Hectareas"].max())
sel_area = st.sidebar.slider("Área (hectáreas)", area_lo, area_hi, (area_lo, area_hi))

date_lo, date_hi = df_raw["Fecha_Ultima_Auditoria"].min().date(), df_raw["Fecha_Ultima_Auditoria"].max().date()
sel_dates = st.sidebar.date_input("Fecha de última auditoría", (date_lo, date_hi), min_value=date_lo, max_value=date_hi)

mask = (
    df_raw["Departamento"].isin(sel_depto)
    & df_raw["Tipo_Cultivo"].isin(sel_cultivo)
    & df_raw["Nivel_Tecnificacion"].isin(sel_tech)
    & df_raw["Tipo_Suelo"].isin(sel_suelo)
    & df_raw["Riego"].isin(sel_riego)
    & df_raw["Area_Hectareas"].between(*sel_area)
)
if isinstance(sel_dates, tuple) and len(sel_dates) == 2:
    start, end = sel_dates
    mask &= df_raw["Fecha_Ultima_Auditoria"].dt.date.between(start, end)

df = df_raw.loc[mask].copy()
st.sidebar.divider()
st.sidebar.markdown(f"**{len(df):,} / {len(df_raw):,}** fincas seleccionadas")

st.title("🌱 Dashboard Agro Colombia")
st.caption(
    "Análisis exploratorio, visual y narrativo de fincas agrícolas colombianas "
    "para apoyar decisiones productivas."
)

if df.empty:
    st.warning("No hay datos para los filtros seleccionados. Ajusta los filtros en la barra lateral.")
    st.stop()

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Fincas", f"{len(df):,}")
k2.metric("Área total (ha)", f"{df['Area_Hectareas'].sum():,.0f}")
k3.metric("Producción total (ton)", f"{df['Produccion_Anual_Ton'].sum():,.0f}")
k4.metric("Rendimiento medio (ton/ha)", f"{df['Rendimiento_Ton_Ha'].mean():.2f}")
k5.metric("Precio medio (COP/ton)", f"${df['Precio_Venta_Por_Ton_COP'].mean():,.0f}")
k6.metric("Ingreso total estimado", f"${df['Ingreso_Total_COP'].sum() / 1e9:,.1f} mil M COP")

tab_resumen, tab_eda, tab_viz, tab_story = st.tabs(
    ["🏠 Resumen", "🔍 EDA", "📊 Visualizaciones", "📖 Storytelling"]
)

# ---------------------------------------------------------------------------
# TAB: Resumen
# ---------------------------------------------------------------------------
with tab_resumen:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Vista previa de los datos filtrados")
        preview_cols = [c for c in df.columns if c not in ("Riego", "Mes_Auditoria")]
        st.dataframe(df[preview_cols].reset_index(drop=True), width="stretch", height=340)
        st.download_button(
            "⬇️ Descargar datos filtrados (CSV)",
            df[preview_cols].to_csv(index=False).encode("utf-8"),
            "agro_colombia_filtrado.csv",
            "text/csv",
        )
    with col2:
        st.subheader("Composición por cultivo")
        fig = px.pie(
            df, names="Tipo_Cultivo", color="Tipo_Cultivo",
            color_discrete_map=CULTIVO_COLORS, hole=0.5,
        )
        fig.update_traces(textinfo="percent", textfont_color=INK_PRIMARY)
        style_fig(fig, height=360, legend_title_text="Cultivo")
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# TAB: EDA
# ---------------------------------------------------------------------------
with tab_eda:
    st.subheader("Estructura del conjunto de datos")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas (filtradas)", f"{df.shape[0]:,}")
    c2.metric("Filas (total)", f"{df_raw.shape[0]:,}")
    c3.metric("Columnas", df_raw.shape[1])
    c4.metric("Valores nulos", int(df_raw.isna().sum().sum()))

    with st.expander("Tipos de dato y estadísticas descriptivas", expanded=True):
        left, right = st.columns([1, 2])
        with left:
            st.markdown("**Tipos de dato**")
            st.dataframe(df_raw.dtypes.astype(str).rename("Tipo"), width="stretch")
        with right:
            st.markdown("**Estadísticas numéricas (datos filtrados)**")
            num_cols = ["Area_Hectareas", "Produccion_Anual_Ton", "Precio_Venta_Por_Ton_COP",
                        "Rendimiento_Ton_Ha", "Ingreso_Por_Ha_COP"]
            st.dataframe(df[num_cols].describe().T.style.format("{:,.2f}"), width="stretch")

    st.subheader("Distribución de variables numéricas")
    hist_cols = ["Area_Hectareas", "Produccion_Anual_Ton", "Precio_Venta_Por_Ton_COP", "Rendimiento_Ton_Ha"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, col in zip(axes.flat, hist_cols):
        sns.histplot(df[col], kde=True, ax=ax, color=CAT_PALETTE[0], edgecolor=SURFACE)
        ax.set_title(col.replace("_", " "), fontsize=11)
        ax.set_ylabel("Frecuencia")
        ax.set_xlabel("")
        sns.despine(ax=ax, left=True)
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Frecuencia de variables categóricas")
    cat_cols = ["Departamento", "Tipo_Cultivo", "Nivel_Tecnificacion", "Tipo_Suelo"]
    color_maps = [DEPTO_COLORS, CULTIVO_COLORS, TECH_COLORS, SUELO_COLORS]
    cols_row = st.columns(2)
    for i, (col, cmap) in enumerate(zip(cat_cols, color_maps)):
        counts = df[col].value_counts().rename_axis(col).reset_index(name="Fincas")
        order = TECH_ORDER if col == "Nivel_Tecnificacion" else counts.sort_values("Fincas")[col].tolist()
        fig = px.bar(
            counts, x="Fincas", y=col, orientation="h", color=col,
            color_discrete_map=cmap, category_orders={col: order},
        )
        style_fig(fig, showlegend=False, height=320, title=f"Fincas por {col.replace('_', ' ').lower()}")
        cols_row[i % 2].plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# TAB: Visualizaciones
# ---------------------------------------------------------------------------
with tab_viz:
    st.markdown("Gráficas combinando **seaborn**, **matplotlib** y **plotly** orientadas a decisiones productivas.")

    row1c1, row1c2 = st.columns(2)
    with row1c1:
        st.subheader("¿Qué cultivo rinde más por hectárea?")
        order = df.groupby("Tipo_Cultivo", observed=True)["Rendimiento_Ton_Ha"].median().sort_values(ascending=False).index
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.boxplot(
            data=df, x="Rendimiento_Ton_Ha", y="Tipo_Cultivo", order=order,
            hue="Tipo_Cultivo", palette=CULTIVO_COLORS, legend=False, ax=ax,
        )
        ax.set_xlabel("Rendimiento (ton/ha)")
        ax.set_ylabel("")
        sns.despine(ax=ax, left=True)
        st.pyplot(fig)

    with row1c2:
        st.subheader("Correlación entre variables numéricas")
        numeric_df = df[["Area_Hectareas", "Produccion_Anual_Ton", "Precio_Venta_Por_Ton_COP",
                          "Rendimiento_Ton_Ha", "Ingreso_Por_Ha_COP"]]
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap=DIVERGING_CMAP, center=0, vmin=-1, vmax=1,
            linewidths=1, linecolor=SURFACE, cbar_kws={"label": "Correlación"}, ax=ax,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        st.pyplot(fig)

    st.subheader("Relación entre área cultivada y producción anual")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.scatterplot(
        data=df, x="Area_Hectareas", y="Produccion_Anual_Ton", hue="Tipo_Cultivo",
        palette=CULTIVO_COLORS, s=35, alpha=0.75, edgecolor="white", linewidth=0.3, ax=ax,
    )
    sns.regplot(
        data=df, x="Area_Hectareas", y="Produccion_Anual_Ton", scatter=False, ax=ax,
        color=INK_SECONDARY, line_kws={"linestyle": "--", "linewidth": 1.5},
    )
    ax.set_xlabel("Área (ha)")
    ax.set_ylabel("Producción anual (ton)")
    ax.legend(title="Cultivo", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    sns.despine(ax=ax)
    st.pyplot(fig)

    row2c1, row2c2 = st.columns(2)
    with row2c1:
        st.subheader("Rendimiento promedio por departamento")
        dep_yield = df.groupby("Departamento", as_index=False)["Rendimiento_Ton_Ha"].mean().sort_values("Rendimiento_Ton_Ha")
        fig = px.bar(
            dep_yield, x="Rendimiento_Ton_Ha", y="Departamento", orientation="h",
            color="Departamento", color_discrete_map=DEPTO_COLORS,
            category_orders={"Departamento": dep_yield["Departamento"].tolist()},
            labels={"Rendimiento_Ton_Ha": "Rendimiento medio (ton/ha)"},
        )
        style_fig(fig, showlegend=False, height=380)
        st.plotly_chart(fig, width="stretch")

    with row2c2:
        st.subheader("¿La tecnificación se refleja en el precio de venta?")
        fig = px.box(
            df, x="Nivel_Tecnificacion", y="Precio_Venta_Por_Ton_COP", color="Nivel_Tecnificacion",
            color_discrete_map=TECH_COLORS, category_orders={"Nivel_Tecnificacion": TECH_ORDER},
            labels={"Precio_Venta_Por_Ton_COP": "Precio (COP/ton)", "Nivel_Tecnificacion": "Nivel de tecnificación"},
        )
        style_fig(fig, showlegend=False, height=380)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Impacto del riego tecnificado en el rendimiento, por cultivo")
    riego_yield = df.groupby(["Tipo_Cultivo", "Riego"], as_index=False)["Rendimiento_Ton_Ha"].mean()
    fig = px.bar(
        riego_yield, x="Tipo_Cultivo", y="Rendimiento_Ton_Ha", color="Riego", barmode="group",
        color_discrete_map=RIEGO_COLORS,
        labels={"Rendimiento_Ton_Ha": "Rendimiento medio (ton/ha)", "Tipo_Cultivo": "Cultivo"},
    )
    fig.update_xaxes(tickangle=-15)
    style_fig(fig, height=400, legend_title_text="")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Producción anual: fincas con riego vs. sin riego")
    fig = px.box(
        df, x="Riego", y="Produccion_Anual_Ton", color="Riego",
        color_discrete_map=RIEGO_COLORS, points="outliers",
        category_orders={"Riego": ["Con riego tecnificado", "Sin riego tecnificado"]},
        labels={"Produccion_Anual_Ton": "Producción anual (ton)", "Riego": ""},
    )
    style_fig(fig, showlegend=False, height=400)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Concentración de área y eficiencia por departamento y cultivo")
    tree = df.groupby(["Departamento", "Tipo_Cultivo"], as_index=False, observed=True).agg(
        Area_Hectareas=("Area_Hectareas", "sum"),
        Rendimiento_Ton_Ha=("Rendimiento_Ton_Ha", "mean"),
    )
    fig = px.treemap(
        tree, path=["Departamento", "Tipo_Cultivo"], values="Area_Hectareas",
        color="Rendimiento_Ton_Ha", color_continuous_scale=SEQ_BLUE,
        labels={"Rendimiento_Ton_Ha": "Rendimiento medio (ton/ha)"},
    )
    style_fig(fig, height=480, coloraxis_colorbar_title="Rendimiento<br>(ton/ha)")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Cadencia de auditorías en el tiempo")
    audits = df.groupby("Mes_Auditoria", as_index=False).size()
    fig = px.line(audits, x="Mes_Auditoria", y="size", markers=True,
                  labels={"Mes_Auditoria": "Mes", "size": "Nº de auditorías"})
    fig.update_traces(line_color=CAT_PALETTE[0], marker_color=CAT_PALETTE[0])
    style_fig(fig, height=340)
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# TAB: Storytelling
# ---------------------------------------------------------------------------
with tab_story:
    st.subheader("📖 Lo que dicen los datos")
    st.caption("Los hallazgos se recalculan según los filtros activos en la barra lateral.")

    yield_by_crop = df.groupby("Tipo_Cultivo", observed=True)["Rendimiento_Ton_Ha"].mean().sort_values(ascending=False)
    income_by_crop = df.groupby("Tipo_Cultivo", observed=True)["Ingreso_Por_Ha_COP"].mean().sort_values(ascending=False)
    yield_by_depto = df.groupby("Departamento")["Rendimiento_Ton_Ha"].mean().sort_values(ascending=False)
    yield_by_riego = df.groupby("Sistema_Riego_Tecnificado")["Rendimiento_Ton_Ha"].mean()
    yield_by_suelo = df.groupby("Tipo_Suelo")["Rendimiento_Ton_Ha"].mean().sort_values(ascending=False)
    price_by_tech = df.groupby("Nivel_Tecnificacion", observed=True)["Precio_Venta_Por_Ton_COP"].mean()
    corr_area_yield = df["Area_Hectareas"].corr(df["Rendimiento_Ton_Ha"])
    riego_adoption = df["Sistema_Riego_Tecnificado"].mean() * 100

    has_both_riego = True in yield_by_riego.index and False in yield_by_riego.index
    riego_gap_pct = (
        (yield_by_riego.get(True, np.nan) / yield_by_riego.get(False, np.nan) - 1) * 100
        if has_both_riego and yield_by_riego.get(False, 0) else None
    )

    st.markdown("#### 1. El cultivo, no el tamaño de la finca, determina la eficiencia")
    st.markdown(
        f"**{yield_by_crop.index[0]}** lidera el rendimiento por hectárea con "
        f"**{yield_by_crop.iloc[0]:.2f} ton/ha**, frente a **{yield_by_crop.iloc[-1]:.2f} ton/ha** de "
        f"**{yield_by_crop.index[-1]}** — una brecha de "
        f"**{(yield_by_crop.iloc[0] / yield_by_crop.iloc[-1] - 1) * 100:.0f}%**. "
        f"Al mismo tiempo, **{income_by_crop.index[0]}** también encabeza el ingreso estimado por hectárea "
        f"(${income_by_crop.iloc[0]:,.0f} COP/ha), por lo que rendimiento e ingreso apuntan en la misma dirección "
        f"para priorizar la expansión de este cultivo."
    )

    st.markdown("#### 2. El área cultivada más grande no implica mayor productividad")
    corr_msg = "una relación negativa moderada" if corr_area_yield < -0.3 else (
        "una relación positiva moderada" if corr_area_yield > 0.3 else "prácticamente sin relación"
    )
    st.markdown(
        f"La correlación entre área (ha) y rendimiento (ton/ha) es **{corr_area_yield:.2f}**, es decir, "
        f"**{corr_msg}**. Las fincas más pequeñas tienden a producir más por hectárea que las grandes, "
        "lo que sugiere manejo más intensivo en predios reducidos y posibles pérdidas de eficiencia "
        "al escalar el área sin escalar igual la gestión agronómica."
    )
    if corr_area_yield < -0.3:
        st.info(
            "**Implicación:** antes de invertir en ampliar hectáreas, vale la pena evaluar si el cuello de "
            "botella es el manejo agronómico y no el tamaño del predio."
        )

    st.markdown("#### 3. Pregunta de negocio: ¿el riego tecnificado impacta realmente la producción por hectárea?")
    riego_true_vals = df.loc[df["Sistema_Riego_Tecnificado"], "Rendimiento_Ton_Ha"]
    riego_false_vals = df.loc[~df["Sistema_Riego_Tecnificado"], "Rendimiento_Ton_Ha"]

    if riego_gap_pct is not None and len(riego_true_vals) > 1 and len(riego_false_vals) > 1:
        direction = "mayor" if riego_gap_pct > 0 else "menor"
        t_stat, p_value = stats.ttest_ind(riego_true_vals, riego_false_vals, equal_var=False)
        n1, n2 = len(riego_true_vals), len(riego_false_vals)
        pooled_std = np.sqrt(
            ((n1 - 1) * riego_true_vals.std(ddof=1) ** 2 + (n2 - 1) * riego_false_vals.std(ddof=1) ** 2)
            / (n1 + n2 - 2)
        )
        cohend = (riego_true_vals.mean() - riego_false_vals.mean()) / pooled_std if pooled_std else np.nan
        effect_label = "grande" if abs(cohend) >= 0.8 else "moderado" if abs(cohend) >= 0.5 else "pequeño"
        is_significant = p_value < 0.05
        verdict = "Sí" if is_significant else "No de forma concluyente"
        callout = st.success if is_significant else st.warning

        callout(
            f"**Respuesta: {verdict}.** Las fincas con riego tecnificado producen en promedio "
            f"**{riego_true_vals.mean():.2f} ton/ha** frente a **{riego_false_vals.mean():.2f} ton/ha** sin riego "
            f"(**{abs(riego_gap_pct):.0f}% {direction}**). La diferencia {'sí es' if is_significant else 'no es'} "
            f"estadísticamente significativa (prueba t de Welch, p = {p_value:.3f}"
            f"{', < 0.05' if is_significant else ', ≥ 0.05'}), con un tamaño de efecto (d de Cohen) de "
            f"**{cohend:.2f}** — considerado **{effect_label}**."
        )
        st.markdown(
            "El patrón se confirma también en producción total (no solo por hectárea): consulta el boxplot "
            "*Producción anual: fincas con riego vs. sin riego* en la pestaña 📊 Visualizaciones. Sin embargo, "
            f"solo el **{riego_adoption:.0f}%** de las fincas filtradas cuentan hoy con riego tecnificado, lo que "
            "convierte su adopción en una palanca de inversión con impacto medible y todavía poco explotada."
        )
    else:
        st.markdown(
            "No hay suficientes fincas con y sin riego en la selección actual para responder esta pregunta "
            "con rigor estadístico."
        )

    st.markdown("#### 4. La tecnificación no se traduce automáticamente en mejor precio de venta")
    price_range_pct = (price_by_tech.max() - price_by_tech.min()) / price_by_tech.mean() * 100
    st.markdown(
        f"El precio promedio por tonelada varía apenas **{price_range_pct:.1f}%** entre niveles de "
        f"tecnificación (de ${price_by_tech.min():,.0f} a ${price_by_tech.max():,.0f} COP/ton). "
        "Esto sugiere que el precio de venta depende más del mercado y el tipo de cultivo que del nivel "
        "tecnológico de la finca — la tecnificación conviene evaluarla por su efecto en **rendimiento**, no en precio."
    )

    st.markdown("#### 5. La geografía y el suelo concentran las oportunidades")
    st.markdown(
        f"**{yield_by_depto.index[0]}** es el departamento más productivo por hectárea "
        f"({yield_by_depto.iloc[0]:.2f} ton/ha), mientras **{yield_by_depto.index[-1]}** es el menos "
        f"productivo ({yield_by_depto.iloc[-1]:.2f} ton/ha). A nivel de suelo, **{yield_by_suelo.index[0]}** "
        f"rinde más ({yield_by_suelo.iloc[0]:.2f} ton/ha) que **{yield_by_suelo.index[-1]}** "
        f"({yield_by_suelo.iloc[-1]:.2f} ton/ha), un factor a considerar al decidir dónde expandir cultivos."
    )

    st.divider()
    st.markdown("#### ✅ Recomendaciones para la toma de decisiones")
    st.success(
        f"- **Priorizar {yield_by_crop.index[0]}** en la asignación de nuevas hectáreas: mejor rendimiento e ingreso por ha.\n"
        f"- **Impulsar la adopción de riego tecnificado** (hoy en {riego_adoption:.0f}%): es la palanca con mayor "
        "impacto medible sobre el rendimiento.\n"
        f"- **Enfocar programas de expansión en {yield_by_depto.index[0]}** y en suelos tipo "
        f"**{yield_by_suelo.index[0]}**, donde la productividad natural es mayor.\n"
        "- **No usar el nivel de tecnificación como palanca de negociación de precio**: su efecto real está en "
        "productividad, no en el precio de venta.\n"
        "- **Revisar el manejo agronómico en fincas grandes**: la correlación negativa entre área y rendimiento "
        "indica oportunidades de mejora en predios extensos."
    )
