"""
☁️ WordCloud — Nube de Palabras Interactiva
Aplicación Streamlit para generar nubes de palabras desde texto, archivos o URLs

Instalación:
    pip install streamlit wordcloud matplotlib pandas nltk requests beautifulsoup4 Pillow

Ejecución:
    streamlit run wordcloud_app.py
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import re
import io
import base64
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from PIL import Image

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="WordCloud Studio",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# ESTILOS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Crimson+Pro:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Crimson Pro', serif;
    }

    .stApp {
        background: #faf8f3;
        background-image:
            radial-gradient(ellipse at 20% 20%, rgba(139, 100, 60, 0.06) 0%, transparent 60%),
            radial-gradient(ellipse at 80% 80%, rgba(60, 100, 139, 0.06) 0%, transparent 60%);
    }

    [data-testid="stSidebar"] {
        background: #f0ebe0;
        border-right: 2px solid #d4c5a9;
    }

    h1 {
        font-family: 'Playfair Display', serif !important;
        color: #2c1810 !important;
        font-weight: 900 !important;
        letter-spacing: -1px !important;
    }
    h2, h3 {
        font-family: 'Playfair Display', serif !important;
        color: #3d2314 !important;
        font-weight: 700 !important;
    }

    p, label, .stMarkdown, li {
        color: #4a3728 !important;
        font-size: 1.05rem !important;
        line-height: 1.7 !important;
    }

    .stTextArea textarea {
        background: #fffdf7 !important;
        border: 2px solid #c8b89a !important;
        border-radius: 4px !important;
        color: #2c1810 !important;
        font-family: 'Crimson Pro', serif !important;
        font-size: 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #8b6428 !important;
        box-shadow: 0 0 0 3px rgba(139, 100, 40, 0.12) !important;
    }

    .stTextInput input {
        background: #fffdf7 !important;
        border: 2px solid #c8b89a !important;
        border-radius: 4px !important;
        color: #2c1810 !important;
        font-family: 'Crimson Pro', serif !important;
    }

    .stSelectbox > div > div {
        background: #fffdf7 !important;
        border: 2px solid #c8b89a !important;
        color: #2c1810 !important;
        font-family: 'Crimson Pro', serif !important;
    }

    .stSlider > div > div > div {
        background: #8b6428 !important;
    }

    .stButton > button {
        background: #2c1810 !important;
        color: #faf8f3 !important;
        border: none !important;
        border-radius: 3px !important;
        font-family: 'Playfair Display', serif !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 1px !important;
        padding: 0.65rem 1.5rem !important;
        width: 100% !important;
        text-transform: uppercase !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: #8b6428 !important;
        box-shadow: 0 4px 16px rgba(44, 24, 16, 0.25) !important;
        transform: translateY(-1px) !important;
    }

    [data-testid="metric-container"] {
        background: #fffdf7;
        border: 2px solid #d4c5a9;
        border-radius: 4px;
        padding: 14px 18px;
    }
    [data-testid="metric-container"] label {
        color: #8b6428 !important;
        font-family: 'Playfair Display', serif !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #2c1810 !important;
        font-family: 'Playfair Display', serif !important;
        font-weight: 900 !important;
    }

    hr { border-color: #d4c5a9 !important; border-width: 1px !important; }

    .stAlert {
        border-radius: 4px !important;
        border-left: 4px solid #8b6428 !important;
    }

    .tag-word {
        display: inline-block;
        background: #f0ebe0;
        border: 1px solid #c8b89a;
        border-radius: 3px;
        padding: 2px 10px;
        margin: 3px;
        font-family: 'Crimson Pro', serif;
        font-size: 0.9rem;
        color: #3d2314;
    }

    .header-banner {
        background: linear-gradient(135deg, #2c1810 0%, #4a2c1a 50%, #3d2314 100%);
        border-radius: 8px;
        padding: 32px 40px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .header-banner::before {
        content: '☁';
        position: absolute;
        right: 30px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 7rem;
        opacity: 0.08;
        pointer-events: none;
    }

    .freq-bar {
        height: 8px;
        background: linear-gradient(90deg, #8b6428, #c8963c);
        border-radius: 4px;
        display: inline-block;
        vertical-align: middle;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #d4c5a9 !important;
        border-radius: 4px !important;
        background: #fffdf7 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# STOPWORDS en español + inglés
# ─────────────────────────────────────────────
STOPWORDS_ES = {
    "de","la","el","en","y","a","los","del","se","las","un","por","con","no","una","su",
    "para","es","al","lo","como","más","pero","sus","le","ya","o","este","sí","porque",
    "esta","entre","cuando","muy","sin","sobre","también","me","hasta","hay","donde",
    "quien","desde","nos","durante","ni","contra","ese","eso","esta","ante","bajo","tras",
    "que","si","fue","son","han","ha","ser","era","está","son","están","siendo","sido",
    "he","has","hemos","habían","tiene","tienen","hacer","puede","pueden","así","tan",
    "parte","todo","todos","todas","cada","otro","otra","otros","otras","mismo","misma",
    "nuestro","nuestra","vuestro","vuestra","ellos","ellas","nosotros","vosotros",
    "les","les","eso","esa","esos","esas","aquel","aquella","aquellos","aquellas",
}

def obtener_stopwords(idioma: str) -> set:
    sw = set(STOPWORDS)  # inglés base de wordcloud
    if idioma in ("Español", "Ambos"):
        sw |= STOPWORDS_ES
    return sw


# ─────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────

def limpiar_texto(texto: str, stopwords: set, min_longitud: int) -> str:
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", "", texto)          # URLs
    texto = re.sub(r"[^a-záéíóúüñàâèêîôùûäëïöü\s]", " ", texto, flags=re.UNICODE)
    palabras = texto.split()
    palabras = [p for p in palabras if p not in stopwords and len(p) >= min_longitud]
    return " ".join(palabras)


def contar_palabras(texto_limpio: str) -> pd.DataFrame:
    palabras = texto_limpio.split()
    conteo = Counter(palabras)
    df = pd.DataFrame(conteo.most_common(50), columns=["Palabra", "Frecuencia"])
    return df


PALETAS = {
    "Otoño cálido":    ["#c0392b","#e67e22","#f39c12","#d35400","#922b21","#784212"],
    "Océano profundo": ["#1a5276","#2980b9","#5dade2","#85c1e9","#1b4f72","#2e86c1"],
    "Bosque":          ["#1d8348","#27ae60","#52be80","#a9dfbf","#145a32","#196f3d"],
    "Atardecer":       ["#6c3483","#8e44ad","#a569bd","#d2b4de","#4a235a","#7d3c98"],
    "Monocromático":   ["#2c3e50","#566573","#839192","#aab7b8","#1c2833","#424949"],
    "Fuego":           ["#922b21","#c0392b","#e74c3c","#f39c12","#f1c40f","#e67e22"],
}

FORMAS_FONDO = {
    "Sin forma (rectángulo)": None,
    "Círculo":     "circle",
    "Nube":        "cloud",
}

def crear_mascara(forma: str, size: int = 400) -> np.ndarray | None:
    if forma == "circle":
        y, x = np.ogrid[:size, :size]
        cx, cy = size // 2, size // 2
        r = size // 2 - 10
        mascara = np.ones((size, size), dtype=np.uint8) * 255
        mascara[(x - cx)**2 + (y - cy)**2 <= r**2] = 0
        return mascara
    return None


def generar_wordcloud(
    texto_limpio: str,
    paleta_nombre: str,
    max_words: int,
    fondo: str,
    forma: str,
    ancho: int = 900,
    alto: int = 500,
) -> plt.Figure:
    colores = PALETAS[paleta_nombre]

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return colores[random_state.randint(0, len(colores) - 1)]

    mascara = crear_mascara(forma, size=min(ancho, alto))

    wc = WordCloud(
        width=ancho,
        height=alto,
        max_words=max_words,
        background_color=fondo,
        color_func=color_func,
        mask=mascara,
        collocations=False,
        min_font_size=10,
        max_font_size=120,
        prefer_horizontal=0.7,
        relative_scaling=0.6,
        margin=4,
    ).generate(texto_limpio)

    fig, ax = plt.subplots(figsize=(ancho / 100, alto / 100))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_facecolor(fondo)
    plt.tight_layout(pad=0)
    return fig, wc


def fig_a_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ☁️ WordCloud Studio")
    st.markdown("*Nube de Palabras Interactiva*")
    st.divider()

    # Fuente de texto
    st.markdown("### 📄 Fuente de Texto")
    fuente = st.radio(
        "Selecciona cómo ingresar el texto:",
        ["✍️ Escribir / Pegar", "📂 Subir archivo (.txt / .csv)"],
        label_visibility="collapsed"
    )

    texto_input = ""

    if fuente == "✍️ Escribir / Pegar":
        texto_input = st.text_area(
            "Ingresa tu texto aquí:",
            height=200,
            placeholder="Pega aquí cualquier texto: artículo, reseña, discurso, canción...",
        )
        # Textos de ejemplo
        with st.expander("💡 Cargar texto de ejemplo"):
            ejemplos = {
                "Inteligencia Artificial": """
                La inteligencia artificial es una rama de la informática que busca crear sistemas capaces
                de realizar tareas que normalmente requieren inteligencia humana. El aprendizaje automático,
                las redes neuronales y el procesamiento del lenguaje natural son pilares fundamentales de
                la inteligencia artificial moderna. Los modelos de lenguaje, la visión por computadora y
                la robótica son aplicaciones que demuestran el avance de la inteligencia artificial.
                Los datos, los algoritmos y la computación son los ingredientes esenciales del aprendizaje
                profundo. La inteligencia artificial transforma industrias como la salud, la educación,
                el transporte y la manufactura.
                """,
                "Colombia": """
                Colombia es un país ubicado en el noroeste de América del Sur, conocido por su
                biodiversidad, cultura y paisajes diversos. Bogotá es la capital y ciudad más grande,
                seguida de Medellín, Cali y Barranquilla. El café colombiano es reconocido mundialmente
                por su calidad y sabor. Las flores colombianas se exportan a todo el mundo.
                Colombia tiene costas en el Océano Pacífico y en el Mar Caribe. La música vallenata,
                el cumbia y el mapalé son expresiones culturales representativas. El Amazonas, los Andes
                y el Caribe hacen de Colombia un país megadiverso con gran riqueza natural.
                """,
                "Tecnología 4.0": """
                La cuarta revolución industrial transforma los procesos productivos mediante tecnologías
                digitales avanzadas. Internet de las cosas, inteligencia artificial, big data, robótica
                y automatización son pilares de la industria 4.0. Las fábricas inteligentes integran
                sensores, datos y conectividad para optimizar la producción. La impresión 3D, la realidad
                aumentada y los gemelos digitales redefinen la manufactura. La nube, el edge computing
                y la ciberseguridad son fundamentales para la transformación digital de las empresas.
                """,
            }
            ejemplo_sel = st.selectbox("Elige un ejemplo:", list(ejemplos.keys()))
            if st.button("📥 Cargar ejemplo"):
                st.session_state["texto_ejemplo"] = ejemplos[ejemplo_sel]
                st.rerun()

        if "texto_ejemplo" in st.session_state and not texto_input:
            texto_input = st.session_state["texto_ejemplo"]

    else:
        archivo = st.file_uploader(
            "Sube tu archivo:",
            type=["txt", "csv"],
            help="Archivos .txt o .csv (se usará la primera columna de texto)"
        )
        if archivo:
            if archivo.name.endswith(".txt"):
                texto_input = archivo.read().decode("utf-8", errors="ignore")
            elif archivo.name.endswith(".csv"):
                df_csv = pd.read_csv(archivo)
                col_texto = st.selectbox("Columna de texto:", df_csv.columns.tolist())
                texto_input = " ".join(df_csv[col_texto].dropna().astype(str).tolist())
            st.success(f"✅ Archivo cargado: {len(texto_input):,} caracteres")

    st.divider()

    # Opciones de procesamiento
    st.markdown("### ⚙️ Procesamiento")
    idioma = st.selectbox("Eliminar stopwords en:", ["Español", "Inglés", "Ambos", "Ninguno"])
    min_longitud = st.slider("Longitud mínima de palabra", 2, 8, 3)

    palabras_extra = st.text_input(
        "Palabras adicionales a excluir (separadas por coma):",
        placeholder="ej: también, así, aquí"
    )

    st.divider()

    # Opciones visuales
    st.markdown("### 🎨 Apariencia")
    paleta_sel   = st.selectbox("Paleta de colores:", list(PALETAS.keys()))
    fondo_sel    = st.radio("Fondo:", ["Blanco", "Negro"], horizontal=True)
    fondo_color  = "white" if fondo_sel == "Blanco" else "black"
    forma_sel    = st.selectbox("Forma:", list(FORMAS_FONDO.keys()))
    max_words    = st.slider("Número máximo de palabras:", 20, 200, 80)

    st.divider()
    generar = st.button("☁️ GENERAR NUBE", use_container_width=True)


# ─────────────────────────────────────────────
# CONTENIDO PRINCIPAL
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1 style="color:#faf8f3 !important; margin:0; font-size:2.4rem;">☁️ WordCloud Studio</h1>
    <p style="color:#c8b89a !important; margin:8px 0 0 0; font-size:1.1rem; font-style:italic;">
        Transforma cualquier texto en una nube de palabras visual e interactiva
    </p>
</div>
""", unsafe_allow_html=True)

# Pantalla de bienvenida
if not generar or not texto_input.strip():
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
        ### ¿Qué es una Nube de Palabras?

        Una **nube de palabras** es una representación visual donde el tamaño de cada
        término refleja su **frecuencia** en el texto. Es una herramienta poderosa para:

        - 📊 Identificar los **temas principales** de un corpus
        - 🔍 Explorar **patrones léxicos** de forma intuitiva
        - 📢 Comunicar hallazgos de **análisis de texto** de manera visual
        - 🎓 Apoyar actividades de **comprensión lectora y lingüística**

        ### ¿Cómo usarlo?

        1. **Ingresa un texto** en el panel lateral (escríbelo, pégalo o sube un archivo)
        2. **Configura** el idioma, paleta de colores y número de palabras
        3. Haz clic en **☁️ GENERAR NUBE**
        4. Descarga el resultado en alta resolución

        """)
    with c2:
        st.markdown("""
        <div style="background:#fffdf7; border:2px solid #d4c5a9; border-radius:6px; padding:24px; margin-top:16px;">
            <h3 style="color:#2c1810 !important; margin-top:0;">💡 Casos de uso</h3>
            <ul style="list-style:none; padding:0;">
                <li style="padding:6px 0; border-bottom:1px solid #e8ddd0;">📰 Análisis de noticias</li>
                <li style="padding:6px 0; border-bottom:1px solid #e8ddd0;">📚 Reseñas de libros</li>
                <li style="padding:6px 0; border-bottom:1px solid #e8ddd0;">💬 Comentarios de clientes</li>
                <li style="padding:6px 0; border-bottom:1px solid #e8ddd0;">🗳️ Discursos políticos</li>
                <li style="padding:6px 0; border-bottom:1px solid #e8ddd0;">🎵 Letras de canciones</li>
                <li style="padding:6px 0;">📋 Encuestas abiertas</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    if not texto_input.strip() and generar:
        st.warning("⚠️ Ingresa un texto en el panel lateral antes de generar la nube.")
    st.stop()

# ─────────────────────────────────────────────
# PROCESAMIENTO Y GENERACIÓN
# ─────────────────────────────────────────────
stopwords_set = obtener_stopwords(idioma) if idioma != "Ninguno" else set()
if palabras_extra.strip():
    extras = {p.strip().lower() for p in palabras_extra.split(",") if p.strip()}
    stopwords_set |= extras

texto_limpio = limpiar_texto(texto_input, stopwords_set, min_longitud)

if not texto_limpio.strip():
    st.error("El texto resultante está vacío después del procesamiento. Prueba reduciendo la longitud mínima o cambiando el idioma de stopwords.")
    st.stop()

df_freq = contar_palabras(texto_limpio)
total_palabras = len(texto_limpio.split())
vocabulario = len(df_freq)

# Métricas
m1, m2, m3, m4 = st.columns(4)
m1.metric("📝 Palabras totales", f"{total_palabras:,}")
m2.metric("📖 Vocabulario único", f"{vocabulario:,}")
m3.metric("🔝 Palabra más frecuente", df_freq.iloc[0]["Palabra"] if not df_freq.empty else "—")
m4.metric("🔢 Frecuencia máxima", int(df_freq.iloc[0]["Frecuencia"]) if not df_freq.empty else 0)

st.markdown("<br>", unsafe_allow_html=True)

# ── Generar nube ──
with st.spinner("✨ Generando tu nube de palabras..."):
    forma_key = FORMAS_FONDO[forma_sel]
    fig_wc, wc_obj = generar_wordcloud(
        texto_limpio,
        paleta_sel,
        max_words,
        fondo_color,
        forma_key,
        ancho=1000,
        alto=560,
    )

# ── Mostrar nube ──
st.markdown("### ☁️ Tu Nube de Palabras")
st.pyplot(fig_wc, use_container_width=True)

# Botón de descarga
img_bytes = fig_a_bytes(fig_wc)
st.download_button(
    label="⬇️ Descargar imagen PNG (alta resolución)",
    data=img_bytes,
    file_name="wordcloud.png",
    mime="image/png",
    use_container_width=True,
)

st.divider()

# ── Análisis de frecuencia ──
col_freq, col_tabla = st.columns([3, 2])

with col_freq:
    st.markdown("### 📊 Top 20 Palabras más Frecuentes")
    top20 = df_freq.head(20)
    max_freq = top20["Frecuencia"].max()

    for _, row in top20.iterrows():
        p = row["Palabra"]
        f = int(row["Frecuencia"])
        pct = f / max_freq
        barra_w = int(pct * 180)
        st.markdown(
            f"""<div style="display:flex; align-items:center; gap:12px; margin:4px 0; padding:6px 10px;
                           background:#fffdf7; border:1px solid #e8ddd0; border-radius:3px;">
                <span style="font-family:'Crimson Pro',serif; font-weight:600; color:#2c1810;
                             min-width:130px; font-size:1rem;">{p}</span>
                <div class="freq-bar" style="width:{barra_w}px;"></div>
                <span style="font-family:'Playfair Display',serif; font-weight:700;
                             color:#8b6428; min-width:32px; text-align:right;">{f}</span>
            </div>""",
            unsafe_allow_html=True
        )

with col_tabla:
    st.markdown("### 📋 Tabla de Frecuencias")
    st.dataframe(
        df_freq.head(30).style
               .background_gradient(subset=["Frecuencia"], cmap="YlOrBr")
               .format({"Frecuencia": "{:,}"}),
        use_container_width=True,
        height=480,
    )
    csv_freq = df_freq.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Exportar frecuencias (.csv)",
        data=csv_freq,
        file_name="frecuencias.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()

# ── Texto procesado (expandible) ──
with st.expander("🔍 Ver texto procesado (después de eliminar stopwords)"):
    st.markdown(
        f"<p style='font-family:Crimson Pro, serif; line-height:1.9; color:#4a3728; font-size:0.95rem;'>{texto_limpio[:2000]}{'...' if len(texto_limpio) > 2000 else ''}</p>",
        unsafe_allow_html=True
    )

plt.close("all")
