import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt
import requests
import folium
from streamlit_folium import st_folium
from io import BytesIO
from fpdf import FPDF

# ==========================================
# CONSTANTES
# ==========================================
C_LUZ     = 299_792_458      
R_TIERRA  = 6_371_000        
K_FACTOR  = 4 / 3            
FACTORES_FREC = {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}

# ==========================================
# FUNCIONES DE CÁLCULO
# ==========================================
def distancia_haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R_TIERRA * math.asin(math.sqrt(a))

def obtener_perfil_topografico(lat1, lon1, lat2, lon2, n_puntos=100):
    lats = np.linspace(lat1, lat2, n_puntos)
    lons = np.linspace(lon1, lon2, n_puntos)
    locations = "|".join([f"{lat},{lon}" for lat, lon in zip(lats, lons)])
    url = f"https://api.opentopodata.org/v1/srtm90m?locations={locations}"
    
    try:
        respuesta = requests.get(url, timeout=15)
        datos = respuesta.json()
        elevaciones = [resultado['elevation'] for resultado in datos['results']]
        return np.array(elevaciones)
    except Exception as e:
        st.warning(f"Error al obtener elevación: {e}. Usando terreno plano.")
        return np.zeros(n_puntos)

def analizar_enlace(lat1, lon1, h1, lat2, lon2, h2, freq_hz, n_puntos=100, criterio=0.6):
    d_total = distancia_haversine(lat1, lon1, lat2, lon2)
    distancias = np.linspace(0, d_total, n_puntos)
    d1, d2 = distancias, d_total - distancias

    elevaciones_reales = obtener_perfil_topografico(lat1, lon1, lat2, lon2, n_puntos)
    alt_suelo_A = elevaciones_reales[0]
    alt_suelo_B = elevaciones_reales[-1]
    
    linea_vista = (alt_suelo_A + h1) + ((alt_suelo_B + h2) - (alt_suelo_A + h1)) * (distancias / d_total)           
    bulto = (d1 * d2) / (2 * K_FACTOR * R_TIERRA)
    obstaculo_total = elevaciones_reales + bulto 
    
    wavelength = C_LUZ / freq_hz                                     
    f1 = np.sqrt(wavelength * d1 * d2 / d_total)                     
    despeje = linea_vista - obstaculo_total 
    
    interior = (distancias > 0.01 * d_total) & (distancias < 0.99 * d_total)
    relativo_i = despeje[interior] / f1[interior]
    idx_local = int(np.argmin(relativo_i))
    idx_critico = np.where(interior)[0][idx_local]

    despeje_min_pct = float(relativo_i[idx_local] * 100)
    cumple = bool(relativo_i[idx_local] >= criterio)

    return {
        "d_total_km": d_total / 1000, "distancias_km": distancias / 1000,
        "linea_vista": linea_vista, "obstaculo_total": obstaculo_total, "f1": f1, "despeje": despeje,
        "idx_critico": idx_critico, "despeje_min_pct": despeje_min_pct,
        "cumple_criterio": cumple, "f1_max": float(f1.max()),
        "wavelength": wavelength, "criterio": criterio,
        "lat1": lat1, "lon1": lon1, "h1": h1 + alt_suelo_A, "lat2": lat2, "lon2": lon2, "h2": h2 + alt_suelo_B,
    }

def generar_pdf(r):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "Diagnostico Zona de Fresnel - ITU-R P.526", ln=True, align="C")
    
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 6, "Estudiante: Laura Maria Montano Poveda | Codigo: 20242678048", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "Parametros del Enlace y Resultados:", ln=True)
    
    pdf.set_font("helvetica", "", 11)
    pdf.cell(0, 6, f"- Distancia total del enlace: {r['d_total_km']:.2f} km", ln=True)
    pdf.cell(0, 6, f"- Longitud de onda (lambda): {r['wavelength']*100:.2f} cm", ln=True)
    pdf.cell(0, 6, f"- Radio maximo de la 1ra Zona de Fresnel (F1): {r['f1_max']:.2f} m", ln=True)
    pdf.cell(0, 6, f"- Despeje minimo encontrado: {r['despeje_min_pct']:.1f}%", ln=True)
    pdf.cell(0, 6, f"- Criterio de evaluacion exigido: >= {r['criterio']*100:.0f}%", ln=True)
    
    estado = "DESPEJE ADECUADO (CUMPLE)" if r['cumple_criterio'] else "POSIBLE OBSTRUCCION"
    pdf.set_font("helvetica", "B", 11)
    pdf.ln(4)
    pdf.cell(0, 8, f"Estado del Diagnostico: {estado}", ln=True)
    
    return bytes(pdf.output())

# ==========================================
# INTERFAZ GRÁFICA (STREAMLIT)
# ==========================================
st.set_page_config(page_title="Diagnóstico Zona de Fresnel", layout="wide")

st.title("📡 Diagnóstico de la Primera Zona de Fresnel — ITU-R P.526")
st.markdown("**Integrantes del equipo:** Laura Maria Montaño Poveda")
st.markdown("**Código:** 20242678048")
st.markdown("---")

st.header("Objetivo")
st.markdown("""
Esta aplicación calcula y **diagnostica el despeje de la primera zona de Fresnel** de un enlace de radio entre dos antenas, a partir de:
- 📍 Coordenadas geográficas (latitud/longitud) de cada torre
- 📏 Altura de cada torre
- 📶 Frecuencia de operación del enlace

El cálculo sigue la geometría descrita en la **Recomendación ITU-R P.526** *(Propagation by diffraction)*, que define el radio de la n-ésima zona de Fresnel como:
""")

st.latex(r"F_n = \sqrt{\frac{n \, \lambda \, d_1 d_2}{d_1 + d_2}}")

st.markdown("""
donde $\\lambda = c/f$ es la longitud de onda de la señal, y $d_1$, $d_2$ son las distancias desde cada extremo del enlace hasta el punto evaluado a lo largo del trayecto.
""")

st.header("Criterio de diagnóstico")
st.markdown("""
Además de la geometría de Fresnel, el modelo incluye la **curvatura de la Tierra** (factor de radio efectivo $k=4/3$), que "levanta" la superficie terrestre hacia la línea de vista en el punto medio del enlace. El despeje disponible se compara contra el radio de la primera zona de Fresnel en cada punto del trayecto:
- Si el despeje ≥ **60 %** de $F_1$ en todo el trayecto → **enlace con despeje adecuado** ✅ (criterio estándar de diseño de enlaces LOS de microondas)
- Si en algún punto el despeje cae por debajo del criterio → **posible obstrucción** ⚠️
""")

st.info("""
💡 **Nota:** El modelo incorpora perfiles topográficos reales (SRTM 90m) obtenidos mediante la API de OpenTopoData junto con la curvatura terrestre, garantizando un diagnóstico preciso para casos de estudio reales.
""")

st.markdown("---")

# Menú lateral
st.sidebar.header("📍 Torre A")
lat1 = st.sidebar.number_input("Latitud A:", value=4.6486, format="%.4f")
lon1 = st.sidebar.number_input("Longitud A:", value=-74.2479, format="%.4f")
h1 = st.sidebar.number_input("Altura torre A (m):", value=30.0, step=1.0)

st.sidebar.header("📍 Torre B")
lat2 = st.sidebar.number_input("Latitud B:", value=4.6097, format="%.4f")
lon2 = st.sidebar.number_input("Longitud B:", value=-74.0817, format="%.4f")
h2 = st.sidebar.number_input("Altura torre B (m):", value=45.0, step=1.0)

st.sidebar.header("📶 Frecuencia de operación")
freq_val = st.sidebar.number_input("Frecuencia:", value=6.0, step=0.5)
freq_unit = st.sidebar.selectbox("Unidad:", options=["Hz", "kHz", "MHz", "GHz"], index=3)

st.sidebar.header("⚙️ Configuración")
criterio = st.sidebar.slider("Criterio despeje F1:", min_value=0.1, max_value=1.0, value=0.6, step=0.05)

if "resultado" not in st.session_state:
    st.session_state.resultado = None

if st.sidebar.button("📡 Calcular diagnóstico", use_container_width=True):
    freq_hz = freq_val * FACTORES_FREC[freq_unit]
    with st.spinner("Descargando relieve topográfico y calculando..."):
        try:
            st.session_state.resultado = analizar_enlace(lat1, lon1, h1, lat2, lon2, h2, freq_hz, criterio=criterio)
        except Exception as e:
            st.error(f"Error al procesar el enlace: {e}")

if st.session_state.resultado is not None:
    r = st.session_state.resultado
    
    diag_ok = r["cumple_criterio"]
    if diag_ok:
        st.success("✅ RESULTADO: DESPEJE ADECUADO")
    else:
        st.error("⚠️ RESULTADO: POSIBLE OBSTRUCCIÓN")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Distancia", f"{r['d_total_km']:.2f} km")
    col2.metric("Longitud de Onda (λ)", f"{r['wavelength']*100:.2f} cm")
    col3.metric("Radio F1 Máximo", f"{r['f1_max']:.2f} m")
    col4.metric("Despeje Mínimo", f"{r['despeje_min_pct']:.1f}%", help="Criterio: ≥ 60%")

    st.markdown("---")
    
    st.subheader("Perfil Topográfico del Enlace")
    fig, ax = plt.subplots(figsize=(10, 4))
    x = r["distancias_km"]
    ax.plot(x, r["linea_vista"], color="#1f77b4", lw=2, label="Línea de vista (LOS)")
    ax.fill_between(x, r["linea_vista"] - r["f1"], r["linea_vista"] + r["f1"], color="#1f77b4", alpha=0.15, label="1ª zona de Fresnel")
    ax.plot(x, r["obstaculo_total"], color="#8B5A2B", lw=2, label="Topografía + Curvatura")
    ax.fill_between(x, 0, r["obstaculo_total"], color="#8B5A2B", alpha=0.5)

    idx_c = r["idx_critico"]
    ax.plot(x[idx_c], r["obstaculo_total"][idx_c], 'ro', ms=8, zorder=5, label=f"Punto crítico ({r['despeje_min_pct']:.0f}% de F1)")
    ax.scatter([x[0]], [r["h1"]], color="black", marker="^", s=100, zorder=5)
    ax.scatter([x[-1]], [r["h2"]], color="black", marker="^", s=100, zorder=5)
    ax.annotate("Torre A", (x[0], r["h1"]), textcoords="offset points", xytext=(5, 8))
    ax.annotate("Torre B", (x[-1], r["h2"]), textcoords="offset points", xytext=(-45, 8))

    ax.set_xlabel("Distancia (km)")
    ax.set_ylabel("Altura (m)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Ruta del Enlace")
    centro = [(r["lat1"] + r["lat2"]) / 2, (r["lon1"] + r["lon2"]) / 2]
    m = folium.Map(location=centro, zoom_start=12, tiles="OpenStreetMap")
    folium.Marker([r["lat1"], r["lon1"]], tooltip=f"Torre A ({r['h1']:.1f} m)", icon=folium.Icon(color="red")).add_to(m)
    folium.Marker([r["lat2"], r["lon2"]], tooltip=f"Torre B ({r['h2']:.1f} m)", icon=folium.Icon(color="blue")).add_to(m)
    folium.PolyLine([[r["lat1"], r["lon1"]], [r["lat2"], r["lon2"]]], color="green", weight=3, tooltip=f"{r['d_total_km']:.2f} km").add_to(m)
    st_folium(m, width=1000, height=400)

    st.markdown("---")
    st.subheader("Exportar Reporte")
    pdf_bytes = generar_pdf(r)
    st.download_button(
        label="📥 Descargar reporte formal en PDF",
        data=pdf_bytes,
        file_name="diagnostico_fresnel_itu.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    # Generar texto del reporte para descarga
    reporte_texto = f"""DIAGNOSTICO DE LA PRIMERA ZONA DE FRESNEL - ITU-R P.526
------------------------------------------------------
Estudiante: Laura Maria Montaño Poveda
Codigo: 20242678048

RESULTADOS DEL ENLACE:
- Distancia total: {r['d_total_km']:.3f} km
- Longitud de onda (lambda): {r['wavelength']*100:.2f} cm
- Radio maximo de F1: {r['f1_max']:.2f} m
- Despeje minimo encontrado: {r['despeje_min_pct']:.1f}%
- Criterio exigido: >= {r['criterio']*100:.0f}% de F1 libre
- Estado del enlace: {"DESPEJE ADECUADO" if r['cumple_criterio'] else "POSIBLE OBSTRUCCION"}
"""

    st.download_button(
        label="📥 Descargar reporte de diagnóstico (.txt)",
        data=reporte_texto,
        file_name="reporte_fresnel.txt",
        mime="text/plain",
        use_container_width=True
    )