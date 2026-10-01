import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os
import pandas as pd

st.set_page_config(
    page_title="Detector de Dígitos MNIST",
    page_icon="✍️",
    layout="wide"
)

# Cargar modelo con caché de Streamlit
@st.cache_resource
def load_mnist_model():
    """Carga el modelo MNIST desde archivo .keras o .h5"""
    model_files = ['mnist_model.keras', 'mnist_model.h5']
    
    for model_path in model_files:
        if os.path.exists(model_path):
            try:
                model = keras.models.load_model(model_path)
                return model, model_path
            except Exception as e:
                st.error(f"❌ Error al cargar {model_path}: {str(e)}")
                continue
    return None, None

model, loaded_path = load_mnist_model()

# --- BARRA LATERAL: Controles Personalizados ---
with st.sidebar:
    st.header("⚙️ Configuración del Lienzo")
    
    # 1. Herramientas de Dibujo
    drawing_mode = st.selectbox(
        "Herramienta:",
        ("freedraw", "line"),
        format_func=lambda x: "✏️ Dibujo Libre" if x == "freedraw" else "📏 Línea Recta"
    )
    
    # 2. Color y Grosor
    stroke_color = st.color_picker("Color del trazo:", "#FFFFFF")
    stroke_width = st.slider("Grosor del pincel:", min_value=10, max_value=50, value=25)
    
    # 3. Tamaño del Tablero
    st.markdown("---")
    st.subheader("📐 Dimensiones del Tablero")
    canvas_width = st.slider("Ancho (px):", min_value=200, max_value=800, value=400, step=50)
    canvas_height = st.slider("Alto (px):", min_value=200, max_value=800, value=400, step=50)
    
    # Información del entorno
    st.markdown("---")
    st.write(f"**TensorFlow:** `{tf.__version__}`")
    if loaded_path:
        st.caption(f"📂 Modelo cargado: `{loaded_path}`")

# --- INTERFAZ PRINCIPAL ---
st.title("✍️ Detección Personalizada de Dígitos")
st.markdown("Dibuja un número sobre el lienzo y presiona **Predecir Dígito**.")

if model is None:
    st.error("❌ No se encontró ningún archivo de modelo (`mnist_model.keras` o `mnist_model.h5`).")
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    st.write("### 🎨 Tablero de Dibujo")
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#000000",
        width=canvas_width,
        height=canvas_height,
        drawing_mode=drawing_mode,
        key="canvas",
    )

with col2:
    st.write("### 🎮 Acciones")
    predict_btn = st.button("🔍 **Predecir Dígito**", type="primary", use_container_width=True)
    clear_btn = st.button("🗑️ Limpiar Lienzo", use_container_width=True)
    
    if clear_btn:
        st.rerun()

# --- PROCESO DE PREDICCIÓN ROBUSTO ---
if predict_btn:
    # Extraer la imagen del canvas
    has_image = False
    img_data = None

    if canvas_result is not None and canvas_result.image_data is not None:
        img_data = canvas_result.image_data
        # Verificar si hay al menos un píxel dibujado (revisar canal Alpha y canales RGB)
        alpha_channel = img_data[:, :, 3]
        rgb_channels = img_data[:, :, :3]
        
        # Hay trazo si el alpha > 0 o si el color RGB no es totalmente negro (0)
        if np.any(alpha_channel > 0) or np.any(rgb_channels > 0):
            has_image = True

    if has_image:
        with st.spinner("🤔 Analizando trazo..."):
            # 1. Convertir la matriz RGBA a imagen PIL
            pil_img = Image.fromarray(img_data.astype("uint8"))
            
            # 2. Convertir a escala de grises
            # Extraer la máscara del trazo para asegurar trazo blanco sobre fondo negro
            gray_img = pil_img.convert("L")
            img_np = np.array(gray_img)

            # Si el trazo no es blanco puro, extraemos la luminosidad del trazo
            if stroke_color.upper() != "#FFFFFF":
                # Convertir canales RGB a máscara de intensidad
                rgb_sum = np.sum(img_data[:, :, :3], axis=2)
                img_np = np.where(rgb_sum > 0, 255, 0).astype(np.uint8)
                pil_img_proc = Image.fromarray(img_np)
            else:
                pil_img_proc = gray_img

            # 3. Redimensionar a 28x28 píxeles (formato exacto MNIST)
            image_resized = pil_img_proc.resize((28, 28), Image.Resampling.LANCZOS)
            
            # 4. Normalizar valores entre 0.0 y 1.0
            img_array = np.array(image_resized) / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)
            
            # 5. Realizar predicción con la red neuronal
            prediction = model.predict(img_array, verbose=0)
            digit = int(np.argmax(prediction))
            confidence = float(np.max(prediction) * 100)
        
        # --- DESPLIEGUE DE RESULTADOS ---
        st.success(f"## 🎯 Dígito detectado: **{digit}**")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Predicción", digit)
        with col_m2:
            st.metric("Confianza", f"{confidence:.1f}%")
        with col_m3:
            alternative = int(np.argsort(prediction[0])[-2])
            st.metric("2ª Opción", alternative)
        
        # Visualización de las imágenes procesadas
        st.write("---")
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.write("**Entrada en lienzo:**")
            st.image(img_data, width=180)
        
        with col_img2:
            st.write("**Vista previa entrada MNIST (28x28):**")
            st.image(image_resized, width=180)
        
        # Gráfico de probabilidades
        st.write("### 📊 Distribución de Probabilidades")
        prob_df = pd.DataFrame({
            'Dígito': [str(i) for i in range(10)],
            'Probabilidad (%)': prediction[0] * 100
        })
        st.bar_chart(prob_df.set_index('Dígito'))
    else:
        st.warning("⚠️ No se detectó ningún dibujo en el lienzo. Dibuja un número antes de presionar Predecir.")
