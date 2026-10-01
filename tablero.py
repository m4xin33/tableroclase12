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

# Cargar modelo con caché
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

# Configuración en la Barra Lateral (Sidebar)
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
    stroke_width = st.slider("Grosor del pincel:", min_value=5, max_value=50, value=25)
    
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

# Interfaz Principal
st.title("✍️ Detección Personalizada de Dígitos")
st.markdown("Dibuja un número sobre el lienzo y ajusta sus propiedades desde el menú lateral.")

if model is None:
    st.error("❌ No se encontró ningún archivo de modelo (`mnist_model.keras` o `mnist_model.h5`).")
    st.info("""
    **Instrucciones para resolverlo:**
    1. Asegúrate de tener tu modelo entrenado guardado como `mnist_model.keras`.
    2. Súbelo a la raíz de tu repositorio en GitHub.
    3. Reinicia la aplicación en Streamlit.
    """)
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    st.write("### 🎨 Tablero de Dibujo")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="black",
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

# Proceso de Predicción
if predict_btn:
    if canvas_result.image_data is not None:
        # Verificar si hay algo dibujado comprobando si hay píxeles no negros
        rgb_data = canvas_result.image_data[:, :, :3]
        if np.all(rgb_data == 0):
            st.warning("⚠️ El lienzo está vacío. Por favor, dibuja un dígito.")
        else:
            with st.spinner("🤔 Analizando trazo..."):
                # Convertir matriz RGBA a escala de grises
                image = Image.fromarray(
                    canvas_result.image_data.astype("uint8")
                ).convert("L")
                
                # Redimensionar a 28x28 píxeles (formato MNIST)
                image_resized = image.resize((28, 28), Image.Resampling.LANCZOS)
                
                # Normalizar los datos
                img_array = np.array(image_resized) / 255.0
                img_array = img_array.reshape(1, 28, 28, 1)
                
                # Realizar predicción
                prediction = model.predict(img_array, verbose=0)
                digit = int(np.argmax(prediction))
                confidence = float(np.max(prediction) * 100)
            
            # Mostrar resultados
            st.success(f"## 🎯 Dígito detectado: **{digit}**")
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Predicción", digit)
            with col_m2:
                st.metric("Confianza", f"{confidence:.1f}%")
            with col_m3:
                alternative = int(np.argsort(prediction[0])[-2])
                st.metric("2ª Opción", alternative)
            
            # Visualizaciones
            st.write("---")
            col_img1, col_img2 = st.columns(2)
            
            with col_img1:
                st.write("**Entrada en lienzo:**")
                st.image(canvas_result.image_data, width=180)
            
            with col_img2:
                st.write("**Vista previa MNIST (28x28):**")
                st.image(image_resized, width=180)
            
            # Gráfico de probabilidades
            st.write("### 📊 Distribución de Probabilidades")
            prob_df = pd.DataFrame({
                'Dígito': [str(i) for i in range(10)],
                'Probabilidad (%)': prediction[0] * 100
            })
            st.bar_chart(prob_df.set_index('Dígito'))
