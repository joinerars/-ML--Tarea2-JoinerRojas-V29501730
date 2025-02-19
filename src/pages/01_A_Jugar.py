
import cv2
import pandas as pd
import streamlit as st
from keras.datasets import mnist
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import joblib
import numpy as np
#im = Image.open("favicon.ico")

#st.set_page_config(
#    "EsKape Room",
#    im,
#    initial_sidebar_state="expanded",
#    layout="wide",
#)

if "number" not in st.session_state:
    st.session_state["number"] = 0


import os
import joblib



# Cargar el modelo
ruta_modelo = 'C:\\Users\\joine\\OneDrive\\UCV2\\SEMESTRE 2-2024\\ML\\Tareas\\Tarea #2\\[ML]-Tarea2-JoinerRojas-V29501730\\src\\models\\output\\DigitModel.joblib'
modelo_digit = joblib.load(ruta_modelo)


# Cargar el modelo
ruta_modelo2 = 'C:\\Users\\joine\\OneDrive\\UCV2\\SEMESTRE 2-2024\\ML\\Tareas\\Tarea #2\\[ML]-Tarea2-JoinerRojas-V29501730\\src\\models\\output\\OperatorsModel.joblib'
modelo_operator = joblib.load(ruta_modelo2)
 

@st.cache_data
def get_mnist_data():
    return mnist.load_data()

   
# Función para convertir números en imagen
def convertir_a_imagen(numeros):
    imagen = Image.new('L', (28, 28), 255)  # Crear una imagen en blanco (28x28 píxeles)
    draw = ImageDraw.Draw(imagen)
    font = ImageFont.load_default()
    
    for i, num in enumerate(numeros):
        draw.text((5, 5 + i * 10), str(num), font=font, fill=0)
    
    return imagen

# Función para preprocesar la imagen
def preprocesar_imagen(imagen):
    imagen_array = np.array(imagen)
    imagen_array = imagen_array / 255.0  # Normalización
    imagen_array = imagen_array.flatten()  # Aplanar el array
    imagen_array = imagen_array.reshape(1, -1)  # Ajustar forma para el modelo (1, 784)
    return imagen_array


# Función para transformar la imagen
def transform_image_to_mnist(image):
    st.write("Transform > Dimensiones de imagen de entrada")
    st.write(image.shape)
    if image.shape[2] == 4:
        st.write("Transform > Remover canal alpha")
        image = image[:, :, :3]
        st.write(image.shape)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    st.write("Transform > Conversion escala de grises")
    st.write(gray_image.shape)
    
    resized_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_AREA)
    equalized_image = cv2.equalizeHist(resized_image)
    st.write("Transform > Dimensiones imagen de salida")
    st.write(resized_image.shape)
    
    return resized_image, equalized_image

def mnist_dataset_viewer(x_train, y_train, x_test, y_test):
    st.header("Sección de mnist")
    option = st.sidebar.selectbox(
        "De cuál dataset quieres ver la imagen?", ("train", "test")
    )

    if option == "train":
        st.session_state["number"] = st.sidebar.slider(
            "Índice de la imagen en entrenamiento", 0, x_train.shape[0], 0
        )
        st.image(x_train[st.session_state["number"]], channels="gray")
    else:
        st.session_state["number"] = st.sidebar.slider(
            "Índice de la imagen en prueba", 0, x_test.shape[0], 0
        )
        st.image(x_test[st.session_state["number"]], channels="gray")

    st.write("Shape of mnist image")
    st.write(x_train[st.session_state["number"]].shape)

def play_canvas():
    # Cómo leer los datos de Keras
    (x_train, y_train), (x_test, y_test) = get_mnist_data()

    # Creando variables del sidebar
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ","#ffffff")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#000000")
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    with st.container():
        (
            number_one,
            _,
            operator_one,
            number_two,
            _,
            operator_two,
            number_three,
        ) = st.columns([3, 1, 2, 3, 1, 2, 3])

        with number_one:
            c1, c2 = st.columns(2)
            with c1:
                st.empty()
            with c2:
                exponent_1 = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    background_color=bg_color,
                    background_image=None,
                    update_streamlit=realtime_update,
                    height=50,
                    width=50,
                    drawing_mode="freedraw",
                    point_display_radius=0,
                    key="exponent_1",
                )

            number_1 = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                background_image=None,
                update_streamlit=realtime_update,
                height=150,
                width=150,
                drawing_mode="freedraw",
                point_display_radius=0,
                key="number_1",
            )

        with operator_one:
            with st.container():
                st.markdown("#")
                st.markdown("#")
                operator_1 = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    background_color=bg_color,
                    background_image=None,
                    update_streamlit=realtime_update,
                    height=100,
                    width=100,
                    drawing_mode="freedraw",
                    point_display_radius=0,
                    key="operator_1",
                )
        with number_two:
            c1, c2 = st.columns(2)
            with c1:
                st.empty()
            with c2:
                exponent_2 = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    background_color=bg_color,
                    background_image=None,
                    update_streamlit=realtime_update,
                    height=50,
                    width=50,
                    drawing_mode="freedraw",
                    point_display_radius=0,
                    key="exponent_2",
                )
            number_2 = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                background_image=None,
                update_streamlit=realtime_update,
                height=150,
                width=150,
                drawing_mode="freedraw",
                point_display_radius=0,
                key="number_2",
            )

        with operator_two:
            st.markdown("#")
            st.markdown("#")
            operator_2 = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                background_image=None,
                update_streamlit=realtime_update,
                height=100,
                width=100,
                drawing_mode="freedraw",
                point_display_radius=0,
                key="operator_2",
            )

        with number_three:
            c1, c2 = st.columns(2)
            with c1:
                st.empty()
            with c2:
                exponent_3 = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    background_color=bg_color,
                    background_image=None,
                    update_streamlit=realtime_update,
                    height=50,
                    width=50,
                    drawing_mode="freedraw",
                    point_display_radius=0,
                    key="exponent_3",
                )

            number_3 = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                background_image=None,
                update_streamlit=realtime_update,
                height=150,
                width=150,
                drawing_mode="freedraw",
                point_display_radius=0,
                key="number_3",
            )

    number, exponent, operator, mnist = st.tabs(
        ["Número", "Exponente", "Operador", "Mnist"]
    )

    

    with number:
        st.header("Sección de Número")
        # Do something interesting with the image data and paths
        if number_1.image_data is not None:
            st.write("Image: ")
            st.image(number_1.image_data)

            st.write("Dimensiones de la imagen")
            st.write(number_1.image_data.shape)

            st.write("Matriz asociada a la imagen")
            st.write(number_1.image_data[1])

            st.write("Transforming image")

            # Usar la función `transform_image_to_mnist` para transformar la imagen
            image_mnist, image_mnist_eq = transform_image_to_mnist(number_1.image_data)

            # Flatten the 28x28 image into a 1D array of size 784
            image_mnist = image_mnist.reshape((1, 28 * 28)).astype('float32') / 255

            st.write("Image Transformed: ")
            st.image(image_mnist, channels="gray", caption="Grayscale Image")

            st.write("Image Transformed equalized: ")
            st.image(image_mnist_eq, channels="gray", caption="Grayscale Image")

            st.write("Matriz asociada a la imagen transformada")
            st.write(image_mnist)

            # Preprocesar la imagen para el modelo
            image_mnist_flattened = preprocesar_imagen(image_mnist)

            # Añadir mensajes de depuración
            st.write("Valores de entrada para el modelo:", image_mnist_flattened)

            # Hacer la predicción
            prediccion = modelo_operator.predict(image_mnist_flattened)

            st.write("Predicción cruda:", prediccion)

            # Mostrar la predicción en pantalla
            st.write(f"Predicción: {prediccion}")

        if number_1.json_data is not None:
            objects = pd.json_normalize(
                number_1.json_data["objects"]
            )  # need to convert obj to str because PyArrow
            for col in objects.select_dtypes(include=["object"]).columns:
                objects[col] = objects[col].astype("str")
            st.dataframe(objects)
            

    with exponent:
        st.header("Sección de Exponente")
        if exponent_1.image_data is not None:
            st.write("Image: ")
            st.image(exponent_1.image_data)

            st.write("Matriz asociada a la imagen")
            st.write(exponent_1.image_data[1])

            st.write("Dimensiones de la imagen")
            st.write(exponent_1.image_data.shape)

            st.write("Transformando exponente")

            image_mnist_exp, image_mnist_exp_eq = transform_image_to_mnist(
                exponent_1.image_data
            )

            st.write("Exponente transformado: ")
            # Display the image with Streamlit
            st.image(image_mnist_exp, channels="gray", caption="Grayscale Image")

            st.write("Exponente transformado equalized: ")
            # Display the image with Streamlit
            st.image(image_mnist_exp_eq, channels="gray", caption="Grayscale Image")

            st.write("Matriz asociada al exponente transformado")
            st.write(image_mnist_exp)
        
        if exponent_1.json_data is not None:
            objects = pd.json_normalize(
                exponent_1.json_data["objects"]
            )  # need to convert obj to str because PyArrow
            for col in objects.select_dtypes(include=["object"]).columns:
                objects[col] = objects[col].astype("str")
            st.dataframe(objects)

    with operator:
        st.header("Sección de Operador")
        if operator_1.image_data is not None:
            st.write("Operador: ")
            st.image(operator_1.image_data)

            st.write("Matriz asociada al operador")
            st.write(operator_1.image_data[1])

            st.write("Dimensiones del operador")
            st.write(operator_1.image_data.shape)

            st.write("Transformando operador")

            image_mnist_op, image_mnist_op_eq = transform_image_to_mnist(
                operator_1.image_data
            )

            st.write("Exponente transformado: ")
            # Display the image with Streamlit
            st.image(image_mnist_op, channels="gray", caption="Grayscale Image")

            st.write("Exponente transformado: ")
            # Display the image with Streamlit
            st.image(image_mnist_op_eq, channels="gray", caption="Grayscale Image")

            st.write("Matriz asociada al exponente transformado")
            st.write(image_mnist_op)

        if operator_1.json_data is not None:
            objects = pd.json_normalize(
                operator_1.json_data["objects"]
            )  # need to convert obj to str because PyArrow
            for col in objects.select_dtypes(include=["object"]).columns:
                objects[col] = objects[col].astype("str")
            st.dataframe(objects)

    with mnist:
        mnist_dataset_viewer(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test
        )

    

def main():
    play_canvas()

if __name__ == "__main__":
    main()
