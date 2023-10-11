import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
import numpy as np

# Diseñamos la capa que transforme las imagenes a color a escala de grises:
class ImagenAGrises(Layer):
    def __init__(self, **kwargs):
        super(ImagenAGrises, self).__init__(**kwargs)

    def call(self, inputs):
        # Convierte la imagen a escala de grises
        imagengrises = tf.image.rgb_to_grayscale(inputs)
        return imagengrises


model = Sequential()
model.add(Input(shape=(None, None, 3)) )
model.add(ImagenAGrises())


model.compile(optimizer='adam', loss='mse')

# Creamos una imagen de prueba a color:
imagen_a_color = np.random.random((1, 512, 512, 3))  

# La guardamos para verificar que efectivamente esta a color:
tf.keras.preprocessing.image.save_img("imagen_a_color.jpg", imagen_a_color[0])

# Pasamos la imagen a traves del modelo utilizando model.predict:
imagen_esc_grises = model.predict(imagen_a_color)


# Guardamos la imagen ya transformada para verificar que efectivamente esta en escala de grises
tf.keras.preprocessing.image.save_img("imagen_esc_grises.jpg", imagen_esc_grises[0])




