import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

class CapaPolinomio(Layer):
    def __init__(self, **kwargs):
        super(CapaPolinomio, self).__init__(**kwargs)
        self.a_0 = self.add_weight(name="a_0", shape=(1,), initializer="random_normal", trainable=True)
        self.a_1 = self.add_weight(name="a_1", shape=(1,), initializer="random_normal", trainable=True)
        self.a_2 = self.add_weight(name="a_2", shape=(1,), initializer="random_normal", trainable=True)
        self.a_3 = self.add_weight(name="a_3", shape=(1,), initializer="random_normal", trainable=True)

    def call(self, x):
        return self.a_0 + self.a_1 * x + self.a_2 * x**2 + self.a_3 * x**3


model = keras.Sequential([
    CapaPolinomio(input_shape=(1,)),
])

model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

x = np.linspace(-1, 1, 100)
y_exact = np.cos(2 * x)

history = model.fit(x, y_exact, epochs=100, verbose=0)

a = model.predict(x)


###############################################################
coeficientes = model.get_layer("capa_polinomio").get_weights()
a_0, a_1, a_2, a_3 = coeficientes
print("a_0:", a_0)
print("a_1:", a_1)
print("a_2:", a_2)
print("a_3:", a_3)
print(a_0 + a_1 + a_2 + a_3)
#Esta parte del codigo se escribio para observar de forma mas clara el valor de los coeficientes y el valor del polinomio en x=1




plt.plot(x, a, color='r', label="aprox")
plt.plot(x, y_exact, color='skyblue', label="exact")
plt.legend()
plt.show()


