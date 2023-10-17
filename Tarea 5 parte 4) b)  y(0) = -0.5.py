import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Activation
from tensorflow.keras.optimizers import RMSprop, Adam

from matplotlib import pyplot as plt
import numpy as np

class ODEsolver(Sequential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mse = tf.keras.losses.MeanSquaredError()

    @property
    def metrics(self):
      return [self.loss_tracker]

    def train_step(self, data):
         batch_size = tf.shape(data)[0]
         min = tf.cast(tf.reduce_min(data),tf.float32)
         max = tf.cast(tf.reduce_max(data),tf.float32)
         x = tf.random.uniform((batch_size,1), minval=min, maxval=max)

         with tf.GradientTape() as tape:
             # Loss value
             with tf.GradientTape(persistent=True) as g:
                 g.watch(x)

                 with tf.GradientTape() as gg:
                     gg.watch(x)
                     y_pred = self.call(x, training=True)


                 y_x = gg.gradient(y_pred, x)
             y_xx = g.gradient(y_x, x)
             x_o = tf.zeros((batch_size,1)) #valor de x en condicion inicial x_0=0
             y_o = self(x_o,training=True) #valor del modelo en en x_0
             eq =  y_xx + y_pred  #Ecuacion diferencial evaluada en el modelo. Queremos que sea muy pequeno
             ic = -0.5 #valor que queremos para la condicion inicial o el modelo en x_0
             loss = self.mse(0., eq) + self.mse(y_o,ic)

        # Apply grads
         grads = tape.gradient(loss, self.trainable_variables)
         self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        #update metrics
         self.loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value
         return {"loss": self.loss_tracker.result()}


model = ODEsolver()


model.add(Dense(150,activation='tanh', input_shape=(1,)))
model.add(Dense(10,activation='tanh'))
model.add(Dense(10,activation='tanh'))
model.add(Dense(1))


model.summary()

model.compile(optimizer=RMSprop(),metrics=['loss'])

x=tf.linspace(-5,5,1000)
history = model.fit(x,epochs=600,verbose=0)


x_testv = tf.linspace(-5,5,1000)
a=model.predict(x_testv)
plt.plot(x_testv,a, color='r', label="aprox")
plt.plot(x_testv,-0.5*np.cos(x),color='skyblue', label="exact")
plt.legend()
plt.show()