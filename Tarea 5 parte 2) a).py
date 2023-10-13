import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from matplotlib import pyplot as plt
import numpy as np
import math


loss_tracker = keras.metrics.Mean(name="loss")
class Funsol(Sequential):
    @property
    def metrics(self):
        return [loss_tracker]

    def train_step(self, data):
        batch_size =100
        x = tf.random.uniform((batch_size,1), minval=-1, maxval=1)
        f = 3.*tf.math.sin(np.pi * x)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = tf.math.reduce_mean(tf.math.square(y_pred-f))

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        loss_tracker.update_state(loss)

        return {"loss": loss_tracker.result()}



model = Funsol()

model.add(Dense(200,activation='tanh', input_shape=(1,)))
model.add(Dense(200,activation='tanh'))
model.add(Dense(200,activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), metrics=['loss'])

x=tf.linspace(-1.5,1.5,100) #x va de -1.5 a 1.5 para poder observar de mejor manera la diferencia entre la aproximacion y el valor exacto en la grafica

history = model.fit(x,epochs=1000,verbose=0)


a=model.predict(x)

plt.plot(x,a,color='r',label="aprox")
plt.plot(x, 3.*tf.math.sin(np.pi * x), color='skyblue',label="exact")
plt.legend()
plt.show()