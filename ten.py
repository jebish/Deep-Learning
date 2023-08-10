import numpy as np
import tensorflow as tf

w=tf.Variable(0,dtype=tf.float16)

Adam=tf.keras.optimizers.Adam(learning_rate=0.05)

def one_step():
    with tf.GradientTape() as tape:

        cost=w**2-4*w+4
    
        grads=tape.gradient(cost,[w])
        Adam.apply_gradients(zip(grads,[w]))

for i in range(70):
    one_step()

print(w)