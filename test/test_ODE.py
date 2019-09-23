import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.eager as tfe

keras = tf.keras
tf.compat.v1.enable_eager_execution()

import sys
sys.path.insert(0, "..")
from neuralode_mod import NeuralODE
from neuralsde import compute_gradients_and_update_path

# oscillator as ODE
class ode_oscillator(tf.keras.Model):

    def __init__(self):
        super(ode_oscillator, self).__init__()
        self.A = tf.cast([[0, 1],[-1, -0.1]], tf.float32)

    def call(self, inputs, **kwargs):
        t, y = inputs
        return tf.matmul(y, self.A)

time_steps = 1000
t_grid = np.linspace(0, 100, time_steps)
y0 = tf.cast([[5., 5.]], tf.float32)

model = NeuralODE(ode_oscillator(), t=t_grid)
yN, free_trajectory = model.forward(y0, return_states="numpy")
free_trajectory = np.concatenate(free_trajectory)

plt.plot(t_grid,free_trajectory[:,0])
plt.savefig("FreeODE.pdf", format='pdf')

# Oscillator with controller defined by neural network
class ode_oscillator_NNcontrol(tf.keras.Model):
    
    def __init__(self):
        super(ode_oscillator_NNcontrol, self).__init__()
        self.linear1 = keras.layers.Dense(50, activation="tanh")
        self.linear2 = keras.layers.Dense(1)
        self.A = tf.cast([[0, 1],[-1, -0.1]],tf.float32)
        
    def call(self, inputs, **kwargs):
        t, y = inputs
        con = self.linear1(y)
        con = self.linear2(con)
        dy = tf.add(tf.matmul(y, self.A), tf.concat([tf.cast([[0.0]],tf.float32),con],1))
        return dy

# Initialise new instances of the model
model = ode_oscillator_NNcontrol()
neural_ode = NeuralODE(model, t=t_grid)
neural_ode_test = NeuralODE(model, t=t_grid)

niters = 5
loss_history = []
optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=1e-5, momentum=0.95)

# Run simulation niters times, updating parameters on each iteration
for step in range(niters+1):
    
    print("Running iteration %i" %(step))
    batch_y0 = y0    # Fixed initial condition
    # batch_y0 = tf.random_uniform([1,2], minval=-15.,maxval=15.) # Random
    
    loss = compute_gradients_and_update_path(batch_y0, neural_ode, optimizer, True)
    loss_history.append(loss.numpy())
    
    if step % 1 == 0:
        print(loss)


yN, trajectory = neural_ode_test.forward(y0, return_states="numpy")
trajectory = np.concatenate(trajectory)
plt.clf()
plt.plot(t_grid, free_trajectory[:,0], t_grid, trajectory[:,0])
plt.legend( ('free oscillator', 'controlled oscillator/system'))
plt.savefig("ControlledODE.pdf", format='pdf')
