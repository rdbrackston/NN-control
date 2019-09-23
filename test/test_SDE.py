import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.eager as tfe

keras = tf.keras
tf.compat.v1.enable_eager_execution()

import sys
sys.path.insert(0, "..")
import importlib
from neuralsde import NeuralSDE, compute_gradients_and_update_path
importlib.reload(sys.modules['neuralsde'])

# oscillator as SDE with white noise disturbance
class sde_oscillator(tf.keras.Model):

    def __init__(self):
        super(sde_oscillator, self).__init__()
        self.A = tf.cast([[0, 1],[-1, -0.1]], tf.float32)

    def call(self, inputs, **kwargs):
        t, y = inputs
        return (tf.matmul(y, self.A), tf.cast([0.0,0.5],tf.float32))

time_steps = 5000
t_grid = np.linspace(0, 200, time_steps)
y0 = tf.cast([[0., 0.]], tf.float32)

model = NeuralSDE(sde_oscillator(), t=t_grid)
yN, free_trajectory = model.forward(y0, return_states="numpy")
free_trajectory = np.concatenate(free_trajectory)

plt.clf()
plt.plot(t_grid,free_trajectory[:,0])
# plt.savefig("FreeSDE.pdf", format='pdf')

# Oscillator with controller defined by neural network
class sde_oscillator_NNcontrol(tf.keras.Model):
    
    def __init__(self):
        super(sde_oscillator_NNcontrol, self).__init__()
        self.linear1 = keras.layers.Dense(50, activation="tanh")
        self.linear2 = keras.layers.Dense(1)
        self.A = tf.cast([[0, 1],[-1, -0.1]],tf.float32)
        
    def call(self, inputs, **kwargs):
        t, y = inputs
        con = self.linear1(y)
        con = self.linear2(con)
        dy = tf.add(tf.matmul(y, self.A), tf.concat([tf.cast([[0.0]],tf.float32),con],1))
        return (dy, tf.cast([0.0,0.5],tf.float32))


# Initialise new instances of the model
model = sde_oscillator_NNcontrol()
neural_sde = NeuralSDE(model, t=t_grid)
neural_sde_test = NeuralSDE(model, t=t_grid)

niters = 5
loss_history = []
optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=1e-5, momentum=0.95)

# Run simulation niters times, updating parameters on each iteration
for step in range(niters+1):
    
    print("Running iteration %i" %(step))
    batch_y0 = y0    # Fixed initial condition
    
    loss = compute_gradients_and_update_path(batch_y0, neural_sde, optimizer, True)
    loss_history.append(loss.numpy())
    
    if step % 1 == 0:
        print(loss)

   
yN, trajectory = neural_sde_test.forward(y0, return_states="numpy")
trajectory = np.concatenate(trajectory)
plt.clf()
plt.plot(t_grid, free_trajectory[:,0], t_grid, trajectory[:,0])
plt.legend( ('free oscillator', 'controlled oscillator/system'))
plt.savefig("ControlledSDE.pdf", format='pdf')


