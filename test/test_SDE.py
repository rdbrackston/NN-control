import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "src")

import importlib
from euler import fwd, fwd_tlist, fwd_sde_tlist
importlib.reload(sys.modules['euler'])
import models as md
importlib.reload(sys.modules['models'])
import training as tr
importlib.reload(sys.modules['training'])
from utilities import rand_batch, tile_batch
importlib.reload(sys.modules['utilities'])

t_end = 50
t_grid = np.linspace(0, t_end, t_end/0.01)
y0 = tf.constant([[5., 5.]], tf.float32)
free_trajectory = fwd_sde_tlist(md.Oscillator(), y0, t_grid,0.01)
plt.clf()
plt.plot(t_grid,free_trajectory[0:len(t_grid),0,0])

# Initialise new instances of the model
model = md.OscillatorNNControl()
niters = 5
loss_history = []
optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=1e-5, momentum=0.95)

def train_sde(model, batch_size, niters, test_freq, t_grid, free_trajectory):
    y0 = tf.constant([[[1.], [0.]]],tf.float32)
    optimizer = tf.keras.optimizers.Adamax(learning_rate=1e-2)
    loss_history = []

    for step in range(niters+1):
        batch_y0 = tile_batch(batch_size, y0) # Fixed
        loss = tr.gradient_update(model, fwd_sde_tlist, optimizer, batch_y0, t_grid)
        loss_history.append(loss.numpy())

        if step % test_freq == 0:
            print(loss)

    return loss_history

batch_size = 5
niters = 5
test_freq = 5

loss_history = train_sde(model, batch_size, niters, test_freq, t_grid, free_trajectory)
trajectory = fwd_sde_tlist(model, y0, t_grid,0.01)
plt.clf()
plt.plot(t_grid,free_trajectory[0:len(t_grid),0,0],\
        t_grid,trajectory[0:len(t_grid),0,0,0])
plt.legend( ('free oscillator', 'controlled oscillator'))
plt.savefig("ControlledSDE.pdf", format='pdf')
