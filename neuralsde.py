from typing import Optional#, List
import numpy as np
import numpy.random as npr
import tensorflow as tf
import tensorflow.contrib.eager as tfe

keras = tf.keras

"""
Milstein update rule for an SDE
"""
def milstein(func, dt, state):
    t,y = state
    a,b = func(state)
    rtdt = tf.cast(tf.sqrt(dt),tf.float32)
    tmp = y + a*dt + b*rtdt # Intermediate state estimate
    _,b2 = func((t,tmp))
    dW = tf.random.normal(tf.shape(b))*rtdt

    return (t+dt, y + a*dt + b*dW + 0.5*(b2-b)*(tf.pow(dW,2)-dt)/rtdt)


"""
Class
"""
class NeuralSDE:
    def __init__(
            self, model: tf.keras.Model, t=np.linspace(0, 1, 40),
            solver=milstein):
        self._t = t
        self._model = model
        self._solver = solver
        self._deltas_t = t[1:] - t[:-1]

    def forward(self, inputs: tf.Tensor, return_states: Optional[str] = None):

        def _forward_dynamics(_state):
            """Used in solver _state == (time, tensor)"""
            return self._model(inputs=_state)

        states = []

        def _append_state(_state):
            tensors = _state[1]
            if return_states == "numpy":
                states.append(tensors.numpy())
            elif return_states == "tf":
                states.append(tensors)

        with tf.name_scope("forward"):
            t0 = tf.cast(self._t[0],tf.float32)
            state = [t0, inputs]
            _append_state(state)
            for dt in self._deltas_t:
                state = self._solver(
                    func=_forward_dynamics, dt=tf.cast(dt,tf.float32), state=state)
                _append_state(state)

        outputs = state[1]
        if return_states:
            return outputs, states
        return outputs

"""
Function for gradient computation and parameter update
"""
def compute_gradients_and_update_path(batch_y0, de_model, optimizer, verbose=False):

    with tf.GradientTape() as g:
        
        # print("Running forward evaluation") if verbose else print()
        pred_y, y_points = de_model.forward(batch_y0, return_states="tf") # solve ODE forward       
        pred_path = tf.stack(y_points)
        loss = tf.reduce_mean(tf.math.square(pred_path), axis=1)
        loss = tf.reduce_mean(loss, axis=0)
        
        # print("Evaluating gradients") if verbose else print()
        gradients = g.gradient(loss, de_model._model.weights)  # tensorflow gradient computation

        # print("Applying to optimization") if verbose else print()
        optimizer.apply_gradients(zip(gradients, de_model._model.weights))
        return loss[0]

# Compile into TensorFlow graph to improve speed (Makes it much slower?)
compute_gradients_and_update_path = tfe.defun(compute_gradients_and_update_path)

