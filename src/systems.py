import tensorflow as tf
keras = tf.keras
from keras import initializers

class OLSystem:
    """
    Parent class for the open loop system.
    """

    def __init__(self, nStates):
        self.nStates = nStates


class VDPOscillator(OLSystem):
    """
    Van-der-Pol oscillator with two states and default nonlinearity param mu=1
    """

    def __init__(self, dfltMu=1.0):
        super().__init__(2)
        self.mu = tf.constant(dfltMu, tf.float32)

    def __call__(self, x, t):
        X = x[0,0,0]
        dX = x[0,1,0]
        dx = [[dX], [self.mu*(1-tf.math.square(X))*dX-X]]
        return tf.stack(dx)


class DenseSeqControl(tf.keras.Model):
    """
    Most basic control architecture consisting of a dense sequential network
    with 50 nodes.
    """

    def __init__(self):
        super(DenseSeqControl, self).__init__()
        self.K = keras.Sequential()
        self.K.add(keras.layers.Dense(50, activation="tanh", input_shape=(2,)))
        self.K.add(keras.layers.Dense(1))

    def __call__(self, x, t):
        out = self.K(tf.squeeze(x,2))
        out = tf.concat([tf.zeros(out.shape),out],1)
        return tf.reshape(out, [-1,2,1]) # Expand to match dimensions
