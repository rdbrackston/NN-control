import tensorflow as tf
keras = tf.keras
from keras import initializers

# oscillator as ODE within Keras
class Oscillator(tf.keras.Model):
    def __init__(self):
        self.A = tf.cast([[0, 1],[-1, -0.1]], tf.float32)

    def forward(self, t, y):
        return tf.matmul(self.A, y)



class OscillatorNNControl(tf.keras.Model):
    """
    Oscillator with controller defined by dense sequential neural network.
    Suitable for use with batch inputs.
    """
    
    def __init__(self):
        super(OscillatorNNControl, self).__init__()
        self.K = keras.Sequential()
        self.K.add(keras.layers.Dense(50, activation="tanh", input_shape=(2,)))
        self.K.add(keras.layers.Dense(1))
        self.A = tf.cast([[0, 1],[-1, -0.1]],tf.float32)
        
    def forward(self, t, y):
        y = tf.reshape(y,[-1,2,1]) # Ensure y has correct dimensions
        con = self.K(tf.compat.v2.squeeze(y,2))
        free = tf.matmul(self.A, y)
        con = tf.concat([tf.zeros(con.shape),con],1)
        con = tf.reshape(con, [-1,2,1]) # Expand to match dimensions
        dy = tf.add(free, con)
        return dy



# Oscillator with controller defined by neural network
class OscillatorLinearControl(tf.keras.Model):
    
    def __init__(self):
        super(OscillatorLinearControl, self).__init__()
        self.linear = keras.layers.Dense(1, input_shape=(2,), kernel_initializer=initializers.random_normal(stddev=0.01))
#         self.linear = keras.layers.Dense(1, input_shape=(2,), kernel_initializer=keras.initializers.Constant(value=[-23,-10]))
        self.A = tf.cast([[0, 1],[-1, -0.1]],tf.float32)
        
    def forward(self, t, y):
        y = tf.reshape(y,[-1,2,1]) # Ensure y has correct dimensions
        con = self.linear(tf.compat.v2.squeeze(y,2))
        free = tf.matmul(self.A, y)
        con = tf.concat([tf.zeros(con.shape),con],1)
        con = tf.reshape(con, [-1,2,1]) # Expand to match dimensions
        dy = tf.add(free, con)
        return dy

