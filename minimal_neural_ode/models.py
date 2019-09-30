import tensorflow as tf
keras = tf.keras

# oscillator as ODE within Keras
class oscillator(tf.keras.Model):
    def __init__(self):
        self.A = tf.cast([[0, 1],[-1, -0.1]], tf.float32)

    def forward(self, t, y):
        return tf.matmul(self.A, y)


"""
Oscillator with controller defined by dense sequential neural network.
Suitable for use with batch inputs.
"""
class oscillator_NNcontrol(tf.keras.Model):
    
    def __init__(self):
        super(oscillator_NNcontrol, self).__init__()
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
class oscillator_linear_control(tf.keras.Model):
    
    def __init__(self):
        super(oscillator_linear_control, self).__init__()
        self.linear = keras.layers.Dense(1, input_shape=(2,))
        self.A = tf.cast([[0, 1],[-1, -0.1]],tf.float32)
        
    def forward(self, t, y):
        y = tf.reshape(y,[-1,2,1]) # Ensure y has correct dimensions
        con = self.linear(tf.compat.v2.squeeze(y,2))
        free = tf.matmul(self.A, y)
        con = tf.concat([tf.zeros(con.shape),con],1)
        con = tf.reshape(con, [-1,2,1]) # Expand to match dimensions
        dy = tf.add(free, con)
        return dy