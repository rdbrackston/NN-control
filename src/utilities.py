import tensorflow as tf

def rand_batch(batch_size, n_states=2, val=5.):
	"""
	Return a batch of randomly generated initial conditions.
	"""
    return tf.random.uniform([batch_size,n_states,1], minval=-val,maxval=val)


def tile_batch(batch_size, vals):
	"""
	Return a batch of identical initial conditions.
	"""
	vals = tf.reshape(vals, [1,-1,1])
    return tf.tile(vals, [batch_size,1,1])