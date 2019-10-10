import tensorflow as tf

def rand_batch(batch_size, n_states=2, val=5.):
    return tf.random.uniform([batch_size,n_states,1], minval=-val,maxval=val)


def tile_batch(batch_size, vals):
    return tf.tile(vals, [batch_size,1,1])