import tensorflow as tf
keras = tf.keras
from copy import deepcopy
from euler import fwd_tlist, fwd_sde_tlist


def gradient_update(mdl, slvr, opt, btch_y0, t_points):
    with tf.GradientTape() as g:
        
        y_points = slvr(mdl, btch_y0, t_points, 1e-2)
        loss = tf.reduce_mean(input_tensor=tf.math.square(y_points), axis=0) # Across time
        loss = tf.reduce_mean(input_tensor=loss, axis=0) # Across batches
    
    gradients = g.gradient(loss, mdl.weights)  # tensorflow gradient computation
    opt.apply_gradients(zip(gradients, mdl.weights))
    return loss[0]



def euler_opt(mdl,opt,y0,T,dt):

    t = 0
    y = deepcopy(y0)
    traj =[]

    while t <= T:
        traj.append(deepcopy(y))

        with tf.GradientTape() as g:
            dy = mdl.forward(t, y)
            y += dt * dy
            loss = y[0,1,0]*tf.sign(y[0,0,0])

        gradients = g.gradient(loss, mdl.weights)  # tensorflow gradient computation
        opt.apply_gradients(zip(gradients, mdl.weights))

        t += dt

    return traj

