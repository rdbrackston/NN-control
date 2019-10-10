import tensorflow as tf
from copy import deepcopy

def fwd(mdl,y0,tf):
    t = 0
    h =0.01
    y = deepcopy(y0)
    traj =[]
    while t <= tf:
        traj.append(deepcopy(y))
#         print(y)
        dy = mdl.forward(0, y)
        t += h
        y += h * dy
    return tf.stack(traj)


def fwd_tlist(mdl,y0,tlist,step):
    t = 0
    h = step
    traj =[]

    # Transform to batch inputs for NN compatibility
    if tf.shape(tf.shape(y0))[0] < 3:
        n = tf.size(y0)
        y = tf.reshape(y0, [-1,n,1])
    else:
        y = deepcopy(y0)
    
    t_ind = 0
    while t < tlist[-1]:
        dy = mdl.forward(t, y)
        t += h
        y = y + h * dy
        
        if t >= tlist[t_ind]:
            traj.append(1*y)
            t_ind += 1
    return tf.stack(traj)


def fwd_sde_tlist(mdl,y0,tlist,step):
    t = 0
    dt = step
    rtdt = tf.sqrt(step)
    traj =[]

    # Transform to batch inputs for NN compatibility
    if tf.shape(tf.shape(y0))[0] < 3:
        n = tf.size(y0)
        y = tf.reshape(y0, [-1,n,1])
    else:
        n = tf.shape(y0)[1]

    # Predefine g for additive noise at final state
    g = tf.zeros([1,n-1,1])
    g = tf.concat([g, tf.cast([[[0.5]]],tf.float32)],1)
    g = tf.tile(g,[tf.shape(y)[0],1,1])
    
    t_ind = 0
    while t < tlist[-1]:
        # f,g = mdl.forward(t, y)
        f = mdl.forward(t,y)
        dW = tf.random.normal(tf.shape(g))*rtdt
        t += dt
        y += f*dt + g*dW
        
        if t >= tlist[t_ind]:
            traj.append(1*y)
            t_ind += 1
    return tf.stack(traj)

