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
    y = 1*y0
    traj =[]
    
    t_ind = 0
    while t < tlist[-1]:
        dy = mdl.forward(0, y)
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
    y = 1*y0
    traj =[]

    # Predefine g for additive noise
    g = tf.cast([[0.0], [0.5]], tf.float32)
    if tf.shape(tf.shape(y))[0] > 2: # Cater for batch inputs
        g = tf.reshape(g,[-1,2,1])
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