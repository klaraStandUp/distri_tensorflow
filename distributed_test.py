import tensorflow as tf
import numpy as np

def get_batch(n):
    x = np.random.random(n)
    y = np.exp(x)
    return x,y

def leaky_relu(x,alpha=0.2):
    return tf.maximum(alpha*x,x)

x_ = tf.placeholder(tf.float32, shape=[None, 1])
t_ = tf.placeholder(tf.float32, shape=[None, 1])

with tf.device("/job:ps/task:0"):
    W1 = tf.Variable(tf.zeros([1,16]))
    W2 = tf.Variable(tf.zeros([16,32]))
    W3 = tf.Variable(tf.zeros([32,1]))

with tf.device("/job:ps_/task:0"):
    b1 = tf.Variable(tf.zeros([16]))
    b2 = tf.Variable(tf.zeros([32]))
    b3 = tf.Variable(tf.zeros([1]))

with tf.device("/job:master/task:0"):
    h1 = leaky_relu(tf.matmul(x_,W1)+b1)
    h2 = leaky_relu(tf.matmul(h1,W2)+b2)
    y  = leaky_relu(tf.matmul(h2,W3)+b3)
    e  = tf.nn.l2_loss(y-t_)

opt=tf.train.AdamOptimizer()
train_step=opt.minimize(e)

with tf.Session("grpc://172.17.0.21:2222") as sess:

    sess.run(tf.initialize_all_variables())
    for i in range(10000):
        x0,t0 = get_batch(100)
        x = x0.astype(np.float32).reshape(100,1)
        t = t0.astype(np.float32).reshape(100,1)

        sess.run(train_step,feed_dict={x_: x, t_:t})

        if i%100==0:
            loss = sess.run(e,feed_dict={x_: x, t_:t})
            if loss < 10:
                break
            else:
                print "loss,", loss
