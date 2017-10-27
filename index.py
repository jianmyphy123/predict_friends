import tensorflow as tf
import numpy as np

xy = np.loadtxt('Facebook Friends.csv', delimiter=',', dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]



X = tf.placeholder(tf.float32, shape=[None,20])
Y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.get_variable('W', shape=[20,1], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis-Y))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:x_data, Y: y_data})
    if step % 100 == 0:
        # print(step, " Cost: ", cost_val, "\nPrediction:\n", hy_val)
        print(step, " Cost: ", cost_val)

print(hy_val)
