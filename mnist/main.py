import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

w = tf.Variable(tf.truncated_normal(shape=[784, 10], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[10]))

z = tf.matmul(x, w) + b
y_ = tf.nn.softmax(z)

# error = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))

error = tf.reduce_sum(tf.squared_difference(y, y_))

acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1)), tf.float32))
opt = tf.train.GradientDescentOptimizer(0.005)
train_step = opt.minimize(error)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for k in range(100000):

        x_batch, y_batch = mnist.train.next_batch(100)

        y_out, step, loss, accuracy = sess.run([y_, train_step, error, acc],
                                               feed_dict={x: x_batch,
                                                          y: y_batch})
        if k % 1000 == 0:
            print("Step", k, "Acc:", accuracy, 'Error:', loss)
            # print(tf.reduce_sum(y_out[1]).eval())

    accuracy = sess.run(acc, feed_dict={x: mnist.test.images,
                                        y: mnist.test.labels})
    print("Final acc:", accuracy)

