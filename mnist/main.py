import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# placeholders:
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

# variables (parameters)
w = tf.Variable(tf.truncated_normal(shape=[784, 10], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[10]))

# model:
z = tf.matmul(x, w) + b
y_ = tf.nn.softmax(z)

error = tf.reduce_sum(-(y*tf.log(y_ + 0.00001)))

prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1))
acc = tf.reduce_mean(tf.cast(prediction, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
train_step = optimizer.minimize(error)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for k in range(10000):

        x_batch, y_batch = mnist.train.next_batch(100)

        step, e = sess.run([train_step, error], feed_dict={x: x_batch, y: y_batch})

        if k % 100 == 0:
            print("Step", k, 'Error:', e)

    accuracy = sess.run(acc, feed_dict={x: mnist.test.images,
                                        y: mnist.test.labels})
    print("Test acc:", accuracy)









