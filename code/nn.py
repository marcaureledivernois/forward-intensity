import tensorflow as tf
import numpy as np
import os


raw_data = np.asarray([[0.3,0.1,0.9,1.2,1],[0.1,0.3,0.5,0.3,0.9]])
raw_data.shape
input_dim = 5
hidden_dim = 3
deltaT = tf.constant(3/12, name='deltaT')
learning_rate=0.1

x_raw = tf.placeholder(tf.float32, shape= (None,input_dim) ,name="x_raw")

w1 = tf.Variable(tf.random_normal([input_dim, hidden_dim],dtype=tf.float32), name='weights_1')
b1 = tf.Variable(tf.zeros([hidden_dim]), name='biases_1')
out1 = tf.nn.sigmoid(tf.matmul(x_raw, w1) + b1, name='output_1')

finalW = tf.Variable(tf.truncated_normal([hidden_dim,1]))
f=tf.matmul(out1,finalW)


loss = tf.math.exp(tf.multiply(-f,deltaT))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf.negative(loss))
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)

for e in range(20):
    for i in range(len(raw_data)):
        sess.run([train_op], feed_dict={x_raw: [raw_data[i]]})
    print('-----',e,'-----')
    loss_value = sess.run(loss, feed_dict={x_raw: raw_data})
    print('loss:', np.mean(loss_value))


#w_val = sess.run(w1)
#b1_val = sess.run(b1)
#f_val = sess.run(f)

sess.close()



def get_batch(X, size):
    a = np.random.choice(len(X), size, replace=False)
    return X[a]

class Autoencoder:
    def __init__(self, input_dim, hidden_dim, epoch=250,
                    learning_rate=0.001):
        self.epoch = epoch
        self.learning_rate = learning_rate

        x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim])

        with tf.name_scope('encode'):
            weights = tf.Variable(tf.random_normal([input_dim, hidden_dim],
                                    dtype=tf.float32), name='weights')
            biases = tf.Variable(tf.zeros([hidden_dim]), name='biases')
            encoded = tf.nn.tanh(tf.matmul(x, weights) + biases)
        with tf.name_scope('decode'):
            weights = tf.Variable(tf.random_normal([hidden_dim, input_dim],
                                    dtype=tf.float32), name='weights')
            biases = tf.Variable(tf.zeros([input_dim]), name='biases')
            decoded = tf.matmul(encoded, weights) + biases

        self.x = x
        self.encoded = encoded
        self.decoded = decoded

        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.x,
                            self.decoded))))
        self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()

    def train(self, data, batch_size=10):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.epoch):
                for j in range(500):
                    batch_data = get_batch(data, batch_size)
                    l, _ = sess.run([self.loss, self.train_op], feed_dict={self.x: batch_data})
                if i % 10 == 0:
                    print('epoch {0}: loss = {1}'.format(i, l))
                    self.saver.save(sess, './model.ckpt')
                self.saver.save(sess, './model.ckpt')


    # no batch training
    # def train(self, data):
    #     num_samples = len(data)
    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         for i in range(self.epoch):
    #             for j in range(num_samples):
    #                 l, _ = sess.run([self.loss, self.train_op],
    #                             feed_dict={self.x: [data[j]]})
    #             if i % 10 == 0:
    #                 print('epoch {0}: loss = {1}'.format(i, l))
    #             self.saver.save(sess, './model.ckpt')
    #         self.saver.save(sess, './model.ckpt')

    def test(self, data):
        with tf.Session() as sess:
            self.saver.restore(sess, './model.ckpt')
            hidden, reconstructed = sess.run([self.encoded, self.decoded],
                                     feed_dict={self.x: data})
        print('input', data)
        print('compressed', hidden)
        print('reconstructed', reconstructed)
        return reconstructed

