import tensorflow as tf
import numpy as np
from fourier1 import *

class LimeBrain:
    def __init__(self, input_dim, hidden_dim, output_dim, epoch=250, learning_rate=0.001):
        self.epoch=epoch
        self.learning_rate = learning_rate

        input_data=tf.placeholder(dtype=tf.complex64, shape=[None, input_dim])
        output_data=tf.placeholder(dtype=tf.complex64, shape=[None, output_dim])
        with tf.name_scope('encode'):
            weights=tf.Variable(tf.complex(
                    tf.random_normal([input_dim, hidden_dim]),
                    tf.random_normal([input_dim, hidden_dim])),
                dtype=tf.complex64, name='weights')
            biases=tf.Variable(tf.zeros([hidden_dim], dtype=tf.complex64), name='biases')
            encoded = tf.nn.tanh(tf.matmul(input_data, weights) + biases)

        with tf.name_scope('decode'):
            weights=tf.Variable(tf.complex(
                    tf.random_normal([input_dim, hidden_dim]),
                    tf.random_normal([input_dim, hidden_dim])),
                dtype=tf.complex64, name='weights')
            biases=tf.Variable(tf.zeros([output_dim], dtype=tf.complex64), name='biases')
            decoded = tf.matmul(encoded,weights)+biases
        
        self.input_data=input_data
        self.output_data=output_data
        self.encoded=encoded
        self.decoded=decoded

        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.output_data, self.decoded))))
        #self.train_op=tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        self.train_op=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.saver=tf.train.Saver()

    def train(self, in_train_data, out_train_data):
        
        if len(in_train_data)!=len(out_train_data):
            print("Error: datos de entrenamiento de longitudes distintas")
            return None

        num_samples=len(in_train_data)
        with tf.Session as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.epoch):
                for j in range(num_samples):
                    l, _ = sess.run([self.loss, self.train_op],
                        feed_dict={self.input_data: [in_train_data[j]],
                        self.output_data: [out_train_data[j]]})
                    
                if i%10:
                    print('epoch {0}: loss = {1}'.format(i,l))
                    self.saver.save(sess, './model.ckpt')
            self.saver.save(sess, './model.ckpt')
            self.trained=True
    
    def run(self, dataIn, dataOut):
        if not self.trained:
            print('The model has not been trained yet!!')
            return None
        with tf.Session() as sess:
            sess.saver.restore(sess, './model.ckpt')
            return sess.run([self.decoded],
                feed_dict={self.input_data : dataIn})

if __name__ == '__main__':
    lib= 'trainer/samples-44100'
    library = scanLibrary(lib)
    book = library[0]
    origin, sizeo = readWave(book[0], 10)
    dest, sized = readWave(book[1], 10)

    for i in range(len(origin)):
        origin[i]=origin[i].flatten()

    for i in range(len(dest)):
        dest[i]=dest[i].flatten()

    origin=np.array(origin)
    dest=np.array(dest)
    brains = LimeBrain(8820, 8820, 8820)
