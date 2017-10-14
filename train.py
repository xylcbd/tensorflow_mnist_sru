#coding: utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sru
import time
import os
import sys
import numpy as np

######################################
#tools
class Tools(object):
    @staticmethod
    def currect_time():
        return time.strftime("%H:%M:%S", time.localtime()) + '.%03d' % (time.time() % 1 * 1000)

    @staticmethod
    def log_print(content):
        print("[" + Tools.currect_time() + "] " + content)

######################################
#setting
class Setting(object):
    @staticmethod
    def checkpoint_dir():
        return "model"

######################################
#network
class Network(object):
    def __init__(self):
        # Network Parameters
        self.num_input = 28 # MNIST data input (img shape: 28*28)
        self.timesteps = 28 # timesteps
        self.num_hidden = 128 # hidden layer num of features
        self.num_classes = 10 # MNIST total classes (0-9 digits)
        self.lstm_layers = 2
        self.using_sru = sys.argv[1] == "SRU"
	print("Using SRU" if self.using_sru else "Using LSTM")

        # tf Graph input
        self.X = tf.placeholder("float", [None, self.timesteps, self.num_input])
        self.Y = tf.placeholder("float", [None, self.num_classes])

        # Define weights
        weights = {
            'out': tf.Variable(tf.random_normal([self.num_hidden, self.num_classes]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([self.num_classes]))
        }

        x = tf.unstack(self.X, self.timesteps, 1)

        # Define a lstm cell with tensorflow    
        if self.using_sru:
            rnn_cell = lambda: sru.SRUCell(self.num_hidden, False)
        else:
            rnn_cell = lambda: tf.nn.rnn_cell.LSTMCell(self.num_hidden, forget_bias=1.0)    

        cell_stack = tf.nn.rnn_cell.MultiRNNCell([rnn_cell() for _ in range(self.lstm_layers)], state_is_tuple=True)

        # Get lstm cell output
        outputs, _ = tf.nn.static_rnn(cell_stack, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        self.logits = tf.matmul(outputs[-1], weights['out']) + biases['out']

        self.prediction = tf.nn.softmax(self.logits)
    
    def get_input_ops(self):
        return self.X, self.Y

    def get_input_shape(self):
        return self.timesteps, self.num_input

    def get_output_ops(self):
        return self.logits, self.prediction

#####################################
#main route
def save_model(saver, sess, model_path):    
    Tools.log_print('save model to {0}.'.format(model_path))
    saver.save(sess, model_path)

def load_model(saver, sess, model_path):
    Tools.log_print('try to load model from {0}.'.format(model_path))    
    saver.restore(sess, model_path)
    Tools.log_print('load model success')
    return True

def train():
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    model_dir = 'model'
    model_path = os.path.join(model_dir, 'mnist_nn')

    network = Network()
    X, Y = network.get_input_ops()    
    timesteps, num_input = network.get_input_shape()
    logits, prediction = network.get_output_ops()

    # Training Parameters
    learning_rate = 0.01
    display_step = 100
    train_epochs = 3
    train_batchsize = 128    
    test_batchsize = 128

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits= logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #train_op = optimizer.minimize(loss_op)
    grads = optimizer.compute_gradients(loss_op)
    for i, (g, v) in enumerate(grads):
        if g is not None:
            grads[i] = (tf.clip_by_norm(g, 5), v)  # clip gradients
    train_op = optimizer.apply_gradients(grads)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        saver = tf.train.Saver(tf.global_variables())
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        for epoch in range(1, train_epochs+1):
            train_steps = len(mnist.train.labels) / train_batchsize   
            for step in range(1, train_steps+1):
                batch_x, batch_y = mnist.train.next_batch(train_batchsize)
                batch_x = batch_x.reshape((train_batchsize, timesteps, num_input))
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                if step % display_step == 0 or step == 1:
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                    Tools.log_print("Epoch[%d/%d] Step[%d/%d] Train Minibatch Loss= %.4f, Training Accuracy= %.4f" % (epoch, train_epochs, step, train_steps, loss, acc))

        Tools.log_print("Optimization Finished!")
        
        save_model(saver, sess, model_path)

        if load_model(saver, sess, model_path):                     
            test_steps = len(mnist.test.labels) / test_batchsize   
            acc_list = []
            for step in range(1, test_steps+1):
                batch_x, batch_y = mnist.test.next_batch(test_batchsize)
                batch_x = batch_x.reshape((test_batchsize, timesteps, num_input))
                batch_acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
                acc_list.append(batch_acc)
            acc = np.mean(acc_list)
            Tools.log_print("Testing Accuracy: {0}".format(acc))


if __name__ == '__main__':
    train()
