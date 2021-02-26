import tensorflow as tf
import numpy as np

import tensorflow as tf
import numpy as np
import os


class LinearAggregator:
    def __init__(self, input_size, nb_pred_standing, name='unnamed_model_low_rate', summary_type="", rate_of_saving=20, verbose=True,start_rate=0.01):


        self.verbose = verbose
        self.rate_of_saving = rate_of_saving
        self.name = name
        self.input_size = input_size
        self.nb_pred_standing = nb_pred_standing
        self.last_saved_epoch = -1

        with tf.name_scope('input_x'):
            self.values = tf.placeholder(tf.float32, [None, self.nb_pred_standing], name='values')

            self.x_raw = tf.placeholder(tf.float32, [None, nb_pred_standing, input_size], name='x_raw')
            self.x_list = tf.convert_to_tensor(tf.split(self.x_raw, num_or_size_splits=self.nb_pred_standing, axis=1, name='x_list'))

        with tf.name_scope('input_y'):
            self.actual = tf.placeholder(tf.float32, [None], name='actual')

        with tf.name_scope('aggregation'):
            # self.W = tf.Variable(tf.truncated_normal([1, input_size]), name='weights')
            self.W = tf.Variable(tf.ones([1, input_size]), name='weights')
            # w = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0],dtype=np.float32)
            # self.W = tf.Variable(w.reshape((1,-1)),name='weights')

            # tf.map_fn(fn=lambda x: tf.multiply(x, self.W), elems=self.x_list, name='omega')
            # tf.map_fn(fn=lambda x: tf.reduce_sum(tf.multiply(x, self.W), axis=(1,2)), elems=self.x_list, name='omega')
            self.omega = tf.abs(
                tf.transpose(tf.map_fn(fn=lambda x: tf.reduce_sum(tf.multiply(x, self.W), axis=(1, 2)), elems=self.x_list, name='omega'))
            )

            self.valuesTimeWeights = tf.reduce_sum(tf.multiply(self.values, self.omega), axis=1, name='value_time_weight')
            self.pred = tf.div(self.valuesTimeWeights, tf.reduce_sum(self.omega, axis=1), name='pred')

        with tf.name_scope('cost_and_opt'):
            self.cost = tf.reduce_sum(tf.square(self.pred - self.actual), name='cost')
            self.cost_abs = tf.reduce_sum(tf.abs(self.pred - self.actual), name='cost_abs')
            # self.train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.cost)
            self.train_op = tf.train.AdamOptimizer(start_rate).minimize(self.cost)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        ## writer
        self.abs_error = 0

        tf.summary.scalar('cost', self.cost)
        # tf.summary.histogram('weights', self.W)
        tf.summary.scalar('abs_cost', self.cost_abs)
        # tf.summary.histogram('pred', self.pred)
        self.merged = tf.summary.merge_all()  # merge the summary, make it easier to go all at once
        if summary_type =="":
            dir = "./logs/" + name
        else:
            dir = "./logs/"+summary_type+"/" + name
        if not os.path.exists(dir):
            os.makedirs(dir)

        self.writer = tf.summary.FileWriter(dir + "/")  # Passes in the logs directory's location to the writer
        self.writer.add_graph(self.sess.graph)

    def initialise(self):
        self.sess.run(tf.global_variables_initializer())

    def load(self, epoch_to_load=-1):
        if epoch_to_load == -1:
            # print('loading ', self.last_saved_epoch)
            l = os.listdir('tf_models')
            ll = []
            for x in l:
                if self.name in x:
                    ll.append(x)
            e = max([int(x.split('.')[0].split(self.name)[1]) for x in ll])

            self.saver.restore(self.sess, 'tf_models/' + self.name + str(e) + '.ckpt')
        else:
            # print('loading ', epoch_to_load)
            self.saver.restore(self.sess, 'tf_models/' + self.name + str(epoch_to_load) + '.ckpt')

    def save_model(self, epoch):
        save_path = self.saver.save(self.sess, 'tf_models/' + self.name + str(epoch) + '.ckpt')
        # print('updated to ', self.last_saved_epoch)
        # print('Model saved to {}'.format(save_path))

    def train_sample(self, train_x, train_values, train_actual, force_save=False, epoch=0):
        """
        :param train_x: shape (sample_size, nb_pred, number_feature)
        :param train_values: shape (sample_size, nb_pred)
        :param train_actual: shape (sample_size, )
        :param force_save: bool
        :param epoch: unique id
        """

        for r in range(self.rate_of_saving):
            _, mse, pred = self.sess.run([self.train_op, self.cost, self.pred],
                                         feed_dict={self.x_raw: train_x, self.values: train_values, self.actual: train_actual})

        if force_save:
            # saving either if saved is forth or if we are in the rate of saving
            save_path = self.saver.save(self.sess, 'tf_models/' + self.name + str(epoch) + '.ckpt')
            self.last_saved_epoch = epoch
            print('updated to ', self.last_saved_epoch)
            print('Model saved to {}'.format(save_path))

    def pred_on_sample_with_summary(self, test_x, test_values, test_actual, epoch):
        """
        :param test_x: shape (sample_size, nb_pred, number_feature)
        :param test_values: shape (sample_size, nb_pred)
        :param test_actual: shape (sample_size, )
        :param epoch:
        :return:
        """
        pred, summary, cost, abs_cost = self.sess.run([self.pred, self.merged, self.cost,self.cost_abs], feed_dict={self.x_raw: test_x, self.values: test_values, self.actual: test_actual})
        # pred= self.sess.run([self.pred], feed_dict={self.x_raw: test_x, self.values: test_values})
        self.writer.add_summary(summary, epoch)

        return pred, cost, abs_cost

    def pred_simple(self, test_x, test_values, test_actual):
        """
        :param test_x: shape (sample_size, nb_pred, number_feature)
        :param test_values: shape (sample_size, nb_pred)
        :param test_actual: shape (sample_size, )
        :param epoch:
        :return:
        """
        pred = self.sess.run(self.pred, feed_dict={self.x_raw: test_x, self.values: test_values, self.actual: test_actual})

        return pred

    def getParams(self):
        # tf.get_variable_scope().reuse_variables()
        W = self.W.eval(self.sess)
        return W


class OneLayerAggregator(LinearAggregator):
    def __init__(self, input_size, nb_pred_standing, name='unnamed_model_low_rate', summary_type ="", rate_of_saving=20, verbose=True,start_rate=0.01):
        # super().__init__(input_size, nb_pred_standing, name, rate_of_saving, verbose)
        self.verbose = verbose
        self.rate_of_saving = rate_of_saving
        self.name = name
        self.input_size = input_size
        self.nb_pred_standing = nb_pred_standing
        self.last_saved_epoch = -1

        with tf.name_scope('input_x'):
            self.values = tf.placeholder(tf.float32, [None, self.nb_pred_standing], name='values')

            self.x_raw = tf.placeholder(tf.float32, [None, nb_pred_standing, input_size], name='x_raw')
            self.x_list = tf.convert_to_tensor(tf.split(self.x_raw, num_or_size_splits=self.nb_pred_standing, axis=1, name='x_list'))

        with tf.name_scope('input_y'):
            self.actual = tf.placeholder(tf.float32, [None], name='actual')

        with tf.name_scope('aggregation'):
            # self.W = tf.Variable(tf.truncated_normal([1, input_size]), name='weights')
            self.W = tf.Variable(tf.ones([1, input_size]), name='weights')
            # w = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0],dtype=np.float32)
            # self.W = tf.Variable(w.reshape((1,-1)),name='weights')

            # tf.map_fn(fn=lambda x: tf.multiply(x, self.W), elems=self.x_list, name='omega')
            # tf.map_fn(fn=lambda x: tf.reduce_sum(tf.multiply(x, self.W), axis=(1,2)), elems=self.x_list, name='omega')
            layers_1_width = 50
            lOneWeights = tf.Variable(tf.truncated_normal([input_size, layers_1_width]), name='layer_one_weights')
            lOneConstant = tf.Variable(tf.constant(0.1, shape=[layers_1_width]), name='layer_one_constant')
            lOneOutput = tf.map_fn(
                fn=lambda x:
                tf.nn.relu(tf.matmul(x[:, 0, :], lOneWeights)) + lOneConstant
                , elems=self.x_list, name='layer_one_output'
            )

            layers_2_width = 25
            lTwoWeights = tf.Variable(tf.truncated_normal([layers_1_width, layers_2_width]), name='layer_two_weights')
            lTwoConstant = tf.Variable(tf.constant(0.1, shape=[layers_2_width]), name='layer_two_constant')
            lTwoOutput = tf.map_fn(
                fn=lambda x:
                tf.nn.relu(tf.matmul(x, lTwoWeights)) + lTwoConstant
                , elems=lOneOutput, name='layer_one_output'
            )

            finalWeights = tf.Variable(tf.truncated_normal([layers_2_width, 1]), name='final_weights')
            finalConstant = tf.Variable(tf.constant(0.1, shape=[layers_2_width]), name='final_constant')
            finalOutput = tf.map_fn(
                fn=lambda x:
                tf.nn.relu(tf.matmul(x, finalWeights)) + finalConstant
                , elems=lTwoOutput, name='final_output'
            )
            self.omega = (tf.transpose(tf.reduce_sum(finalOutput, axis=(2)), name='omega'))
            # self.omega =tf.abs(
            #     tf.transpose(tf.map_fn(fn=lambda x: tf.reduce_sum(tf.multiply(x, self.W), axis=(1, 2)), elems=self.x_list, name='omega'))
            # )
            self.valuesTimeWeights = tf.reduce_sum(tf.multiply(self.values, self.omega), axis=1, name='value_time_weight')
            self.pred = tf.div(self.valuesTimeWeights, tf.reduce_sum(self.omega, axis=1), name='pred')

        with tf.name_scope('cost_and_opt'):
            self.cost = tf.reduce_sum(tf.square(self.pred - self.actual), name='cost')
            self.cost_abs = tf.reduce_sum(tf.abs(self.pred - self.actual), name='cost_abs')
            # self.train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.cost)
            self.train_op = tf.train.AdamOptimizer(start_rate).minimize(self.cost)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        ## writer
        self.abs_error = 0

        tf.summary.scalar('cost', self.cost)
        # tf.summary.histogram('weights', self.W)
        tf.summary.scalar('abs_cost', self.cost_abs)
        # tf.summary.histogram('pred', self.pred)
        self.merged = tf.summary.merge_all()  # merge the summary, make it easier to go all at once

        if summary_type =="":
            dir = "./logs/" + name
        else:
            dir = "./logs/"+summary_type+"/" + name
        if not os.path.exists(dir):
            os.makedirs(dir)

        self.writer = tf.summary.FileWriter(dir + "/")  # Passes in the logs directory's location to the writer
        self.writer.add_graph(self.sess.graph)


class MultipleLayerAggregator(LinearAggregator):
    def __init__(self,input_size, nb_pred_standing, layer_width, name='unnamed_model_low_rate', summary_type ="", rate_of_saving=20, verbose=True,start_rate=0.01):
        # super().__init__(input_size, nb_pred_standing, name, rate_of_saving, verbose)
        self.layer_width = layer_width
        self.verbose = verbose
        self.rate_of_saving = rate_of_saving
        self.name = name
        self.input_size = input_size
        self.nb_pred_standing = nb_pred_standing
        self.last_saved_epoch = -1

        with tf.name_scope('input_x'):
            self.values = tf.placeholder(tf.float32, [None, self.nb_pred_standing], name='values')

            self.x_raw = tf.placeholder(tf.float32, [None, nb_pred_standing, input_size], name='x_raw')
            self.x_list = tf.convert_to_tensor(tf.split(self.x_raw, num_or_size_splits=self.nb_pred_standing, axis=1, name='x_list'))

        with tf.name_scope('input_y'):
            self.actual = tf.placeholder(tf.float32, [None], name='actual')

        with tf.name_scope('aggregation'):
            # self.W = tf.Variable(tf.truncated_normal([1, input_size]), name='weights')
            self.W = tf.Variable(tf.ones([1, input_size]), name='weights')
            # w = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0],dtype=np.float32)
            # self.W = tf.Variable(w.reshape((1,-1)),name='weights')

            # tf.map_fn(fn=lambda x: tf.multiply(x, self.W), elems=self.x_list, name='omega')
            # tf.map_fn(fn=lambda x: tf.reduce_sum(tf.multiply(x, self.W), axis=(1,2)), elems=self.x_list, name='omega')
            previous_size = input_size
            previous_layer = self.x_list
            for i in range(len(layer_width)):
                lOneWeights = tf.Variable(tf.truncated_normal([previous_size, layer_width[i]]), name='layer_'+str(i)+'_weights')
                lOneConstant = tf.Variable(tf.constant(0.1, shape=[layer_width[i]]), name='layer_'+str(i)+'_constant')
                if i == 0:
                    lOneOutput = tf.map_fn(
                        fn=lambda x:
                        tf.nn.relu(tf.matmul(x[:, 0, :], lOneWeights)) + lOneConstant
                        , elems=previous_layer, name='layer_'+str(i)+'_output'
                    )
                else:
                    lOneOutput = tf.map_fn(
                        fn=lambda x:
                        tf.nn.relu(tf.matmul(x, lOneWeights)) + lOneConstant
                        , elems=previous_layer, name='layer_'+str(i)+'_output'
                    )
                previous_layer = lOneOutput
                previous_size = layer_width[i]
            # after adding all the layers
            final_constant = tf.Variable(tf.constant(0.1, shape=[previous_size]), name='final_constant')
            self.finalWeights = tf.Variable(tf.truncated_normal([previous_size, 1]), name='final_weights')
            finalOutput = tf.map_fn(
                fn=lambda x:
                tf.nn.relu(tf.matmul(x, self.finalWeights))+final_constant
                , elems=previous_layer, name='final_output'
            )
            self.omega = (tf.transpose(tf.reduce_sum(finalOutput, axis=(2)), name='omega'))
            # self.omega =tf.abs(
            #     tf.transpose(tf.map_fn(fn=lambda x: tf.reduce_sum(tf.multiply(x, self.W), axis=(1, 2)), elems=self.x_list, name='omega'))
            # )
            self.valuesTimeWeights = tf.reduce_sum(tf.multiply(self.values, self.omega), axis=1, name='value_time_weight')
            self.pred = tf.div(self.valuesTimeWeights, tf.reduce_sum(self.omega, axis=1), name='pred')

        with tf.name_scope('cost_and_opt'):
            self.cost = tf.reduce_sum(tf.square(self.pred - self.actual), name='cost')
            self.cost_abs = tf.reduce_sum(tf.abs(self.pred - self.actual), name='cost_abs')
            # self.train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.cost)
            self.train_op = tf.train.AdamOptimizer(start_rate).minimize(self.cost)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        ## writer
        self.abs_error = 0

        tf.summary.scalar('cost', self.cost)
        # tf.summary.histogram('weights', self.W)
        tf.summary.scalar('abs_cost', self.cost_abs)
        # tf.summary.histogram('pred', self.pred)
        self.merged = tf.summary.merge_all()  # merge the summary, make it easier to go all at once

        if summary_type =="":
            dir = "./logs/" + name
        else:
            dir = "./logs/"+summary_type+"/" + name
        if not os.path.exists(dir):
            os.makedirs(dir)

        self.writer = tf.summary.FileWriter(dir + "/")  # Passes in the logs directory's location to the writer
        self.writer.add_graph(self.sess.graph)

if __name__ == '__main__':
    # generating sample data
    nb_pred_standing = 4
    input_size = 5
    sample_size = 1000
    feats = np.abs(np.random.normal(scale=10, size=(sample_size, nb_pred_standing, input_size)))
    max_pre = np.max(np.abs(feats)) + 0.01
    true_value = 10
    vals = []
    for i in range(sample_size):
        f = feats[i, :, :]
        v = []
        for p in f[:, 0]:
            # print(p)
            sc = np.abs((20) / (1 + p))
            v.append(true_value + np.random.normal(scale=sc))
        v = np.array(v)
        vals.append(v)
    vals = np.array(vals)
    vals.shape
    feats.shape
    actuals_all = np.full(fill_value=true_value, shape=sample_size)

    model = MultipleLayerAggregator(start_rate = 0.01, input_size=5, nb_pred_standing=4, rate_of_saving=1,layer_width=[5000],name='large_one_slow_opt',summary_type='simple_simulation')
    # model = LinearAggregator(input_size=5, nb_pred_standing=4, rate_of_saving=1, name='linear_test',
    #                                  summary_type='simple_simulation')
    # model = OneLayerAggregator(start_rate=1, input_size=5, nb_pred_standing=4, rate_of_saving=1, name='tb',
    #                                 summary_type='simple_simulation')
    self = model
    model.initialise()
    # model.finalWeights.eval(model.sess)
    e=0
    for e in range(100):
        print('------ e', e, '------')
        model.train_sample(train_x=feats, train_values=vals, train_actual=actuals_all, epoch=e)
        outputs, abs_error, sqr_error = model.pred_on_sample_with_summary(test_actual=actuals_all, test_x=feats, test_values=vals, epoch=e)
        pred = model.pred_simple(test_x=feats,test_values=vals,test_actual=actuals_all)
        print('oos m_abs_err', abs_error, '| oos_m_sqr_err', sqr_error)

    print(outputs)
    print(model.getParams())
    model.sess.close()

# tensorboard --logdir=./logs --host localhost --port 8088
