import tensorflow as tf
import numpy as np
import os
from data_creation import *
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

######################################### Comments #####################################################################

# comment : 1. for smaller horizons single  layer with few neurons seems to work better...!
#              but for longer horizons 2 layers with few neurons work better. show it by showing roc with few neurons 1 layer for longer horizons are worse than roc with few leurons 2 layers
#              nice result bcz : makes sense that on short horizons intensities can be approx by linear assump but the longer the horizon the more likely non-linearities show up and makes the linear assumption inefficent
#           2. !!!!!! very nice advantage of NN : can "chose" fraction of true positive you want... for instance if we converge quickly
#              to 100% of defaulted firms predicted, its very nice, if not we can still "overfit more" to try to achieve it. in duan, much less
#              flexibility because we have 1 solution which is local max of likelihood and no choice than use this. in NN more flexibility because
#              span of solutions is bigger and can look into bigger surface.
#              i can motivate this by showing roc curve converging faster to  almost 100% true positive


# TODO DONE
#todo : try replicate duan model WITH LINEAR ASSUMPTION in a neural network
#todo : calibrate with average number of defaults and average number of defaults predicted by model
#todo : test many architecture and create table to see which one is better

# TODO
#todo : tensorboard : graph [5, 3]
#todo : goodtest : for a model where outsample loss decrease nicely, look roc curve epoch after epoch and see if it really becomes better or not
#todo : at a good epoch, try decrease learning_rate and look for improvement
#todo : compute true proba (combining f and g) and true roc (for longer horizons) -> WIP
#todo : improve learning process (train/valid/test)
#todo : dummy variables to help model predict
#todo : try crossentropy instead of duan loss
#todo : random forest
#todo : show how model converges epoch after epoch. i tried and it actually converges nicely. (look at model.getparam and roc curve epoch after epoch). see that param doesnt change much after x epoch... "stable parameters"
#todo : sensitivities for each horizon. can also do a graph to show sensi of 1 var for all horizons (to compare with duan)
#todo : plot of auc score for each model architecture, showing that short horizons need few neurons few layers and longer horizons need deeper net + compare with duan
#todo : term strucutre (+nelson siegel), sensi par horizon, 3d plot roc, label specific roc curve, xaxis=tau yaxis=auc plot all models
#todo : show graph model.getparam() for each tau... see how param evolves through horizon
#todo : lorenz curve, precision recall, compare with ratings (see leippold)
#todo : adam vs default algo
#todo : convert term struc into cumprob an prob (need true prob), see how forwarad intensities evolves through time

#todo : link my sensi plots to existing literature
#todo : firms generally pay a spread over the default-free rate of interest that is proportional to their default probability to compensate lenders for this uncertainty

######################################### Test model ###################################################################

def testmodel(model,x,y,path, save = True):
    loss_value, f_value = model.pred(x,y)      # outsample
    res = pd.DataFrame(data={'f':f_value.reshape(f_value.shape[0]),'y':y})
    res = res.sort_values('y')
    res['prob'] = 1-np.exp(-res['f']/12)   # maybe play with the delta constant 
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(res['y'], res['prob'])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    if save == True:
        plt.savefig(path + model.name + '.png')
        print('saving roc curve... ' + model.name + '.png')
    plt.close()
    return res, loss_value, roc_auc, fpr, tpr

def LorenzCurve(model,x_test,y_test,tau,label,color,path, save = True):
    '''require model, x_test and y_test'''
    loss_value, f_value = model.pred(x_test, y_test)  # outsample
    df = pd.DataFrame(data={'f': f_value.reshape(f_value.shape[0]), 'y': y_test})
    df['prob'] = 1-np.exp(-df['f']/12)   # maybe play with the delta constant 
    df = df.sort_values('prob')
    df['cumy'] = df['y'].cumsum() / sum(df['y'])
    df['perc'] = list(range(1,len(df)+1,1))
    df['perc'] = df['perc'] / len(df)

    gini = (0.5 - np.trapz(df['cumy'], x=df['perc']))/0.5

    plt.plot(df['perc'], df['cumy'], color=color, label=label + ' -- Gini : %0.2f' % gini)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('Fraction of Population included')
    plt.ylabel('Fraction of Defaults included')
    plt.title('Lorenz Curve : horizon ' + str(tau))
    plt.legend(loc="upper left")
    if save == True:
        plt.savefig(path + model.name + '.png')
        print('saving lorenz curve... ' + model.name + '.png')
    plt.close()
    return df['perc'],df['cumy'], gini

##################################### model design #####################################################################

class NeuralNetwork():
    def __init__(self, hidden_dim = [100,100], deltaT = 1/12, learning_rate = 0.01,
                 feature_size = 12, batch_size = 128,perc = 0.5, path ="", name = ""):
        self.hidden_dim = hidden_dim
        self.deltaT = tf.constant(deltaT, name='deltaT')
        self.one = tf.constant(1., name="one")
        self.fmin = tf.constant(0.00001, name="fmin")
        self.fmax = tf.constant(9999999999. , name="fmax")
        self.learning_rate = learning_rate
        self.feature_size = feature_size
        self.batch_size = batch_size
        self.name = name
        self.path = path
        self.perc = perc

        with tf.name_scope('input_feeding'):
            self.x_raw = tf.placeholder(tf.float32, shape= (None,feature_size) ,name="x_raw")
            self.y_raw = tf.placeholder(tf.float32, shape= (None,) ,name="y_raw")

        with tf.name_scope('nets'):
            if len(self.hidden_dim) == 1:
                self.w1 = tf.Variable(tf.random_normal([feature_size, hidden_dim[0]], dtype=tf.float32), name='weights_1')
                self.b1 = tf.Variable(tf.zeros([hidden_dim[0]]), name='biases_1')
                self.out1 = tf.nn.sigmoid(tf.matmul(self.x_raw, self.w1) + self.b1, name='output_1')  #add back
                #self.out1 = tf.exp(tf.matmul(self.x_raw, self.w1) + self.b1, name='output_1')       #remove

                self.finalW = tf.Variable(tf.truncated_normal([hidden_dim[0], 1]))   #add back
                self.f=tf.matmul(self.out1,self.finalW)                              #add back
                #self.f = self.out1                                                    #remove

            if len(self.hidden_dim) == 2:
                self.w1 = tf.Variable(tf.random_normal([feature_size, hidden_dim[0]],dtype=tf.float32), name='weights_1')
                self.b1 = tf.Variable(tf.zeros([hidden_dim[0]]), name='biases_1')
                self.out1 = tf.nn.sigmoid(tf.matmul(self.x_raw, self.w1) + self.b1, name='output_1')        #add back


                self.w2 = tf.Variable(tf.random_normal([hidden_dim[0], hidden_dim[1]],dtype=tf.float32), name='weights_2')
                self.b2 = tf.Variable(tf.zeros([hidden_dim[1]]), name='biases_2')
                self.out2 = tf.nn.sigmoid(tf.matmul(self.out1, self.w2) + self.b2, name='output_2')

                self.finalW = tf.Variable(tf.truncated_normal([hidden_dim[1],1]))
                self.f=tf.matmul(self.out2,self.finalW)

            self.f = tf.clip_by_value(self.f,self.fmin,self.fmax)
            # self.f = tf.exp(self.f)

        with tf.name_scope('loss_and_training'):
            self.loss = tf.negative(tf.reduce_sum(
                tf.add(
                    tf.multiply(tf.transpose(tf.multiply(tf.negative(self.f),deltaT)),tf.add(self.one,tf.negative(self.y_raw))),
                    tf.multiply(tf.transpose(tf.log(tf.add(self.one,tf.negative(tf.exp(tf.multiply(tf.negative(self.f),deltaT)))))),self.y_raw)
                )
            ),name="loss")

            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()

        tf.summary.scalar('loss', self.loss)
        #tf.summary.histogram('weights1', self.w1)

        self.merged = tf.summary.merge_all()  # merge the summary, make it easier to go all at once
        summary_type = 'marc_is_nice_4'
        dir = "./logs/" + summary_type + "/" + name
        if not os.path.exists(dir):
            os.makedirs(dir)

        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter(dir + "/")  # Passes in the logs directory's location to the writer
        self.writer.add_graph(self.sess.graph)


    def initialise(self):
        self.sess.run(tf.global_variables_initializer())

    def load(self,name="test_model"):
        self.saver.restore(self.sess, 'tf_models/' + name + '.ckpt')

    def save_model(self,epoch,path,name):
        if not os.path.exists('tf_models/' + self.path):
            os.makedirs('tf_models/' + self.path)
        save_path = self.saver.save(self.sess, 'tf_models/' + path + name + str(epoch) + '.ckpt')
        print('Model saved to {}'.format(save_path))

    def training(self,x_train,y_train,e):
        nb0 = sum(y_train == 0)
        nb1 = sum(y_train == 1)

        print('-----', e, '-----')

        for enum in range(x_train.shape[0] // self.batch_size):
            id_s = np.random.randint(nb0, size=int(self.batch_size * self.perc))
            id_d = nb0 + np.random.randint(nb1, size=int(self.batch_size * (1 - self.perc)))
            i = np.concatenate((id_s, id_d), axis=None)
            self.sess.run([self.train_op], feed_dict={self.x_raw: x_train[i], self.y_raw: y_train[i]})
        self.save_model(e, self.path, self.name)

    def pred(self,x,y):
        loss_value, f_value = self.sess.run([self.loss, self.f], feed_dict={self.x_raw: x, self.y_raw: y})
        return loss_value, f_value

    def pred_and_write_summary(self, x,y, epoch):
        """
        :param test_x: shape (sample_size, nb_pred, number_feature)
        :param test_values: shape (sample_size, nb_pred)
        :param test_actual: shape (sample_size, )
        :param epoch:
        :return:
        """
        summary, loss_value, f_value = self.sess.run([self.merged, self.loss, self.f], feed_dict={self.x_raw: x, self.y_raw: y})
        self.writer.add_summary(summary, epoch)
        return loss_value, f_value

    def out_f(self,x):
        f_value = self.sess.run([self.f], feed_dict={self.x_raw: x})
        return f_value

    def getParams(self):
        w1_val = self.w1.eval(self.sess)
        b1_val = self.b1.eval(self.sess)
        return w1_val, b1_val









