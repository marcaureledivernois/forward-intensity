import tensorflow as tf
import numpy as np
import os
from data_creation import *

###################################### fake data creation ##############################################################

feature_size = 12

def FakeData():
    sample_size = 1000
    pos_feature = [1,3]     # pos_feat means : if higher value, higher proba of being defaulted
    neg_feature = [2]       # neg_feat means : if higher value, lower proba of being defaulted
    x = np.random.normal(size=(sample_size,feature_size))

    temp=np.add(np.sum(x[:,pos_feature].clip(0,100),axis=1),
                np.abs(x[:,neg_feature].clip(-100,0)).reshape(sample_size))
    percentage_default = 0.05
    treshold=np.quantile(temp,1-percentage_default)
    noise_epsilon=0.1
    y = 1*((temp+np.random.normal(size=sample_size,scale=noise_epsilon))>treshold)
    return x,y

###################################### duan data creation ##############################################################

import scipy.io

path = os.path.join('C:\\Users', 'divernoi', 'Dropbox', 'PhD', 'Duan Replication')
mat = scipy.io.loadmat(os.path.join(path, 'data0.mat'))

# firm0 : surviving, firm1 : default, firm2 : otherexits

def RealData():
    joined_data_x = np.concatenate((mat['firm0'],  mat['firm2'], mat['firm1']), axis=0)
    y = np.concatenate((np.zeros(mat['firm0'].shape[0]+mat['firm2'].shape[0]),np.ones(mat['firm1'].shape[0])),axis=0)
    x = (joined_data_x[:,1:13] - np.mean(joined_data_x[:,1:13],axis=0))/np.std(joined_data_x[:,1:13],axis=0)
    return x,y


######################################### DATA #########################################################################

x_train,y_train,x_test,y_test = RealData_f()

##################################### model design #####################################################################

hidden_dim = 100
deltaT = tf.constant(3/12, name='deltaT')
one = tf.constant(1., name="one")
fmin = tf.constant(0.01, name="fmin")
fmax = tf.constant(9999999999. , name="fmax")
learning_rate=0.1

with tf.name_scope('input_feeding'):
    x_raw = tf.placeholder(tf.float32, shape= (None,feature_size) ,name="x_raw")
    y_raw = tf.placeholder(tf.float32, shape= (None,) ,name="y_raw")

with tf.name_scope('nets'):
    w1 = tf.Variable(tf.random_normal([feature_size, hidden_dim],dtype=tf.float32), name='weights_1')
    b1 = tf.Variable(tf.zeros([hidden_dim]), name='biases_1')
    out1 = tf.nn.sigmoid(tf.matmul(x_raw, w1) + b1, name='output_1')

    finalW = tf.Variable(tf.truncated_normal([hidden_dim,1]))
    f=tf.matmul(out1,finalW)
    #f = tf.clip_by_value(f,fmin,fmax)
    f = tf.exp(f)

with tf.name_scope('loss_and_training'):
    loss = tf.negative(tf.reduce_sum(
        tf.add(
            tf.multiply(tf.transpose(tf.multiply(tf.negative(f),deltaT)),tf.add(one,tf.negative(y_raw))),
            tf.multiply(tf.transpose(tf.log(tf.add(one,tf.negative(tf.exp(tf.multiply(tf.negative(f),deltaT)))))),y_raw)
        )
    ),name="loss")

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

def load(epoch_to_load=-1,name="f11_hidden1_learning0.001_batch128_perc0.9_"):
    if epoch_to_load == -1:
        # print('loading ', self.last_saved_epoch)
        l = os.listdir('tf_models')
        ll = []
        for x in l:
            if name in x:
                ll.append(x)
        e = max([int(x.split('.')[0].split(name)[1]) for x in ll])
        saver.restore(sess, 'tf_models/' + name + str(e) + '.ckpt')
    else:
        # print('loading ', epoch_to_load)
        saver.restore(sess, 'tf_models/' + name + str(epoch_to_load) + '.ckpt')
def save_model(epoch,name):
    save_path = saver.save(sess, 'tf_models/' + name + str(epoch) + '.ckpt')
    # print('updated to ', self.last_saved_epoch)
    # print('Model saved to {}'.format(save_path))

############################################### session ################################################################

sess = tf.Session()

load = False

if load==True:
    load()      # if nothing specified, load last epoch. otherwise load(35) will load 35th epoch

tf.summary.scalar('loss', loss)
# tf.summary.histogram('weights1', w1)
# tf.summary.histogram('pred', self.pred)
merged = tf.summary.merge_all()  # merge the summary, make it easier to go all at once
summary_type = 'marc_is_even_nicer3'
dir = "./logs/" + summary_type
if not os.path.exists(dir):
    os.makedirs(dir)

writer = tf.summary.FileWriter(dir + "/")  # Passes in the logs directory's location to the writer
writer.add_graph(sess.graph)

sess.run(init)

# sess.run(loss,feed_dict={x_raw:x[0,:].reshape(1,-1),y_raw : y[0].reshape((1,))})

# feed one by one

for e in range(10):
    for i in range(len(x)):
        sess.run([train_op], feed_dict={x_raw: [x_train[i]], y_raw: [y_train[i]]})
    print('-----',e,'-----')
    loss_value, f_value = sess.run([loss,f], feed_dict={x_raw: x_train, y_raw: y_train})
    print('loss:', loss_value)
    #writer.add_summary(summary, e)
    name="model_2_fakedata"
    save_model(e,name)
print('final_f', f_value)

# smart feeding one by one

# for e in range(10):
#     for enum in range(100000):
#         if np.random.randint(2) == 1:
#             i = np.random.randint(mat['firm0'].shape[0])
#         else:
#             i = mat['firm0'].shape[0] + mat['firm2'].shape[0] + np.random.randint(mat['firm1'].shape[0])
#         sess.run([train_op], feed_dict={x_raw: [x[i]], y_raw: [y[i]]})
#     print('-----',e,'-----')
#     loss_value, f_value, summary = sess.run([loss,f, merged], feed_dict={x_raw: x, y_raw: y})
#     print('loss:', loss_value)
#     writer.add_summary(summary, e)
#     name="model_1_realdata"
#     save_model(e,name)
# print('final_f', f_value)


# smart feeding batches
batch_size = 128
perc = 0.5     # percentage of surv firms in each batch
nb0 = sum(y_train == 0)
nb1 = sum(y_train == 1)

for e in range(10):
    for enum in range(100000):
        id_s = np.random.randint(nb0, size=int(batch_size * perc))
        id_d = nb0 + np.random.randint(nb1, size=int(batch_size * (1 - perc)))
        i = np.concatenate((id_s, id_d), axis=None)
        sess.run([train_op], feed_dict={x_raw: x_train[i], y_raw: y_train[i]})
    print('-----',e,'-----')
    loss_value, f_value = sess.run([loss,f], feed_dict={x_raw: x_train, y_raw: y_train})
    print('loss:', loss_value)
    # writer.add_summary(summary, e)
    name="model_1_realdata"
    save_model(e,name)
print('final_f', f_value)


# for i in range(1000):
#     loss1 = sess.run(loss,feed_dict={x_raw:x[i,:].reshape(1,-1),y_raw : y[i].reshape((1,))})
#     summ += loss1
#     print(summ)

# full batch

# for e in range(10):
#     sess.run(train_op, feed_dict={x_raw: x, y_raw: y})
#     print('-----',e,'-----')
#     loss_value, f_value = sess.run([loss,f], feed_dict={x_raw: x, y_raw: y})
#     print('loss:', loss_value)
#     #writer.add_summary(summary, e)
#     name="test_model_fullbatches"
#     save_model(e,name)
# print('final_f', f_value)


w_val = sess.run(w1)
b1_val = sess.run(b1)
#f_val = sess.run(f)

#sess.close()


###################################### test model ######################################################################


import pandas as pd

loss_value, f_value = sess.run([loss, f], feed_dict={x_raw: x, y_raw: y})
res = pd.DataFrame(data={'f':f_value.reshape(f_value.shape[0]),'y':y})

res = res.sort_values('y')
res

res['prob'] = 1-np.exp(-res['f']/12)
# res = res.sort_values('prob')

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Compute ROC curve and ROC area for each class
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
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

#todo : batches instead of 1 by  1 ---->>>> need decrease learning rate otherwise loss doesnt decrease4
#todo : compute confusion matrix
#todo : inferences, how does f change if change a little bit 1 variable of x... average everything -> done, very good results

###################################### comparative study ###############################################################

# % col0 = SPret
# % col1= Tbill
# % col2= Cash/TA
# % col3= NI/TA
# % col4= Size
# % col5= DtD
# % col6= MBratio
# % col7=  Diff_1 CASH/TA
# % col8 =  Diff_1 NI/TA
# % col9 =  Diff_1 Size
# % col10 =  Diff_1 DtD
# % col11 =  Diff_1 MBratio


def createcol(data,mult,col_to_change):
    data[:,col_to_change] += mult
    new_f = sess.run(f,feed_dict={x_raw:data,y_raw : y})
    tmp = pd.DataFrame(data={'f': new_f.reshape(new_f.shape[0])})
    data[:, col_to_change] -= mult
    return tmp

def comp_study(col):
    comp = createcol(x,-2,col)

    for i in np.linspace(-1.8,1.9,19):
        newcol = createcol(x,i,col)
        comp = comp.join(newcol,rsuffix=i)
    return comp

comp0 = comp_study(0)
comp1 = comp_study(1)
comp2 = comp_study(2)
comp3 = comp_study(3)
comp4 = comp_study(4)
comp5 = comp_study(5)
comp6 = comp_study(6)
comp7 = comp_study(7)
comp8 = comp_study(8)
comp9 = comp_study(9)
comp10 = comp_study(10)
comp11 = comp_study(11)



plt.plot(np.linspace(-2,2,20),np.mean(comp0),'r')
plt.plot(np.linspace(-2,2,20),np.mean(comp1),'b')
plt.plot(np.linspace(-2,2,20),np.mean(comp2),'g')
plt.plot(np.linspace(-2,2,20),np.mean(comp3),'y')
plt.plot(np.linspace(-2,2,20),np.mean(comp4),'o')
plt.plot(np.linspace(-2,2,20),np.mean(comp5),'p')
plt.plot(np.linspace(-2,2,20),np.mean(comp6),'c')
plt.plot(np.linspace(-2,2,20),np.mean(comp7),'b')
plt.plot(np.linspace(-2,2,20),np.mean(comp8),'v')
plt.plot(np.linspace(-2,2,20),np.mean(comp9),'m')
plt.plot(np.linspace(-2,2,20),np.mean(comp10),'g')
plt.plot(np.linspace(-2,2,20),np.mean(comp11))
plt.legend(('SPret', 'Tbill', 'CASH/TA','NI/TA','Size','DtD','MBrat','D_CASH/TA','D_NI/TA','D_Size','D_DtD','D_MBrat'),
           loc='upper left')
plt.show()








