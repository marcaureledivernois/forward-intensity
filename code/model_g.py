import tensorflow as tf
import numpy as np
import os
from data_creation import *

##################################### model design #####################################################################


hidden_dim = 100
deltaT = tf.constant(1/12, name='deltaT')
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
    f = tf.clip_by_value(f,fmin,fmax)

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

def load(epoch_to_load=-1,name="test_model"):
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

#marche pas trop mal : test_model epoch 99 , test_model_2 epoch 99
# test_model epoch 99 : fake data, feed 1 by 1
# test_model_2 epoch 1-99 : fake data, feed 1 by 1
# test_model_batches : fake data, batches feeding
# "test_model_fullbatches" : fake data, feed full data --> the more u feed data the less learning rate has to be
# model_1_realdata999 : 93% AUC on f
# model_1_realdata_h99 : 70% AUC on h
# model_traintest_realdata_h9 : 63% on h outofsample
#

######################################### DATA #########################################################################

x_train,y_train,x_test,y_test = RealData_f()
# x_train,y_train,x_test,y_test = RealData_h()
# x,y = FakeData()

############################################### session ################################################################

sess = tf.Session()

load = False

if load==True:
    load(9,"model_1_realdata")      # if nothing specified, load last epoch. otherwise load(35) will load 35th epoch

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

# smart feeding batches

########### f model ##########

batch_size = 128
perc = 0.5     # percentage of surv firms in each batch
nb0 = sum(y_train==0)
nb1 = sum(y_train==1)

for e in range(10):
    for enum in range(100000):
        id_s = np.random.randint(nb0,size=int(batch_size*perc))
        id_d = nb0 + np.random.randint(nb1,size=int(batch_size*(1-perc)))
        i = np.concatenate((id_s,id_d),axis=None)
        sess.run([train_op], feed_dict={x_raw: x_train[i], y_raw: y_train[i]})
    print('-----',e,'-----')
    loss_value, f_value= sess.run([loss,f], feed_dict={x_raw: x_train, y_raw: y_train})
    print('loss:', loss_value)
    #writer.add_summary(summary, e)
    name="model_traintest_realdata_f"
    save_model(e,name)
print('final_f', f_value)

# smart feeding batches

############## h model ##########

# batch_size = 128
# perc = 0.5     # percentage of label 0 firms in each batch
# nb0 = sum(y_train==0)
# nb1 = sum(y_train==1)
#
# for e in range(10):
#     for enum in range(100000):
#         id_s = np.random.randint(nb0,size=int(batch_size*perc))                                                         #random select 64 observations of label 0
#         id_d = nb0 + np.random.randint(nb1,size=int(batch_size*(1-perc)))                                               #random select 64 observations of label 1
#         i = np.concatenate((id_s,id_d),axis=None)                                                                       #concatenate into batch of 128 observations
#         sess.run([train_op], feed_dict={x_raw: x_train[i], y_raw: y_train[i]})                                          #feed batch
#     print('-----',e,'-----')
#     loss_value, h_value, summary = sess.run([loss,f, merged], feed_dict={x_raw: x_train, y_raw: y_train})
#     print('loss:', loss_value)
#     writer.add_summary(summary, e)
#     name="model_traintest_realdata_h"
#     save_model(e,name)
# print('final_h', h_value)


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


loss_value, f_value = sess.run([loss, f], feed_dict={x_raw: x_test, y_raw: y_test})
res = pd.DataFrame(data={'f':f_value.reshape(f_value.shape[0]),'y':y_test})

# loss_value, h_value = sess.run([loss, f], feed_dict={x_raw: x_test, y_raw: y_test})
# res = pd.DataFrame(data={'f':h_value.reshape(h_value.shape[0]),'y':y_test})

res = res.sort_values('y')

res['prob'] = 1-np.exp(-res['f']/12)   # maybe play with the delta constant to cheat on AUC a bit
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





#todo : batches instead of 1 by  1 ....... done
#todo : compute confusion matrix ........ done
#todo : inferences, how does f change if change a little bit 1 variable of x... average everything -> done, very good results
#todo : out of sample .... in progress
#todo : calibration, compute true proba (combining f and g), improve learning process
#todo : test duan

################################# test duan ############################################################################

#insample

param_duan = np.asarray([-7.65734125130025,
-0.562149707094731,
0.0378205129635966,
-0.209429482121766,
-2.59008813839482,
-0.408954058692953,
0.0233235904849505,
-0.342577280673194,
1.01639749959294,
-0.843774086776647,
-1.44584737215847,
-0.0629270992719251,
0.122237233351354])

fullx = np.concatenate((mat['firm0'], mat['firm2'], mat['firm1'] ), axis=0)
fully =  np.concatenate((np.zeros(mat['firm0'].shape[0] + mat['firm2'].shape[0]),np.ones(mat['firm1'].shape[0])),axis = 0)

f_duan = np.exp(np.dot(fullx,param_duan))

res_duan = pd.DataFrame(data={'f':f_duan.reshape(f_duan.shape[0]),'y':fully})
res_duan = res_duan.sort_values('y')

res_duan['prob'] = 1-np.exp(-res_duan['f']/12)   # maybe play with the delta constant to cheat on AUC a bit

fpr, tpr, _ = roc_curve(res_duan['y'], res_duan['prob'])
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

###################################### test outsample duan #############################################################

param_duan = np.asarray([-7.86144506618542,
-0.552610487193675,
0.0548615146486248,
-0.0801804520048898,
-2.62125971730804,
-0.422817276072109,
0.0276430120170916,
-0.327003515117461,
1.21551392895314,
-0.957699686717992,
-1.46885707089988,
-0.0525584240814198,
0.103615682032496])

path = os.path.join('C:\\Users', 'divernoi', 'Dropbox', 'PhD', 'Duan Replication')
testmat = scipy.io.loadmat(os.path.join(path, 'test.mat'))

fullx = np.concatenate((testmat['firm0_test'], testmat['firm2_test'], testmat['firm1_test'] ), axis=0)
fully =  np.concatenate((np.zeros(testmat['firm0_test'].shape[0] + testmat['firm2_test'].shape[0]),np.ones(testmat['firm1_test'].shape[0])),axis = 0)

f_duan = np.exp(np.dot(fullx,param_duan))

res_duan = pd.DataFrame(data={'f':f_duan.reshape(f_duan.shape[0]),'y':fully})
res_duan = res_duan.sort_values('y')

res_duan['prob'] = 1-np.exp(-res_duan['f']/12)   # maybe play with the delta constant to cheat on AUC a bit

fpr, tpr, _ = roc_curve(res_duan['y'], res_duan['prob'])
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




################################# test insample duan ###################################################################

#insample

param_duan = np.asarray([-7.65734125130025,
-0.562149707094731,
0.0378205129635966,
-0.209429482121766,
-2.59008813839482,
-0.408954058692953,
0.0233235904849505,
-0.342577280673194,
1.01639749959294,
-0.843774086776647,
-1.44584737215847,
-0.0629270992719251,
0.122237233351354])

fullx = np.concatenate((mat['firm0'], mat['firm2'], mat['firm1'] ), axis=0)
fully =  np.concatenate((np.zeros(mat['firm0'].shape[0] + mat['firm2'].shape[0]),np.ones(mat['firm1'].shape[0])),axis = 0)

f_duan = np.exp(np.dot(fullx,param_duan))

res_duan = pd.DataFrame(data={'f':f_duan.reshape(f_duan.shape[0]),'y':fully})
res_duan = res_duan.sort_values('y')

res_duan['prob'] = 1-np.exp(-res_duan['f']/12)   # maybe play with the delta constant to cheat on AUC a bit

fpr, tpr, _ = roc_curve(res_duan['y'], res_duan['prob'])
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
    new_f = sess.run(f,feed_dict={x_raw:data,y_raw : y_test})
    tmp = pd.DataFrame(data={'f': new_f.reshape(new_f.shape[0])})
    data[:, col_to_change] -= mult
    return tmp

def comp_study(col):
    comp = createcol(x_test,-2,col)

    for i in np.linspace(-1.8,1.9,19):
        newcol = createcol(x_test,i,col)
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








