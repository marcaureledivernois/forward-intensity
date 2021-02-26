import tensorflow as tf
import numpy as np
import os
from data_creation import *
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
import matplotlib.pyplot as plt
import hdf5storage
import pickle
from model_class import NeuralNetwork

###################################### roc test outsample duan #########################################################

auc_scores_duan = []
fpr_score_duan = []
tpr_score_duan = []

path = os.path.join('C:\\Users', 'divernoi', 'Dropbox', 'PhD', 'Duan Replication')
param_duan = scipy.io.loadmat(os.path.join(path, 'paramduan.mat'))['paramduan']
testdata = hdf5storage.loadmat(os.path.join(path, 'testdata.mat'))['testdata']

out_directory = 'Results NN for Intensity estimation\\real dataset\\roc\\duan\\'

# numpdata[0] : surviving
# numpdata[1] : default
# numpdata[2] : other exits


for tau in range(0,36):
    numpdata = testdata[tau]
    fullx = np.concatenate((numpdata[0],numpdata[2],numpdata[1]), axis=0)
    fully =  np.concatenate((np.zeros(numpdata[0].shape[0] + numpdata[2].shape[0]),np.ones(numpdata[1].shape[0])),axis = 0)

    f_duan = np.exp(np.dot(fullx,param_duan[:,tau]))

    res_duan = pd.DataFrame(data={'f':f_duan.reshape(f_duan.shape[0]),'y':fully})
    #res_duan = res_duan.sort_values('probmod')
    res_duan['prob'] = 1-np.exp(-res_duan['f']/12)

    fpr, tpr, _ = roc_curve(res_duan['y'], res_duan['probcal'])
    roc_auc = auc(fpr, tpr)
    fpr_score_duan.append(fpr)
    tpr_score_duan.append(tpr)
    auc_scores_duan.append(roc_auc)

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
    #plt.savefig('duan' + str(tau) + '.png')
    plt.close()
    #print('saving plot... ' + 'duan' + str(tau) + '.png')


# save scores
with open(out_directory + "auc.txt", "w") as f:
    for s in auc_scores_duan:
        f.write(str(s) +"\n")

#with open(out_directory + "fpr.txt", "w") as f:
#    for s in fpr_score_duan:
#        f.write(str(s) +"\n")
#
#with open(out_directory + "tpr.txt", "w") as f:
#    for s in tpr_score_duan:
#        f.write(str(s) +"\n")

###################################### lorenz test outsample duan ######################################################

path = os.path.join('C:\\Users', 'divernoi', 'Dropbox', 'PhD', 'Duan Replication')
param_duan = scipy.io.loadmat(os.path.join(path, 'paramduan.mat'))['paramduan']
testdata = hdf5storage.loadmat(os.path.join(path, 'testdata.mat'))['testdata']

out_directory = 'Results NN for Intensity estimation\\real dataset\\lorenz\\f\\duan\\'

# numpdata[0] : surviving
# numpdata[1] : default
# numpdata[2] : other exits

all_gini = []

for tau in range(0,36):
    numpdata = testdata[tau]
    fullx = np.concatenate((numpdata[0],numpdata[2],numpdata[1]), axis=0)
    fully =  np.concatenate((np.zeros(numpdata[0].shape[0] + numpdata[2].shape[0]),np.ones(numpdata[1].shape[0])),axis = 0)

    f_duan = np.exp(np.dot(fullx,param_duan[:,tau]))

    df = pd.DataFrame(data={'f': f_duan.reshape(f_duan.shape[0]), 'y': fully})
    df['prob'] = 1 - np.exp(-df['f'] / 12)
    df = df.sort_values('prob')
    df['cumy'] = df['y'].cumsum() / sum(df['y'])
    df['perc'] = list(range(1, len(df) + 1, 1))
    df['perc'] = df['perc'] / len(df)
    gini = (0.5 - np.trapz(df['cumy'], x=df['perc'])) / 0.5
    plt.plot(df['perc'], df['cumy'], color='darkorange', label='Linear -- Gini : %0.2f' % gini)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('Fraction of Population included')
    plt.ylabel('Fraction of Defaults included')
    plt.title('Lorenz Curve : horizon ' + str(tau))
    plt.legend(loc="upper left")
    plt.savefig(out_directory+ 'duan' + str(tau) + '.png')
    print('saving lorenz curve... duan'+str(tau)+'.png')
    plt.close()
    all_gini.append(gini)

# save scores
with open(out_directory + "gini.txt", "w") as f:
    for s in all_gini:
        f.write(str(s) +"\n")



######################################## dataframe comparison ##########################################################
# everything is coming from a test set

# f : intensity duan
# y : true label
# prob : def prob duan
# probcal : def prob neuralnet
# fmodcal : def intensity neuralnet

tau = 0
intensitytype = 'f'
hidden_dim = [5, 3]
deltaT = 1 / 12
learning_rate = 0.001
feature_size = 12
batch_size = 256
perc = 0.9
duan_replic = ""


path = os.path.join('C:\\Users', 'divernoi', 'Dropbox', 'PhD', 'Duan Replication')
param_duan = scipy.io.loadmat(os.path.join(path, 'paramduan.mat'))['paramduan']
testdata = hdf5storage.loadmat(os.path.join(path, 'testdata.mat'))['testdata']

out_directory = 'Results NN for Intensity estimation\\real dataset\\roc\\duan\\'

numpdata = testdata[tau]
fullx = np.concatenate((numpdata[0], numpdata[2], numpdata[1]), axis=0)
fully = np.concatenate((np.zeros(numpdata[0].shape[0] + numpdata[2].shape[0]), np.ones(numpdata[1].shape[0])), axis=0)

f_duan = np.exp(np.dot(fullx, param_duan[:, tau]))

res_duan = pd.DataFrame(data={'f_duan': f_duan.reshape(f_duan.shape[0]), 'y': fully})

stdfullx,meanfullx,standfullx = standardize_data(fullx)

name = intensitytype + str(tau) + '_hidden' + str(hidden_dim[0]) + '_layers' + str(len(hidden_dim)) + '_learning' + str(
    learning_rate) + '_batch' + \
       str(batch_size) + '_perc' + str(perc) + duan_replic + '_'  # remove
path = intensitytype + str(tau) + '_hidden' + str(hidden_dim[0]) + '_layers' + str(len(hidden_dim)) + '_learning' + str(
    learning_rate) + '_batch' + \
       str(batch_size) + '_perc' + str(perc) + duan_replic + '\\'  # remove
model = NeuralNetwork(hidden_dim=hidden_dim, deltaT=deltaT, learning_rate=learning_rate,
                      feature_size=feature_size, batch_size=batch_size, perc=perc, path=path, name=name)

model.load(model.path + model.name + str(999))

fmod = model.out_f(standfullx)

res_duan['fmod'] = fmod[0]
res_duan['probmod'] = 1-np.exp(-res_duan['fmod']/12)
res_duan['prob_duan'] = 1-np.exp(-res_duan['f_duan']/12)
res_duan['prob_nn'] = res_duan['probmod'] / np.mean(res_duan['probmod']) * np.mean(res_duan['y'])
res_duan['f_nn'] = - 12 * np.log(1-res_duan['prob_nn'])

# dataframe comparison
res_duan.to_pickle('res.pkl')
res = pd.read_pickle('res.pkl')

# parameters comparison (only sign is meanginful bcz one dataset is standardized and the other is not)
# all param have same sign :-)
model.getParams()
param_duan[:, tau]   #constant is first item


model.sess.close()
tf.reset_default_graph()


res_duan = res_duan.sort_values('prob_duan')
res_duan['cumy'] = res_duan['y'].cumsum() / sum(res_duan['y'])
res_duan['perc'] = list(range(1,len(res_duan)+1,1))
res_duan['perc'] = res_duan['perc'] / len(res_duan)

res_duan2 = res_duan.sort_values('prob_nn')
res_duan2['cumy_nn'] = res_duan2['y'].cumsum() / sum(res_duan2['y'])
res_duan2['perc_nn'] = list(range(1,len(res_duan2)+1,1))
res_duan2['perc_nn'] = res_duan2['perc_nn'] / len(res_duan2)

plt.plot(res_duan['perc'], res_duan['cumy'], color='darkorange', label='duan')
plt.plot(res_duan2['perc_nn'], res_duan2['cumy_nn'], color='green', label='duan_rep')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('Fraction of Population included')
plt.ylabel('Fraction of Defaults included')
plt.title('Lorenz Curve : horizon ' + str(tau))
plt.legend(loc="lower right")
plt.show()


####################### plot differences between models ##################################

res = pd.read_pickle('res.pkl')

nbtokeep = 30

length0 = sum(res['y']==0)
a = np.arange(length0)
np.random.shuffle(a)

length1 = sum(res['y']==1)
b = np.arange(length1)
np.random.shuffle(b)


res2plot = pd.concat([res.iloc[a[:nbtokeep]],res.iloc[length0 + b[:nbtokeep]]])

plt.plot(np.linspace(1,nbtokeep*2,nbtokeep*2),res2plot['prob_nn'], color='green', label='NN + [1] + exp')
plt.plot(np.linspace(1,nbtokeep*2,nbtokeep*2),res2plot['prob_duan'], color='darkorange', label='duan')
plt.axvline(x=nbtokeep)
plt.ylim([0.0, 0.02])
plt.xlabel('Firm')
plt.ylabel('Default probability')
plt.title('Comparison between Duan and NN + [1] + exp activation')
plt.legend(loc="upper right")


######################### compare intensities for surviving / defaulted firms ########################################

tau = 0
intensitytype = 'f'
hidden_dim = [5, 3]
deltaT = 1 / 12
learning_rate = 0.001
feature_size = 12
batch_size = 256
perc = 0.9
duan_replic = ""


path = os.path.join('C:\\Users', 'divernoi', 'Dropbox', 'PhD', 'Duan Replication')
param_duan = scipy.io.loadmat(os.path.join(path, 'paramduan.mat'))['paramduan']
testdata = hdf5storage.loadmat(os.path.join(path, 'testdata.mat'))['testdata']

out_directory = 'Results NN for Intensity estimation\\real dataset\\roc\\duan\\'

numpdata = testdata[tau]
fullx = np.concatenate((numpdata[0], numpdata[2], numpdata[1]), axis=0)
fully = np.concatenate((np.zeros(numpdata[0].shape[0] + numpdata[2].shape[0]), np.ones(numpdata[1].shape[0])), axis=0)

f_duan = np.exp(np.dot(fullx, param_duan[:, tau]))

res_duan = pd.DataFrame(data={'f_duan': f_duan.reshape(f_duan.shape[0]), 'y': fully})

stdfullx,meanfullx,standfullx = standardize_data(fullx)

name = intensitytype + str(tau) + '_hidden' + str(hidden_dim[0]) + '_layers' + str(len(hidden_dim)) + '_learning' + str(
    learning_rate) + '_batch' + \
       str(batch_size) + '_perc' + str(perc) + duan_replic + '_'  # remove
path = intensitytype + str(tau) + '_hidden' + str(hidden_dim[0]) + '_layers' + str(len(hidden_dim)) + '_learning' + str(
    learning_rate) + '_batch' + \
       str(batch_size) + '_perc' + str(perc) + duan_replic + '\\'  # remove
model = NeuralNetwork(hidden_dim=hidden_dim, deltaT=deltaT, learning_rate=learning_rate,
                      feature_size=feature_size, batch_size=batch_size, perc=perc, path=path, name=name)

model.load(model.path + model.name + str(999))

fmod = model.out_f(standfullx)

res_duan['fmod'] = fmod[0]
res_duan['probmod'] = 1-np.exp(-res_duan['fmod']/12)
res_duan['prob_duan'] = 1-np.exp(-res_duan['f_duan']/12)
res_duan['prob_nn'] = res_duan['probmod'] / np.mean(res_duan['probmod']) * np.mean(res_duan['y'])
res_duan['f_nn'] = - 12 * np.log(1-res_duan['prob_nn'])

nbtokeep = 30

length0 = sum(res_duan['y']==0)
a = np.arange(length0)
np.random.shuffle(a)

length1 = sum(res_duan['y']==1)
b = np.arange(length1)
np.random.shuffle(b)

res2plot = pd.concat([res_duan.iloc[a[:nbtokeep]],res_duan.iloc[length0 + b[:nbtokeep]]])

#plt.plot(np.linspace(1,nbtokeep*2,nbtokeep*2),res2plot['prob_nn'], color='green', label='NN + [1] + exp')
#plt.plot(np.linspace(1,nbtokeep*2,nbtokeep*2),res2plot['prob_duan'], color='darkorange', label='duan')
plt.plot(np.linspace(1,nbtokeep*2,nbtokeep*2),res2plot['prob_nn2'], color='red', label='[5, 3]')
plt.plot(np.linspace(1,nbtokeep,nbtokeep).reshape(-1,1),np.matlib.repmat(np.mean(res2plot['prob_nn'][:30]),1,nbtokeep).reshape(-1,1), color='red', linestyle = 'dashed', label = 'average')
plt.plot(np.linspace(30,nbtokeep*2,nbtokeep+1).reshape(-1,1),np.matlib.repmat(np.mean(res2plot['prob_nn'][30:]),1,nbtokeep+1).reshape(-1,1), color='red', linestyle='dashed')
#plt.plot(np.linspace(1,nbtokeep,nbtokeep).reshape(-1,1),np.matlib.repmat(np.mean(res2plot['prob_nn'][:30])+1.6*np.std(res2plot['prob_nn'][:30]),1,nbtokeep).reshape(-1,1), color='black', label='[5, 3]')
#plt.plot(np.linspace(31,nbtokeep*2,nbtokeep).reshape(-1,1),np.matlib.repmat(np.mean(res2plot['prob_nn'][30:])-1.6*np.std(res2plot['prob_nn'][30:]),1,nbtokeep).reshape(-1,1), color='black', label='[5, 3]')
plt.axvline(x=nbtokeep)
plt.ylim([0.0, 0.01])
plt.xlabel('Firm')
plt.ylabel('Default probability')
plt.title('Default probabilities for surviving / defaulted firms')
plt.legend(loc="upper right")






