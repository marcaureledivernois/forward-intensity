from model_class import *
from data_creation import *
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import pickle
import pylatex as lat
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


################################ xaxis : tau , yaxis : auc_score #######################################################

auc_scores_duan = []
auc_1 = []
auc_21 = []
auc_32 = []
auc_53 = []
auc_105 = []

# load
with open("Results NN for Intensity estimation\\real dataset\\roc\\duan\\auc.txt", "r") as f:
  for line in f:
      auc_scores_duan.append(float(line.strip()))

with open("Results NN for Intensity estimation\\real dataset\\roc\\f\\[1]\\auc.txt", "r") as f:
  for line in f:
      auc_1.append(float(line.strip()))

with open("Results NN for Intensity estimation\\real dataset\\roc\\f\\[2, 1]\\auc.txt", "r") as f:
  for line in f:
      auc_21.append(float(line.strip()))

with open("Results NN for Intensity estimation\\real dataset\\roc\\f\\[3, 2]\\auc.txt", "r") as f:
    for line in f:
        auc_32.append(float(line.strip()))

with open("Results NN for Intensity estimation\\real dataset\\roc\\f\\[5, 3]\\auc.txt", "r") as f:
  for line in f:
      auc_53.append(float(line.strip()))

with open("Results NN for Intensity estimation\\real dataset\\roc\\f\\[10, 5]\\auc.txt", "r") as f:
  for line in f:
      auc_105.append(float(line.strip()))

plt.figure()
lw = 2
tau = np.linspace(1,36,36)
plt.plot(tau, auc_scores_duan, color='darkorange',
         lw=lw, label='Duan')
plt.plot(tau, auc_1, color='black',
         lw=lw, label='[1]')
plt.plot(tau, auc_21, color='blue',
         lw=lw, label='[2, 1]')
plt.plot(tau, auc_32, color='red',
         lw=lw, label='[3, 2]')
plt.plot(tau, auc_53, color='green',
         lw=lw, label='[5, 3]')
plt.plot(tau, auc_105, color='lightblue',
         lw=lw, label='[10, 5]')
plt.xlim([1.0, 36.0])
plt.ylim([0.5, 1.0])
plt.xlabel('Tau')
plt.ylabel('AUC Score')
plt.title('AUC -- Duan vs NeuralNetwork')
plt.legend(loc="lower right")



####################################### roc plots #######################################################################

all_fpr = []
all_tpr = []
all_auc = []

intensitytype = 'f'
hidden_dim = [5, 3]
deltaT = 1 / 12
learning_rate = 0.001
feature_size = 12
batch_size = 256
perc = 0.9
duan_replic = 'duan_replic'

directory = 'Results NN for Intensity estimation\\real dataset\\roc\\' + intensitytype + '\\' + str(hidden_dim) + '\\'

for tau in range(0,36):
    name = intensitytype + str(tau) + '_hidden' + str(hidden_dim[0]) + '_layers' + str(
        len(hidden_dim)) + '_learning' + str(learning_rate) + '_batch' + \
           str(batch_size) + '_perc' + str(perc) + '_'
    path = intensitytype + str(tau) + '_hidden' + str(hidden_dim[0]) + '_layers' + str(
        len(hidden_dim)) + '_learning' + str(learning_rate) + '_batch' + \
           str(batch_size) + '_perc' + str(perc) + '\\'

    model = NeuralNetwork(hidden_dim=hidden_dim, deltaT=deltaT, learning_rate=learning_rate,
                          feature_size=feature_size, batch_size=batch_size, perc=perc, path=path, name=name)
    if intensitytype == 'f':
        x_train, y_train, x_test, y_test,_,_ = RealData_f(tau=tau)
    else:
        x_train, y_train, x_test, y_test,_,_ = RealData_h(tau=tau)

    model.load(model.path + model.name + str(995))

    _,_,auc_score,fpr_vec,tpr_vec = testmodel(model,x_test,y_test, path = directory,save = True)

    all_fpr.append(fpr_vec)
    all_tpr.append(tpr_vec)
    all_auc.append(auc_score)
    model.sess.close()
    tf.reset_default_graph()

# save
with open(directory + "auc.txt", "w") as f:
    for s in all_auc:
        f.write(str(s) +"\n")


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

for tau in [6,12,24,35]:
    intensitytype = 'f'
    hidden_dim = [5, 3]
    deltaT = 1 / 12
    learning_rate = 0.001
    feature_size = 12
    batch_size = 128    #128
    perc = 0.9

    directory = 'Results NN for Intensity estimation\\real dataset\\sensitivities\\' + intensitytype + '\\' + str(hidden_dim) + '\\'
    if not os.path.exists(directory):
        os.makedirs(directory)

    name = intensitytype + str(tau) + '_hidden' + str(hidden_dim[0]) + '_layers' + str(
        len(hidden_dim)) + '_learning' + str(learning_rate) + '_batch' + \
           str(batch_size) + '_perc' + str(perc) + '_'
    path = intensitytype + str(tau) + '_hidden' + str(hidden_dim[0]) + '_layers' + str(
        len(hidden_dim)) + '_learning' + str(learning_rate) + '_batch' + \
           str(batch_size) + '_perc' + str(perc) + '\\'
    model = NeuralNetwork(hidden_dim=hidden_dim, deltaT=deltaT, learning_rate=learning_rate,
                          feature_size=feature_size, batch_size=batch_size, perc=perc, path=path, name=name)

    model.load(model.path + model.name + str(9))       #9


    if intensitytype == 'f':
        x_train, y_train, x_test, y_test,_,_ = RealData_f(tau=tau)
    else:
        x_train, y_train, x_test, y_test,_,_ = RealData_h(tau=tau)


    def createcol(data,mult,col_to_change):
        data[:,col_to_change] += mult                                                                                   # uncomment this line when want to average output instead of average input
        #data[col_to_change] += mult

        loss_value, new_f = model.pred(data, y_test)                                                                    # uncomment this line when want to average output instead of average input
        #loss_value, new_f = model.pred(data.reshape(1,-1), y_test)

        # new_f = sess.run(f,feed_dict={x_raw:data,y_raw : y_test})

        tmp = pd.DataFrame(data={'f': new_f.reshape(new_f.shape[0])})

        data[:,col_to_change] -= mult                                                                                   # uncomment this line when want to average output instead of average input
        #data[col_to_change] -= mult
        return tmp

    def comp_study(col):
        comp = createcol(x_test, -2, col)                                                                               # uncomment this line when want to average output instead of average input
        #comp = createcol(np.mean(x_test,0),-2,col)

        for i in np.linspace(-1.8,2,20):
            newcol = createcol(x_test,i,col)                                                                            # uncomment this line when want to average output instead of average input
            #newcol = createcol(np.mean(x_test, 0), i, col)

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

    # todo : y axis start at 0, %change in yaxis instead of abs value?, subplot
    def plot_sensitivities(compdata,color,perc):
        plt.plot(np.linspace(-2, 2, 21), np.mean(compdata), color)
        plt.plot(np.linspace(-2, 2, 21), np.mean(compdata) + perc * np.std(compdata), 'g--')
        plt.plot(np.linspace(-2, 2, 21), np.mean(compdata) - perc * np.std(compdata), 'r--')
        #plt.xlabel('Absolute change of variable')
        #plt.ylabel('Change in forward intensity')


    #fig = plt.figure(1)
    #fig.suptitle('Sensitivities - horizon 0')
    #ax = plt.subplot(211)
    #plot_sensitivities(comp0,'b',0.05)
    #ax.set_title('SPret')
    #ax = plt.subplot(212)
    #plot_sensitivities(comp1,'b',0.05)
    #ax.set_title('Tbill')
    my_dpi = 65
    plt.figure(figsize=(800 / my_dpi, 600 / my_dpi), dpi=my_dpi)
    plt.plot(np.linspace(-2,2,21),np.mean(comp0),'r')
    plt.plot(np.linspace(-2,2,21),np.mean(comp1),'darkorange')
    plt.plot(np.linspace(-2,2,21),np.mean(comp2),'g')
    plt.plot(np.linspace(-2,2,21),np.mean(comp3),'y')
    plt.plot(np.linspace(-2,2,21),np.mean(comp4),'o')
    plt.plot(np.linspace(-2,2,21),np.mean(comp5),'p')
    plt.plot(np.linspace(-2,2,21),np.mean(comp6),'c')
    plt.plot(np.linspace(-2,2,21),np.mean(comp7),'b')
    plt.plot(np.linspace(-2,2,21),np.mean(comp8),'v')
    plt.plot(np.linspace(-2,2,21),np.mean(comp9),'m')
    plt.plot(np.linspace(-2,2,21),np.mean(comp10),'yellow')
    plt.plot(np.linspace(-2,2,21),np.mean(comp11), 'k')
    #plt.legend(('SPret', 'Tbill', 'CASH/TA','NI/TA','Size','DtD','MBrat','D_CASH/TA','D_NI/TA','D_Size','D_DtD','D_MBrat'),
    #           loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol = 6)
    axes = plt.gca()
    axes.set_ylim([0, 5])
    plt.xlabel('Absolute change of variable')
    plt.ylabel('Average forward intensity')
    plt.title('Sensitivities')
    plt.show()

    plt.savefig(directory + str(tau) + '_corr.png')
    print('saving plot... ' + directory + str(tau) + '_corr.png')
    plt.close()

    model.sess.close()
    tf.reset_default_graph()


########################################## term structure ##############################################################

path_matlab = os.path.join('C:\\Users', 'divernoi', 'Dropbox', 'PhD', 'Duan Replication')

for exittype in ['surviving']:
    if exittype == 'surviving':
        cusip = 2817
    else:
        cusip = 1108


    termstruc_data = scipy.io.loadmat(os.path.join(path_matlab, 'termstruc_data' + str(cusip) + '.mat'))['data']

    intensitytype = 'f'
    hidden_dim = [5, 3]
    deltaT = 1 / 12
    learning_rate = 0.001
    feature_size = 12
    batch_size = 128
    perc = 0.9

    for z in [3, 12, 24, 36]:   # z months before bankruptcy

        termstructurecal = []
        termstructure = []
        in_data = termstruc_data[-z]
        directory = 'Results NN for Intensity estimation\\real dataset\\term structures\\' + str(hidden_dim) + '\\' + exittype + '\\' + str(cusip) + '\\'

        if not os.path.exists(directory):
            os.makedirs(directory)

        for tau in range(0,36):
            if intensitytype == 'f':
                x_train, y_train, x_test, y_test, _, _ = RealData_f(tau=tau)
            else:
                x_train, y_train, x_test, y_test, _, _ = RealData_h(tau=tau)

            _,_,_,_,mean_train,std_train = RealData_f(tau)

            stand_in_data = (in_data - mean_train) / std_train

            name = intensitytype + str(tau) + '_hidden' + str(hidden_dim[0]) + '_layers' + str(
                len(hidden_dim)) + '_learning' + str(learning_rate) + '_batch' + \
                   str(batch_size) + '_perc' + str(perc) + '_'
            path = intensitytype + str(tau) + '_hidden' + str(hidden_dim[0]) + '_layers' + str(
                len(hidden_dim)) + '_learning' + str(learning_rate) + '_batch' + \
                   str(batch_size) + '_perc' + str(perc) + '\\'

            model = NeuralNetwork(hidden_dim=hidden_dim, deltaT=deltaT, learning_rate=learning_rate,
                                  feature_size=feature_size, batch_size=batch_size, perc=perc, path=path, name=name)


            model.load(model.path + model.name + str(9))

            # calibrate intensity so that average probability outputted by model correspond to average defaults

            f = model.out_f(stand_in_data.reshape(1,-1))[0][0][0]

            probacal = (1 - np.exp(- model.out_f(stand_in_data.reshape(1,-1))[0][0][0] / 12)) / np.mean((1 - np.exp(- model.out_f(x_train)[0] / 12))) * np.mean(y_train)
            fcal = - 12 * np.log(1 - probacal)

            termstructurecal.append(fcal)
            termstructure.append(f)

            model.sess.close()
            tf.reset_default_graph()


        my_dpi = 115
        plt.figure(figsize=(1900 / my_dpi, 1200 / my_dpi), dpi=my_dpi)
        plt.plot(np.linspace(1,36,36),termstructurecal)
        plt.xlim([1.0, 36.0])
        if exittype == 'surviving':
            plt.ylim([0.0, .01])
        else:
            plt.ylim([0.0, .03])
        plt.legend('Term Structure', loc='upper right')
        plt.xlabel('Tau')
        plt.ylabel('Forward Intensity')
        plt.title('Model : ' + str(hidden_dim) + ' --  Term structure ' + str(z) + ' months before default or end of dataset')
        plt.show()

        plt.savefig(directory + str(z) + ' months before' + '.png')
        print('saving plot... ' + directory + str(tau) + '.png')
        plt.close()


        # save
        with open(directory + str(z) + ".txt", "w") as f:
            for s in termstructurecal:
                f.write(str(s) +"\n")


# load

my_dpi = 115
for z in [3, 12, 24, 36]:
    plt.figure(figsize=(1900 / my_dpi, 1200 / my_dpi), dpi=my_dpi)
    for exittype in ['surviving','defaulted']:

        if exittype == 'surviving':
            cusip = 1004
        else:
            cusip = 1108

        directory = 'Results NN for Intensity estimation\\real dataset\\term structures\\' + str(hidden_dim) + '\\' + exittype + '\\' + str(cusip) + '\\'



        tmp = []
        with open(directory + str(z) + ".txt", "r") as f:
          for line in f:
              tmp.append(float(line.strip()))

        plt.plot(np.linspace(1, 36, 36), tmp)
    plt.legend(('surviving','defaulted'), loc='upper right')
    plt.xlabel('Tau')
    plt.title('Term Structure ' + str(z) + ' months before dataset end or default')
    plt.ylabel('Forward Intensity')
    plt.show()

######################################## lorenz curve ##################################################################

## comparison lorenz with linear assumption (duan) and NN replic of duan (exp activation + layer [1])

# my model
intensitytype = 'f'
hidden_dim = [5, 3]
deltaT = 1 / 12
learning_rate = 0.001
feature_size = 12
batch_size = 256
perc = 0.9

# duan
path = os.path.join('C:\\Users', 'divernoi', 'Dropbox', 'PhD', 'Duan Replication')
param_duan = scipy.io.loadmat(os.path.join(path, 'paramduan.mat'))['paramduan']
testdata = hdf5storage.loadmat(os.path.join(path, 'testdata.mat'))['testdata']

# duan replic in NN
duan_replic = 'duan_replic'
directory_lorenz = 'Results NN for Intensity estimation\\real dataset\\lorenz\\' + intensitytype + '\\' + duan_replic + '\\'
with open(directory_lorenz + 'lorenz.pkl', 'rb') as f:
    lorenz_NNreplic = pickle.load(f)



directory = 'Results NN for Intensity estimation\\real dataset\\lorenz\\' + intensitytype + '\\' + str(hidden_dim) + '\\'

if not os.path.exists(directory):
    os.makedirs(directory)

all_gini = []

for tau in [0, 3, 6,12,24,35]:
    #compute NN res
    name = intensitytype + str(tau) + '_hidden' + str(hidden_dim[0]) + '_layers' + str(len(hidden_dim)) \
           + '_learning' + str(learning_rate) + '_batch' + str(batch_size) + '_perc' + str(perc) + '_'
    path = intensitytype + str(tau) + '_hidden' + str(hidden_dim[0]) + '_layers' + str(
        len(hidden_dim)) + '_learning' + str(learning_rate) + '_batch' + str(batch_size) + '_perc' + str(perc) + '\\'
    model = NeuralNetwork(hidden_dim=hidden_dim, deltaT=deltaT, learning_rate=learning_rate,
                          feature_size=feature_size, batch_size=batch_size, perc=perc, path=path, name=name)
    if intensitytype == 'f':
        x_train, y_train, x_test, y_test,_,_ = RealData_f(tau=tau)
    else:
        x_train, y_train, x_test, y_test,_,_ = RealData_h(tau=tau)

    model.load(model.path + model.name + str(19))

    percx, cumy, gini = LorenzCurve(model, x_test, y_test, tau, str(model.hidden_dim), color = 'green', path = directory, save = True)
    all_gini.append(gini)
    model.sess.close()
    tf.reset_default_graph()

    #compute res duan
    numpdata = testdata[tau]
    fullx = np.concatenate((numpdata[0], numpdata[2], numpdata[1]), axis=0)
    fully = np.concatenate((np.zeros(numpdata[0].shape[0] + numpdata[2].shape[0]), np.ones(numpdata[1].shape[0])), axis=0)
    f_duan = np.exp(np.dot(fullx, param_duan[:, tau]))
    res_duan = pd.DataFrame(data={'f': f_duan.reshape(f_duan.shape[0]), 'y': fully})
    res_duan['prob'] = 1-np.exp(-res_duan['f']/12)   # maybe play with the delta constant to cheat on AUC a bit
    res_duan = res_duan.sort_values('prob')
    res_duan['cumy'] = res_duan['y'].cumsum() / sum(res_duan['y'])
    res_duan['perc'] = list(range(1,len(res_duan)+1,1))
    res_duan['perc'] = res_duan['perc'] / len(res_duan)

    #plot all three curves for each horizon
    plt.figure()
    plt.plot(res_duan['perc'], res_duan['cumy'], color='darkorange', label='Linear')
    #plt.plot(lorenz_NNreplic[tau][0],lorenz_NNreplic[tau][1],color = 'green',label = 'NN+exp+[1]')
    plt.plot(percx,cumy,color = 'red',label = str(model.hidden_dim))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('Fraction of Population included')
    plt.ylabel('Fraction of Defaults included')
    plt.title('Lorenz Curve : horizon ' + str(tau))
    plt.legend(loc="upper left")
    plt.savefig('Results NN for Intensity estimation\\real dataset\\lorenz\\'+ str(tau) +'.png')
    print('saving lorenz curve... ' + model.name + '.png')


with open(directory + "gini.txt", "w") as f:
    for s in all_gini:
        f.write(str(s) + "\n")

###################################### compare NN_duan_replic with duan ################################################

# load NN_duan_replic

intensitytype = 'f'
hidden_dim = [1]
deltaT = 1 / 12
learning_rate = 0.001
feature_size = 12
batch_size = 256
perc = 0.9
duan_replic = "duan_replic"

name = intensitytype + str(tau) + '_hidden' + str(hidden_dim[0]) + '_layers' + str(len(hidden_dim)) + '_learning' + str(
    learning_rate) + '_batch' + \
       str(batch_size) + '_perc' + str(perc) + duan_replic + '_'  # remove
path = intensitytype + str(tau) + '_hidden' + str(hidden_dim[0]) + '_layers' + str(len(hidden_dim)) + '_learning' + str(
    learning_rate) + '_batch' + \
       str(batch_size) + '_perc' + str(perc) + duan_replic + '\\'  # remove


model = NeuralNetwork(hidden_dim=hidden_dim, deltaT=deltaT, learning_rate=learning_rate,
                          feature_size=feature_size, batch_size=batch_size, perc=perc, path=path, name=name)
model.load(model.path + model.name + str(19))

_,f_duanreplic = model.pred(x_test,y_test)

probacal_duanreplic = (1 - np.exp(- f_duanreplic / 12)) / np.mean((1 - np.exp(- model.out_f(x_train)[0] / 12))) * np.mean(y_train)
fcal_duanreplic = - 12 * np.log(1 - probacal)

# load duan


########################################### true proba #################################################################

monthsbeforeend = 0

#f
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

numpdata = testdata[monthsbeforeend]
fullx = np.concatenate((numpdata[0], numpdata[2], numpdata[1]), axis=0)
fully = np.concatenate((np.zeros(numpdata[0].shape[0] + numpdata[2].shape[0]), np.ones(numpdata[1].shape[0])), axis=0)

stdfullx,meanfullx,standfullx = standardize_data(fullx)

f_horizon = pd.DataFrame(index=pd.RangeIndex(start=0,stop=standfullx.shape[0],step=1))


for tau in range(0,35):
    # load model
    name = intensitytype + str(tau) + '_hidden' + str(hidden_dim[0]) + '_layers' + str(len(hidden_dim)) + '_learning' + str(
        learning_rate) + '_batch' + \
           str(batch_size) + '_perc' + str(perc) + duan_replic + '_'  # remove
    path = intensitytype + str(tau) + '_hidden' + str(hidden_dim[0]) + '_layers' + str(len(hidden_dim)) + '_learning' + str(
        learning_rate) + '_batch' + \
           str(batch_size) + '_perc' + str(perc) + duan_replic + '\\'  # remove
    model = NeuralNetwork(hidden_dim=hidden_dim, deltaT=deltaT, learning_rate=learning_rate,
                          feature_size=feature_size, batch_size=batch_size, perc=perc, path=path, name=name)
    model.load(model.path + model.name + str(999))

    #store forward default intensity of each horizon
    _,fmod = model.pred(standfullx,fully)
    f_horizon[str(tau)] = fmod
    model.sess.close()
    tf.reset_default_graph()


#h

intensitytype = 'h'
hidden_dim = [30, 15]
deltaT = 1 / 12
learning_rate = 0.001
feature_size = 12
batch_size = 256
perc = 0.9
duan_replic = ""

h_horizon = pd.DataFrame(index=pd.RangeIndex(start=0, stop=standfullx.shape[0], step=1))

for tau in range(0, 35):
    # load model
    name = intensitytype + str(tau) + '_hidden' + str(hidden_dim[0]) + '_layers' + str(
        len(hidden_dim)) + '_learning' + str(
        learning_rate) + '_batch' + \
           str(batch_size) + '_perc' + str(perc) + duan_replic + '_'  # remove
    path = intensitytype + str(tau) + '_hidden' + str(hidden_dim[0]) + '_layers' + str(
        len(hidden_dim)) + '_learning' + str(
        learning_rate) + '_batch' + \
           str(batch_size) + '_perc' + str(perc) + duan_replic + '\\'  # remove
    model = NeuralNetwork(hidden_dim=hidden_dim, deltaT=deltaT, learning_rate=learning_rate,
                          feature_size=feature_size, batch_size=batch_size, perc=perc, path=path, name=name)
    model.load(model.path + model.name + str(9))

    # store forward default intensity of each horizon
    _, hmod = model.pred(standfullx, fully)
    h_horizon[str(tau)] = hmod
    model.sess.close()
    tf.reset_default_graph()


# compute cumulative def probability

taupred = 0
deltaT = 1/12

proba = pd.DataFrame(index=pd.RangeIndex(start=0, stop=standfullx.shape[0], step=1))


proba[str(taupred)] =  1 - np.exp(-f_horizon[str(taupred)]*deltaT)
proba[str(1)] =  np.exp((-f_horizon[str(0)]-h_horizon[str(0)])*deltaT) * ( 1 - np.exp(-f_horizon[str(1)]*deltaT))



################################### summary statistics #################################################################

path = os.path.join('C:\\Users', 'divernoi', 'Dropbox', 'PhD', 'Duan Replication')
tau = 1
fulldata = hdf5storage.loadmat(os.path.join(path, 'fulldata.mat'))
numpdata = np.array(list(fulldata.items()))[0][1][tau]

param_name = [
 'constant'
,'SPret'
,'Tbill'
,'Cash/TA'
,'NI/TA'
,'Size'
,'DtD'
,'MBratio'
,' Δ CASH/TA'
,'Δ NI/TA'
,'Δ Size'
,'Δ DtD'
,'Δ MBratio']

sumstat = pd.DataFrame(data = {'Surviving' : np.mean(numpdata[0],axis=0), 'Default':np.mean(numpdata[1],axis=0) , 'Other exits':np.mean(numpdata[2],axis=0)} , index = param_name)

# corrmat

sns.set(style="white")
df = pd.DataFrame(data=np.concatenate((numpdata[0], numpdata[2], numpdata[1]), axis=0)[:,1:],    # values
             index=pd.RangeIndex(start=0, stop=numpdata[0].shape[0] + numpdata[2].shape[0] + numpdata[1].shape[0], step=1))
df.columns = param_name[1:]

corr = df.corr()
# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.5})





################################ xaxis : tau , yaxis : gini ############################################################

gini_duan = []
gini_1 = []
gini_21 = []
gini_32 = []
gini_53 = []
gini_105 = []

# load
with open("Results NN for Intensity estimation\\real dataset\\lorenz\\f\\duan\\gini.txt", "r") as f:
  for line in f:
      gini_duan.append(float(line.strip()))

with open("Results NN for Intensity estimation\\real dataset\\lorenz\\f\\[1]\\gini.txt", "r") as f:
  for line in f:
      gini_1.append(float(line.strip()))

with open("Results NN for Intensity estimation\\real dataset\\lorenz\\f\\[2, 1]\\gini.txt", "r") as f:
  for line in f:
      gini_21.append(float(line.strip()))

with open("Results NN for Intensity estimation\\real dataset\\lorenz\\f\\[3, 2]\\gini.txt", "r") as f:
    for line in f:
      gini_32.append(float(line.strip()))

with open("Results NN for Intensity estimation\\real dataset\\lorenz\\f\\[5, 3]\\gini.txt", "r") as f:
  for line in f:
      gini_53.append(float(line.strip()))

with open("Results NN for Intensity estimation\\real dataset\\lorenz\\f\\[10, 5]\\gini.txt", "r") as f:
  for line in f:
      gini_105.append(float(line.strip()))

plt.figure()
lw = 2
tau = np.linspace(1,36,36)
plt.plot(tau, gini_duan, color='darkorange',
         lw=lw, label='Linear')
plt.plot(tau, gini_1, color='black',
         lw=lw, label='[1]')
plt.plot(tau, gini_21, color='blue',
         lw=lw, label='[2, 1]')
plt.plot(tau, gini_32, color='red',
         lw=lw, label='[3, 2]')
plt.plot(tau, gini_53, color='green',
         lw=lw, label='[5, 3]')
plt.plot(tau, gini_105, color='lightblue',
         lw=lw, label='[10, 5]')
plt.xlim([1.0, 36.0])
plt.ylim([0, 1.0])
plt.xlabel('Tau')
plt.ylabel('Gini coefficient')
plt.title('Gini -- Linear vs NeuralNetwork')
plt.legend(loc="lower right")

