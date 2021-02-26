from model_class import *
from data_creation import *
import pickle
######################################### LEARNING PROCESS #############################################################

all_fpr = []
all_tpr = []
all_auc = []
lorenz = []
all_gini = []

# loop to train models
for intensitytype in ('f'):    # in ('f','h')

    if intensitytype == 'f':
        hidden_dim = [5, 3]
    else:
        hidden_dim = [30, 15]

    duan_replic = ""   #duan_replic : full duan replic || #exp : exp activation function

    # plot directory
    directory_roc = 'Results NN for Intensity estimation\\real dataset\\roc\\' + intensitytype + '\\'  + str(hidden_dim) + '\\'    #add back
    #directory_roc = 'Results NN for Intensity estimation\\real dataset\\roc\\' + intensitytype + '\\' + duan_replic + '\\'          #remove
    if not os.path.exists(directory_roc):
        os.makedirs(directory_roc)

    directory_lorenz = 'Results NN for Intensity estimation\\real dataset\\lorenz\\' + intensitytype + '\\'  + str(hidden_dim) + '\\'    #add back
    # directory_lorenz = 'Results NN for Intensity estimation\\real dataset\\lorenz\\' + intensitytype + '\\' + duan_replic + '\\'          #remove
    if not os.path.exists(directory_lorenz):
        os.makedirs(directory_lorenz)

    for tau in range(36):
        print('Estimating model ... ' + intensitytype + str(tau))
        deltaT = 1/12
        learning_rate=0.001
        feature_size = 12
        batch_size = 256
        perc = 0.9
        name = intensitytype + str(tau) + '_hidden' + str(hidden_dim[0]) + '_layers' + str(len(hidden_dim)) + '_learning' + str(learning_rate) + '_batch' + \
               str(batch_size) + '_perc' + str(perc) + '_'                   #add back
        path = intensitytype + str(tau) + '_hidden' + str(hidden_dim[0]) + '_layers' + str(len(hidden_dim))  + '_learning' + str(learning_rate) + '_batch' + \
               str(batch_size) + '_perc' + str(perc) + '\\'                  #add back
        #name = intensitytype + str(tau) + '_hidden' + str(hidden_dim[0]) + '_layers' + str(len(hidden_dim)) + '_learning' + str(learning_rate) + '_batch' + \
        #       str(batch_size) + '_perc' + str(perc) + duan_replic + '_'     #remove
        #path = intensitytype + str(tau) + '_hidden' + str(hidden_dim[0]) + '_layers' + str(len(hidden_dim))  + '_learning' + str(learning_rate) + '_batch' + \
        #       str(batch_size) + '_perc' + str(perc) + duan_replic + '\\'    #remove

        if intensitytype == 'f':
            x_train,y_train,x_test,y_test,_,_ = RealData_f(tau=tau)
        else:
            x_train,y_train,x_test,y_test,_,_ = RealData_h(tau=tau)


        model = NeuralNetwork(hidden_dim = hidden_dim, deltaT = deltaT, learning_rate = learning_rate,
                              feature_size = feature_size, batch_size = batch_size, perc = perc, path = path,name = name)
        self = model

        # trick to avoid bad initialization (output of first forward propagation algorithm too close to boundary)
        model.initialise()
        _ , ff = model.pred(x_train,y_train)
        gg = 0.00001*np.ones(ff.shape)
        bad_init = np.allclose(ff,gg)  # test if predictions are close to fmin boundary. if yes, initialize again
        if bad_init is True:
            while bad_init is True:
                model.initialise()
                _, ff = model.pred(x_train, y_train)
                gg = 0.00001 * np.ones(ff.shape)
                bad_init = np.allclose(ff, gg)

        # training process
        for e in range(20):
            model.training(x_train,y_train,e)
            in_loss_value, in_f_value = model.pred(x_train,y_train)
            out_loss_value, out_f_value  = model.pred_and_write_summary(x_test,y_test,e)
            print('insample-loss:', in_loss_value, '// outsample-loss:',out_loss_value)

        #_,_,auc_score, fpr_val, tpr_val = testmodel(model = model, x = x_test,y = y_test, path = directory_roc, save=True)  # computes auc score and save roc curve to directory
        perc, cumy, gini = LorenzCurve(model, x_test, y_test, tau, str(model.hidden_dim), color='green', path=directory_lorenz, save=True)

        #all_auc.append(auc_score)    # store auc_score for this particular model
        #all_fpr.append(fpr_val)  # store auc_score for this particular model
        #all_tpr.append(tpr_val)  # store auc_score for this particular model
        lorenz.append((perc,cumy))
        all_gini.append(gini)

        tf.reset_default_graph()
        model.sess.close()

    with open(directory_lorenz + "gini.txt", "w") as f:
        for s in all_gini:
            f.write(str(s) + "\n")

    with open(directory_lorenz + 'lorenz.pkl', 'wb') as f:
        pickle.dump(lorenz, f)

    all_auc = []
    lorenz = []








