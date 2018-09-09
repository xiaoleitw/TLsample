

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from keras import backend as K


def plot_history(history, model, figname, n=32, m=2):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    # print(len(loss_list))
    #logstr = str(m) + '\t' + str(n) + '\t' + '\t' + str(timeval) + '\t'

    trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    logstr =  str(trainable_count) + '\t' + str(non_trainable_count) + '\t'

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(n+m)
    plt.subplot(211)

    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
        logstr = logstr + str(format(history.history[l][-1], '.5f')) + '\t'

    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
        logstr = logstr + str(format(history.history[l][-1], '.5f')) + '\t'

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    ## Accuracy
    #plt.figure()
    plt.subplot(212)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
        logstr = logstr + str(format(history.history[l][-1], '.5f')) + '\t'

    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
        logstr = logstr + str(format(history.history[l][-1], '.5f')) + '\t'

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    #plt.show()
    #figname = 'LossAcc.png'
    plt.savefig(figname)
    print("History Saved...", figname)

    with open("Performance-ACC-Log.txt", "a") as myfile:
        myfile.write(logstr+'\n')
    return figname