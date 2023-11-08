import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from sklearn.metrics import roc_curve, auc
from scipy import interp


def ROC_plot(score_all, label_all):
    num_class = 20
    score_array = score_all.numpy()
    # turn label to noise
    label_tensor = label_all.to(torch.int64)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], num_class)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)

    # cal fpr amd tpr with sklearn.
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in range(num_class):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
    # micro
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_class)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
    # Finally average it and compute AUC
    mean_tpr /= num_class
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

    # roc-curve.
    lw = 2
    plt.plot(fpr_dict["micro"], tpr_dict["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr_dict["macro"], tpr_dict["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["macro"]),
             color='navy', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim([0.0, 1.0])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Receiver Operating Characteristic to Multi-class', fontsize=16)
    plt.rcParams.update({'font.size': 18})
    plt.legend(loc="lower right")


# plot confusion matrix
class DrawConfusionMatrix:
    def __init__(self, labels_name, normalize=True):
        """
		normalize:percentage or not
        """
        self.normalize = normalize
        self.labels_name = labels_name
        self.num_classes = len(labels_name)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype="float32")

    def update(self, predicts, labels):
        """

        :param predicts: eg:array([0,5,1,6,3,...],dtype=int64)
        :param labels:   eg:array([0,5,0,6,2,...],dtype=int64)
        :return:
        """
        for predict, label in zip(predicts, labels):
            self.matrix[predict, label] += 1

    def getMatrix(self, normalize=True):
        """
        If normalize is true, the matrix elements are converted to percentage form,
        If normalize is false, the matrix elements are numbers
        Returns: returns a matrix whose elements are percentage or quantity
        """

        if normalize:
            per_sum = self.matrix.sum(axis=1)  # sum of every row
            for i in range(self.num_classes):
                self.matrix[i] = (self.matrix[i] / per_sum[i])  # convert to percentage form
            self.matrix = np.around(self.matrix, 2)  # keep 2 decimal places
            self.matrix[np.isnan(self.matrix)] = 0  # change NaN to 0
        return self.matrix

    def drawMatrix(self):
        id = [4, 15, 16, 0, 7, 14, 3, 6, 5, 2, 8, 1, 11, 12, 9, 10, 19, 17, 18, 13]
        self.matrix = self.getMatrix(self.normalize)
        M = self.matrix.copy()

        # substitution
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                self.matrix[y, x] = M[id[y], id[x]]

        plt.imshow(self.matrix, cmap=plt.cm.Blues, aspect='auto')  # paint boxes without text
        plt.title("Normalized Confusion Matrix", fontsize=16)  # title
        plt.xlabel("Predict label", fontsize=10)
        plt.ylabel("True label", fontsize=10)
        plt.yticks(range(self.num_classes), self.labels_name, fontsize=10)
        plt.xticks(range(self.num_classes), self.labels_name, fontsize=10)

        # split line
        z = [-0.5, -0.5 + self.num_classes]
        z1 = [0.5, 0.5]
        z2 = [5.5, 5.5]
        z3 = [9.5, 9.5]
        z4 = [12.5, 12.5]
        z5 = [16.5, 16.5]

        plt.plot(z1, z, z2, z, z3, z, z4, z, z5, z, c='r', linewidth=0.5)
        plt.plot(z, z1, z, z2, z, z3, z, z4, z, z5, c='r', linewidth=0.5)

        for x in range(self.num_classes):
            for y in range(self.num_classes):
                value = float(format('%.2f' % self.matrix[y, x]))
                plt.text(x, y, value, verticalalignment='center', horizontalalignment='center',
                         fontsize=6)  # write value

        plt.tight_layout()  # automatically adjust the sub graph parameters to fill the entire image area

        cb = plt.colorbar()  # color bar on the right
        cb.ax.tick_params(labelsize=12)


def plot_model(epoch, train_loss,loss_cla,loss_sas,loss_bfa,loss_dih,loss_cor,val_loss, cormultip_mean, filename):
    plt.figure(figsize=(15, 15))
    plt.figure(1)

    ax1 = plt.subplot(221)
    plt.plot(train_loss, label='train_loss')
    plt.plot(loss_cla, label='classification loss')
    plt.plot(loss_sas, label='sasa loss')
    plt.plot(loss_bfa, label='b-factor loss')
    plt.plot(loss_dih, label='dihedral loss')
    plt.plot(loss_cor, label='coor loss')
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if epoch < 100:
        plt.xlim((0, 101))
    else:
        plt.xlim((0, 500))
    plt.ylim((-0.2, 3.2))
    plt.rcParams.update({'font.size': 18})
    plt.legend()

    ax2 = plt.subplot(222)
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss,label = 'val loss')
    plt.plot(cormultip_mean, label='multi_corr')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if epoch < 500:
        plt.xlim((0, 501))
    else:
        plt.xlim((0, 5001))
    plt.ylim((-0.2, 3.2))   
    plt.legend()
    plt.title(
        f'best multiple mutation: {max(cormultip_mean):.4f}',
        fontsize=16)
#     ROC_plot(score_all, label_all)
    os.makedirs('result/figure/',exist_ok=True)
    plt.savefig('result/figure/' + filename + '.png', bbox_inches='tight', dpi=400)
    plt.close()

