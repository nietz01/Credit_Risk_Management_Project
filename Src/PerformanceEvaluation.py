import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import pandas as pd

def calculateGiniCoefficient(y_true, y_pred_proba):

    auc = roc_auc_score(y_true, y_pred_proba)
    gini_coeff = 2 * auc - 1

    return gini_coeff

def calculateConfusionMatrix(y_true, y_pred, freq=True):

    labels = np.unique(y_true)
    conf = confusion_matrix(y_true, y_pred, labels=labels)

    if freq:
        conf = conf / conf.flatten().sum()

    conf = pd.DataFrame(conf, columns=['Predicted Good', 'Predicted Bad'], index=['Actual Good', 'Actual Bad'])
    return conf




