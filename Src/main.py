# own modules import
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from Src.DataImport import dataImport
from Src.DataPreprocessing import dropColumns
from Src.DataPreprocessing import dropColumnsWithNAs
from Src.DataPreprocessing import dropRowsWithNAs

from Src.FeatureEngineering import calculate_woe_iv
from Src.FeatureEngineering import dropFeaturesLowIV
from Src.FeatureEngineering import replaceCategoricalValues
from Src.FeatureEngineering import showHeatMap
from Src.FeatureEngineering import dropFeaturesHighlyCorr
from Src.FeatureEngineering import forwardFeatureSelection

from Src.LinearClassifier import linearClassifier

from Src.PerformanceEvaluation import calculateGiniCoefficient
from Src.PerformanceEvaluation import calculateConfusionMatrix

#%% import data
dir = "C:/Users/Albert Nietz/PyCharm_Projects/Credit_Risk_Management_Project/Data"
data = dataImport(dir)
data = data.replace('XNA', np.nan)

#%% drop features that are not related to credit risk
columns = ['SK_ID_CURR']
data = dropColumns(data, columns)

#%% drop columns with more than 50% na's
data = dropColumnsWithNAs(data, 0.6)

#%% drop rows with na's
data_preprocessed = dropRowsWithNAs(data)

#%% transform numerical variables into bins & caluclate WoE and IV for all independent variables
df_iv, df_woe = calculate_woe_iv(data_preprocessed, 'TARGET', 10)

#%% drop features wth low IV
data_preprocessed = dropFeaturesLowIV(data_preprocessed, df_iv, 0.02)

#%% replace categorical values and bins by the corresponding WoE / "WoEization"
data_features = data_preprocessed.drop('TARGET', axis=1)
data_features_woe = replaceCategoricalValues(data_features, df_woe)

#%% perform correlation analysis & drop out highly correlated features (> 50%)
showHeatMap(data_features_woe)
data_features_woe = dropFeaturesHighlyCorr(data_features_woe, 0.50)
showHeatMap(data_features_woe)

#%% perform forward (feature) selection procedure
data_features_woe = forwardFeatureSelection(data_features_woe, data_preprocessed['TARGET'], 10)

#%% preparations for model development & evaluation
X = data_features_woe
y = data_preprocessed['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(y_train.value_counts())

#%% perform linear classifier (logistic regression, logit)
y_pred_proba_train, y_pred_train, y_pred_proba_test, y_pred_test = linearClassifier(X_train, X_test, y_train)

#%% performance evaluation (gini coefficient (AR) & confusion matrix)
# in-sample
gini_coeff_train = calculateGiniCoefficient(y_train, y_pred_proba_train)
conf_matrix_train = calculateConfusionMatrix(y_train, y_pred_train, freq=False)
# out-of-sample
gini_coeff_test = calculateGiniCoefficient(y_test, y_pred_proba_test)
conf_matrix_test = calculateConfusionMatrix(y_test, y_pred_test, freq=False)










































