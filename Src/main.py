# own modules import
import pandas as pd
import numpy as np

from Src.DataImport import dataImport
from Src.DataPreprocessing import dropColumns
from Src.DataPreprocessing import dropColumnsWithNAs
from Src.DataPreprocessing import dropRowsWithNAs
from Src.FeatureEngineering import calculate_woe_iv
from Src.FeatureEngineering import dropFeaturesLowIV
from Src.FeatureEngineering import replaceCategoricalValues
from Src.FeatureEngineering import showHeatMap
from Src.FeatureEngineering import dropFeaturesHighlyCorr

#%% import data
dir = "C:/Users/Albert Nietz/PyCharm_Projects/Credit_Risk_Management_Project/Data"
data = dataImport(dir)
data = data.replace('XNA', np.nan)

#%% drop features that are not related to credit risk
columns = ['SK_ID_CURR']
data = dropColumns(data, columns)

#%% drop columns with more than 50% na's
data = dropColumnsWithNAs(data, 0.5)

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
data_features_woe = dropFeaturesHighlyCorr(data_features_woe, 0.3)
showHeatMap(data_features_woe)

#%% add to features_woe dataframe the target column
data_features_woe['TARGET'] = data_preprocessed['TARGET']







































