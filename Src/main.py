# own modules import
from Src.DataImport import dataImport
from Src.DataPreprocessing import dropColumns
from Src.DataPreprocessing import dropColumnsWithNAs
from Src.DataPreprocessing import convertCategoricalData
import numpy as np

#%% import data
dir = "C:/Users/Albert Nietz/PyCharm_Projects/Credit_Risk_Management_Project/Data"
data = dataImport(dir)

#%% drop features that are not related to credit risk
columns = ['SK_ID_CURR']
data = dropColumns(data, columns)

#%% drop columns with more than 50% null values
data = dropColumnsWithNAs(data, 0.5)

#%% convert categorical variables to numerical data
data = convertCategoricalData(data)







