# Own modules import
from Src.DataImport import dataImport

# Data import
dir = "C:/Users/Albert Nietz/PyCharm_Projects/Credit_Risk_Management_Project/Data"
data = dataImport(dir)

print(data.head())