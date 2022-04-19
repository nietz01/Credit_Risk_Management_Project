import numpy as np
import pandas as pd
from Src.prompt import prompt

def dropColumns(data, columns):

    data = data.drop(columns=columns)

    return data

def dropColumnsWithNAs(data, thresh):

    data = data.dropna(thresh=data.shape[0]*thresh, how='all', axis=1)

    return data

def convertCategoricalData(data):

    cat_columns = data.select_dtypes(['category']).columns
    data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes.astype(pd.Int64Dtype()))

    data[cat_columns] = data[cat_columns].replace(-1, np.nan)

    return data


def dropRowsWithNAs(data):

    print(data.isna().sum())

    rowwiseIsNA = np.vectorize(pd.isna)
    rowwiseNA = rowwiseIsNA(data)
    numRemovedRows = sum([any(row) for row in rowwiseNA])
    prompt(f"There will be {numRemovedRows} rows removed")

    data = data.dropna(axis=0)

    return data