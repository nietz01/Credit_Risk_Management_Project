import numpy as np
import pandas as pd

def calculate_woe_iv(data, target, bins):

    # empty dataframe
    newDF, woeDF = pd.DataFrame(), pd.DataFrame()

    # extract column names
    cols = data.columns

    # calculate WOE and IV for all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars])) > 10):
            binned_x = pd.qcut(data[ivars], bins, duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Bad']
        d['% of Bad'] = np.maximum(d['Bad'], 0.5) / d['Bad'].sum()
        d['Good'] = d['N'] - d['Bad']
        d['% of Good'] = np.maximum(d['Good'], 0.5) / d['Good'].sum()
        d['WoE'] = np.log(d['% of Good'] / d['% of Bad'])
        d['IV'] = d['WoE'] * (d['% of Good'] - d['% of Bad'])
        d.insert(loc=0, column='Variable', value=ivars)

        temp = pd.DataFrame({"Variable": [ivars], "IV": [d['IV'].sum()]}, columns=["Variable", "IV"])
        newDF = pd.concat([newDF, temp], axis=0)
        woeDF = pd.concat([woeDF, d], axis=0)

    return newDF, woeDF

def dropColumnsLowIV(data_preprocessed, df_iv, thresh):

    columnsLowIV = df_iv[df_iv['IV'] < thresh]['Variable'].tolist()
    data = data_preprocessed.drop(columnsLowIV, axis=1)

    return data

def replaceCategoricalValues(features, df_woe):

    features = features.copy()
    column_names = features.columns.values.tolist()

    for ivars in column_names:

        if features[ivars].dtype.kind in 'bifc':

            cond = (df_woe['Variable']==ivars)
            bins = df_woe[cond]['Cutoff'].tolist()

            for jvars in bins:

                cond = (df_woe['Variable']==ivars) & (df_woe['Cutoff']==jvars)
                woe = df_woe[cond].WoE.values[0]

                if isinstance(jvars, int):

                    features[ivars] = np.where((features[ivars]==jvars), woe, features[ivars])

                else:

                    features[ivars] = np.where((features[ivars] > jvars.left) & (features[ivars] <= jvars.right),woe,features[ivars])
        else:

            for jvars in features[ivars].unique():

                cond = (df_woe['Variable']==ivars) & (df_woe['Cutoff']==jvars)
                woe = df_woe[cond].WoE.values[0]

                features[ivars] = np.where((features[ivars]==jvars),woe,features[ivars])

    features = features.apply(pd.to_numeric)
    return features





