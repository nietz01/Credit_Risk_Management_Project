import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression

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

def dropFeaturesLowIV(data_preprocessed, df_iv, thresh):

    columnsLowIV = df_iv[df_iv['IV'] < thresh]['Variable'].tolist()
    print(columnsLowIV)
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

def showHeatMap(data_preprocessed):

    plt.figure(figsize=(16, 16))
    sb.heatmap(data_preprocessed.corr(), annot=True, cmap=plt.cm.Reds)
    plt.show()

def dropFeaturesHighlyCorr(data_preprocessed, thresh):

    corr_matrix = data_preprocessed.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > thresh)]
    print(to_drop)
    data_preprocessed = data_preprocessed.drop(to_drop, axis=1)

    return data_preprocessed

def forwardFeatureSelection(X, y, thresh):

    logit = LogisticRegression()
    sfs = SequentialFeatureSelector(logit, n_features_to_select=thresh)
    sfs.fit(X, y)

    X_new = sfs.transform(X)
    retained_feat = sfs.get_feature_names_out()
    X_new_df = pd.DataFrame(X_new, columns=retained_feat)

    to_drop = X.drop(retained_feat, axis=1)
    print(list(to_drop.columns))

    return X_new_df









