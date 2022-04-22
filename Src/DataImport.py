import numpy as np
import pandas as pd
import os

def dataImport(dir):

    dtypes = {
        'SK_ID_CURR': np.int64,
        'TARGET': np.int64,
        'NAME_CONTRACT_TYPE': pd.CategoricalDtype(),
        'CODE_GENDER': pd.CategoricalDtype(),
        'FLAG_OWN_CAR': pd.CategoricalDtype(),
        'FLAG_OWN_REALTY': pd.CategoricalDtype(),
        'CNT_CHILDREN': np.int64,
        'AMT_INCOME_TOTAL': np.float64,
        'AMT_CREDIT': np.float64,
        'AMT_ANNUITY': np.float64,
        'AMT_GOODS_PRICE': np.float64,
        'NAME_TYPE_SUITE': pd.CategoricalDtype(),
        'NAME_INCOME_TYPE': pd.CategoricalDtype(),
        'NAME_EDUCATION_TYPE': pd.CategoricalDtype(),
        'NAME_FAMILY_STATUS': pd.CategoricalDtype(),
        'NAME_HOUSING_TYPE': pd.CategoricalDtype(),
        'REGION_POPULATION_RELATIVE': np.float64,
        'DAYS_BIRTH': np.int64,
        'DAYS_EMPLOYED': np.int64,
        'DAYS_REGISTRATION': np.int64,
        'DAYS_ID_PUBLISH': np.int64,
        'OWN_CAR_AGE': pd.Int64Dtype(),
        'FLAG_MOBIL': pd.CategoricalDtype(),
        'FLAG_EMP_PHONE': pd.CategoricalDtype(),
        'FLAG_WORK_PHONE': pd.CategoricalDtype(),
        'FLAG_CONT_MOBILE': pd.CategoricalDtype(),
        'FLAG_PHONE': pd.CategoricalDtype(),
        'FLAG_EMAIL': pd.CategoricalDtype(),
        'OCCUPATION_TYPE': pd.CategoricalDtype(),
        'CNT_FAM_MEMBERS': np.int64,
        'REGION_RATING_CLIENT': pd.CategoricalDtype(),
        'REGION_RATING_CLIENT_W_CITY': pd.CategoricalDtype(),
        'WEEKDAY_APPR_PROCESS_START': pd.CategoricalDtype(),
        'HOUR_APPR_PROCESS_START': np.int64,
        'REG_REGION_NOT_LIVE_REGION': pd.CategoricalDtype(),
        'REG_REGION_NOT_WORK_REGION': pd.CategoricalDtype(),
        'LIVE_REGION_NOT_WORK_REGION': pd.CategoricalDtype(),
        'REG_CITY_NOT_LIVE_CITY': pd.CategoricalDtype(),
        'REG_CITY_NOT_WORK_CITY': pd.CategoricalDtype(),
        'LIVE_CITY_NOT_WORK_CITY': pd.CategoricalDtype(),
        'ORGANIZATION_TYPE': pd.CategoricalDtype(),
        'EXT_SOURCE_2': np.float64,
        'EXT_SOURCE_1': np.float64,
        'APARTMENTS_AVG': np.float64,
        'COMMONAREA_AVG': np.float64,
        'LIVINGAREA_AVG': np.float64,
        'NONLIVINGAREA_AVG': np.float64,
        'HOUSETYPE_MODE': pd.CategoricalDtype(),
        'WALLSMATERIAL_MODE': pd.CategoricalDtype(),
        'EMERGENCYSTATE_MODE': pd.CategoricalDtype(),
        'OBS_30_CNT_SOCIAL_CIRCLE': pd.Int64Dtype(),
        'DEF_30_CNT_SOCIAL_CIRCLE': pd.Int64Dtype(),
        'OBS_60_CNT_SOCIAL_CIRCLE': pd.Int64Dtype(),
        'DEF_60_CNT_SOCIAL_CIRCLE': pd.Int64Dtype(),
        'DAYS_LAST_PHONE_CHANGE': pd.Int64Dtype(),
        'AMT_REQ_CREDIT_BUREAU_HOUR': pd.Int64Dtype(),
        'AMT_REQ_CREDIT_BUREAU_DAY': pd.Int64Dtype(),
        'AMT_REQ_CREDIT_BUREAU_WEEK': pd.Int64Dtype(),
        'AMT_REQ_CREDIT_BUREAU_MON': pd.Int64Dtype(),
        'AMT_REQ_CREDIT_BUREAU_QRT': pd.Int64Dtype(),
        'AMT_REQ_CREDIT_BUREAU_YEAR': pd.Int64Dtype()
    }

    file_path = os.path.join(dir, 'AppData.csv')
    data = pd.read_csv(file_path, dtype=dtypes, usecols=list(dtypes.keys()))

    return data