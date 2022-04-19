import numpy as np
import os
import pandas as pd

def dataImport(dir):

    dtypes = {
        'SK_ID_CURR': np.int64,
        'TARGET': np.int64,
        'NAME_CONTRACT_TYPE': np.str,
        'CODE_GENDER': np.str,
        'FLAG_OWN_CAR': np.str,
        'FLAG_OWN_REALTY': np.str,
        'CNT_CHILDREN': np.int64,
        'AMT_INCOME_TOTAL': np.float64,
        'AMT_CREDIT': np.float64,
        'AMT_ANNUITY': np.float64,
        'AMT_GOODS_PRICE': np.float64,
        'NAME_TYPE_SUITE': np.str,
        'NAME_INCOME_TYPE': np.str,
        'NAME_EDUCATION_TYPE': np.str,
        'NAME_FAMILY_STATUS': np.str,
        'NAME_HOUSING_TYPE': np.str,
        'REGION_POPULATION_RELATIVE': np.float64,
        'DAYS_BIRTH': np.int64,
        'DAYS_EMPLOYED': np.int64,
        'DAYS_REGISTRATION': np.int64,
        'DAYS_ID_PUBLISH': np.int64,
        'OWN_CAR_AGE': pd.Int64Dtype(),
        'FLAG_MOBIL': np.int64,
        'FLAG_EMP_PHONE': np.int64,
        'FLAG_WORK_PHONE': np.int64,
        'FLAG_CONT_MOBILE': np.int64,
        'FLAG_PHONE': np.int64,
        'FLAG_EMAIL': np.int64,
        'OCCUPATION_TYPE': np.str,
        'CNT_FAM_MEMBERS': np.int64,
        'REGION_RATING_CLIENT': np.int64,
        'REGION_RATING_CLIENT_W_CITY': np.int64,
        'WEEKDAY_APPR_PROCESS_START': np.str,
        'HOUR_APPR_PROCESS_START': np.int64,
        'REG_REGION_NOT_LIVE_REGION': np.int64,
        'REG_REGION_NOT_WORK_REGION': np.int64,
        'LIVE_REGION_NOT_WORK_REGION': np.int64,
        'REG_CITY_NOT_LIVE_CITY': np.int64,
        'REG_CITY_NOT_WORK_CITY': np.int64,
        'LIVE_CITY_NOT_WORK_CITY': np.int64,
        'ORGANIZATION_TYPE': np.str,
        'EXT_SOURCE_2': np.float64,
        'EXT_SOURCE_1': np.float64,
        'APARTMENTS_AVG': np.float64,
        'COMMONAREA_AVG': np.float64,
        'LIVINGAREA_AVG': np.float64,
        'NONLIVINGAREA_AVG': np.float64,
        'HOUSETYPE_MODE': np.str,
        'WALLSMATERIAL_MODE': np.str,
        'EMERGENCYSTATE_MODE': np.str,
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