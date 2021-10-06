import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .custom_scaler import CustomStandardScaler
from src.entities.feature_params import FeatureParams


def build_numerical_pipeline_w_scaler() -> Pipeline:
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("custom_scaler", CustomStandardScaler()),
        ]
    )
    return num_pipeline


def build_numerical_pipeline_wo_scaler() -> Pipeline:
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean"))
        ]
    )
    return num_pipeline


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="if_binary")),
        ]
    )
    return categorical_pipeline


def build_transformer(params: FeatureParams) -> ColumnTransformer:

    if params.normalize_numerical:
        transformer = ColumnTransformer(
            [
                (
                    "categorical_pipeline",
                    build_categorical_pipeline(),
                    params.categorical_features,
                ),
                (
                    "numerical_pipeline",
                    build_numerical_pipeline_w_scaler(),
                    params.numerical_features,
                ),
            ]
        )
    else:
        transformer = ColumnTransformer(
            [
                (
                    "categorical_pipeline",
                    build_categorical_pipeline(),
                    params.categorical_features,
                ),
                (
                    "numerical_pipeline",
                    build_numerical_pipeline_wo_scaler(),
                    params.numerical_features,
                ),
            ]
        )

    return transformer


def featurize(df_train):
    U1 = df_train['url'].str.split('//', n=-1, expand=True)[1]
    U2 = U1.str.split('www.', n=-1, expand=True)[1]
    webname = U2.str.split('.', n=-1, expand=True)[0]
    U3 = U2.str.split('.', n=-1, expand=True)[1]
    domain = U3.str.split('/', n=-1, expand=True)[0]
    website_type = U3.str.split('/', n=-1, expand=True)[1]
    U4 = U3.str.split('/', n=-1, expand=True)[2]
    website_type2 = U4.str.split('/', n=-1, expand=True)[0]

    ###================= Categorical Features out of this ======
    df_train["website"] = webname
    df_train["website_type"] = website_type
    df_train["website_type2"] = website_type2
    df_train["domain"] = domain

    # ============== Other Features ==================================
    df_train['alchemy_category_score'] = pd.to_numeric(df_train['alchemy_category_score'], errors='coerce')
    df_train["is_news"] = pd.to_numeric(df_train["is_news"], errors='coerce')
    df_train["news_front_page"] = pd.to_numeric(df_train["news_front_page"], errors='coerce')

    return df_train


def prepare_dataset(df: pd.DataFrame, params: FeatureParams) -> pd.Series:

    df_train = featurize(df)


    df_train['website_type'] = df_train['website_type'].replace({'2007': 'YEAR', '2008': 'YEAR',
                                                                 '2009': 'YEAR', '2010': 'YEAR',
                                                                 '2011': 'YEAR', '2012': 'YEAR',
                                                                 '2013': 'YEAR'})

    columns = [i for i in df_train.columns if i not in ['url', 'urlid', 'boilerplate', 'label']]

    cat_variable = []
    k = 0
    for i in df_train[columns].dtypes:
        if i == 'O':
            cat_variable.append(k)
        k = k + 1

    #for i in cat_variable:
    #    S = set(df_train[columns[i]])
    #    for k in S:
    #        df_train.loc[columns[i]] = df_train[columns[i]].replace([k, '<', np.nan], 'NaN')

    df_train.fillna(0, inplace=True)

    #TODO убрать костыль
    return df_train[list(set(columns) - set(['alchemy_category', 'website', 'website_type', 'website_type2', 'domain']))]


def get_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    return df[params.target_col]

