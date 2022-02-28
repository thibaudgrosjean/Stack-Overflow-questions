import pandas as pd


def get_num(dataframe:pd.DataFrame):

  return dataframe.select_dtypes([int,float])


def get_cat(dataframe:pd.DataFrame):

  return dataframe.select_dtypes(object)


def get_feat(dataframe:pd.DataFrame, starts_with:str, inverse=False):

  # or tuple of strings
  if inverse == True:
    return [feature for feature in dataframe.columns if feature.endswith(starts_with)]
  else:
    return [feature for feature in dataframe.columns if feature.startswith(starts_with)]


def gen_feat(dataframe:pd.DataFrame, prefix:str, inverse=False):

    if inverse == True:
      return [f'{prefix}{feature}' for feature in dataframe.columns]
    else:
      return [f'{feature}{prefix}' for feature in dataframe.columns]
