import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer, MinMaxScaler
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
from DataGT.Utils import get_num


def num_report(dataframe, fig_fill_min=100):


  samples = dataframe.shape[0]
  dataframe = dataframe.select_dtypes([int, float])
  if dataframe.empty: 
    return None
  else:
    report = dataframe.describe().transpose()
    report['fill_%'] = ((report['count'] / samples) * 100).astype(float).round(2)
    report['nans'] = dataframe.isna().sum()
    report['nans_%'] = ((report['nans'] / samples) * 100).astype(float).round(2)

    for i in report.index:
      zeroes_count = dataframe[i][dataframe[i] == 0].shape[0]
      report.loc[i, 'zeroes'] = zeroes_count
      report.loc[i, 'zeroes_%'] = round(((zeroes_count / samples) * 100), 2)
    report = report[['count', 'fill_%', 'nans', 'nans_%','zeroes', 'zeroes_%','mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    report = report.sort_values(by='count', ascending=False).rename_axis(mapper='feature', axis=0)
    report_df = report[report['fill_%'] >= fig_fill_min][['fill_%', 'nans_%', 'zeroes_%']].transpose()
    report_fig = go.Figure(data=[go.Bar(name=str(report_df.index[index]), x=list(report_df.columns.values), y=list(report_df.iloc[index,:].values)) for index in range(report_df.shape[0])])
    
    if fig_fill_min == 0: title = 'numerical features characteristics'
    else: title = f'numerical features characteristics (fill >= {fig_fill_min}%: {report_df.shape[1]})'

    report_fig.update_layout(title=title)
    report_fig.show()

    return report


def cat_report(dataframe, fig_fill_min=100):

  samples = dataframe.shape[0]
  dataframe = dataframe.select_dtypes(['object', 'datetime'])
  if dataframe.empty: 
    return None
  else:
    report = dataframe.describe().transpose()
    report['fill_%'] = ((report['count'] / samples) * 100).astype(float).round(2)
    report['uniques_%'] = ((report['unique'] / samples) * 100).astype(float).round(2)
    report['nans'] = dataframe.isna().sum()
    report['nans_%'] = ((report['nans'] / samples) * 100).astype(float).round(2)
    report = report[['count', 'fill_%', 'unique', 'uniques_%', 'nans', 'nans_%', 'top', 'freq']]
    report = report.sort_values(by='count', ascending=False).rename_axis(mapper='feature', axis=0)
    report_df = report[report['fill_%'] >= fig_fill_min][['fill_%', 'nans_%', 'uniques_%']].transpose()
    report_fig = go.Figure(data=[go.Bar(name=str(report_df.index[index]), x=list(report_df.columns.values), y=list(report_df.iloc[index,:].values)) for index in range(report_df.shape[0])])

    if fig_fill_min == 0: title = 'categorical features characteristics'
    else: title = f'categorical features characteristics (fill >= {fig_fill_min}%: {report_df.shape[1]})'

    report_fig.update_layout(title=title)
    report_fig.show()

    return report



def report(dataframe, fig_fill_min=0):

  # fig 1
  num_report(dataframe,fig_fill_min)
  # fig 2
  cat_report(dataframe,fig_fill_min)

def shape_diff(shape_0,shape_1):

  dropped_samples = shape_0[0] - shape_1[0]
  dropped_features = shape_0[1] - shape_1[1]
  if dropped_samples > 0:
    print(f'Dropped samples: {dropped_samples}')
  if dropped_features > 0:
    print(f'Dropped features: {dropped_features}')


def scatter_matrix(dataframe:pd.DataFrame, third_dimension='', title='Scatter matrix of the dataset'):

    dataframe = get_num(dataframe)

    if len(third_dimension) > 0:

      fig = px.scatter_matrix(dataframe,
        dimensions=dataframe.columns,
        color=third_dimension,#, symbol="nutriscore_grade",
        title=title,
        labels={col:col.replace('_', ' ') for col in dataframe.columns}) # remove underscore

    else:

      fig = px.scatter_matrix(dataframe,
        dimensions=dataframe.columns,
        title=title,
        labels={col:col.replace('_', ' ') for col in dataframe.columns}) # remove underscore

    fig.update_traces(diagonal_visible=True)
    fig.update_layout(height=750)
    fig.show()


def split_by_corr(dataframe:pd.DataFrame, max_corr=0.5):

  cor_matrix = get_num(dataframe).corr().abs()
  upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))

  high_corr = [column for column in upper_tri.columns if any(upper_tri[column] >= max_corr)]
  low_corr = list(set(dataframe.columns).difference(set(high_corr)))

  return low_corr, high_corr


def dual_scatter_matrix(dataframe:pd.DataFrame, third_dimension='', split_corr=0.5):

  if len(third_dimension) > 0:

    third_dimension_values = dataframe[third_dimension]
    dataframe = dataframe.drop(third_dimension, axis=1)

  if split_corr != 0:

    title_low = f'Scatter matrix of the dataset (correlations: <{(split_corr * 100)}%)'
    title_high = f'Scatter matrix of the dataset (correlations: >{split_corr * 100}%)'
    low_corr, high_corr = split_by_corr(dataframe, split_corr)

    for dataframe_subset, title in zip([dataframe[low_corr],dataframe[high_corr]],[title_low,title_high]):
      
      dataframe_subset[third_dimension] = third_dimension_values
      scatter_matrix(dataframe_subset,third_dimension,title)
  
  else:

    scatter_matrix(dataframe,third_dimension)


def plot_distributions(dataframe:pd.DataFrame, bin_size=500,normalizer=None,normalizer_name=''):

  dataframe.fillna(0,inplace=True)

  for feature in dataframe.select_dtypes([int,float]).columns:

    original = dataframe[feature].values

    log_transformer = FunctionTransformer(lambda value: np.log(value + 1), inverse_func = lambda value: np.exp(value - 1), check_inverse = True)
    logged = log_transformer.transform(original).flatten()

    if not normalizer:

      normalizer = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
      normalizer_name = f'QuantileTransformer (n_quantiles=1000)'
    
    if normalizer:

      normalizer_name = normalizer_name

    normalized = normalizer.fit_transform(original.reshape(-1,1)).flatten()

    subplots = {'Original':original,'Logged':logged, normalizer_name:normalized}

    bin_original, bin_logged, bin_normalized = (original.max() - original.min()) / bin_size, (logged.max() - logged.min()) / bin_size, (normalized.max() - normalized.min()) / bin_size
    distplot = ff.create_distplot([original,logged,normalized], group_labels=[feature,'Log','Normalized'], bin_size=[bin_original, bin_logged, bin_normalized], curve_type='normal', show_rug=False)

    fig = make_subplots(
      rows=1, cols=3,
      specs=[[{}, {}, {}]],
      subplot_titles=list(subplots.keys()))
    
    histogram_dict, scatter_dict = dict(enumerate(distplot['data'][:3])), dict(enumerate(distplot['data'][3:]))

    for column, title in enumerate(subplots.keys()):

      values = pd.Series(subplots[title])
      skewness, kurtosis = round(values.skew(),2), round(values.kurtosis(),2)
      legend = f'Skewness: {skewness} - Kurtosis: {kurtosis}'

      fig.add_trace(go.Histogram(histogram_dict[column]), row=1, col=column + 1)
      fig.add_trace(go.Scatter(scatter_dict[column]), row=1, col=column + 1)
      fig.update_xaxes(title_text=legend, row=1, col=column + 1)

    fig.update_xaxes(type='log',row=1, col=2)
    fig.update_layout( showlegend=False, title_text=feature)
    fig.show()


def plot_series_vs(primary:pd.Series, secondary:pd.Series, X_param:pd.Series, X_type='linear'):

  fig = make_subplots(specs=[[{"secondary_y": True}]])

  for metric in [primary] + [secondary]:
    if len(metric) > 0:
      metric_format = f'mean_test_{metric}'
      min_max_scaler = MinMaxScaler()
      values = metric.values
      scaled_values = min_max_scaler.fit_transform(values.reshape(-1,1))
      values_flat = values.flatten()
      values_range = [values_flat.min(),values_flat.max()]
      if metric.name == primary.name:
        secondary_axis = False
        fig.update_yaxes(title_text=primary.name,secondary_y=False,range=values_range)
      else:
        secondary_axis = True
        fig.update_yaxes(title_text=secondary.name,secondary_y=True,range=values_range)
      fig.add_trace(go.Scatter(x=X_param.values, y=values_flat,
                        mode='markers',
                        name= metric.name
                        ),secondary_y = secondary_axis)
      
    else:
      continue

  fig.update_xaxes(type=X_type, title_text=f'{X_param.name} ({X_type})', exponentformat="e")
  fig.update_layout( title=f'Scaled {primary.name} vs. {secondary.name} by {X_param.name}')
  fig.show()


def knn_optimizer(model, X_train:pd.DataFrame, y_train:pd.DataFrame, X_val:pd.DataFrame, y_val:pd.DataFrame, metric, range=range(1,10)):  

  best_id, best_neighbors, best_score = 0, 0, None

  for id, neighbors in enumerate(range):

    knn = model(n_neighbors=neighbors)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_val)

    if metric == 'accuracy':

      score = accuracy_score(y_val, predictions)
      score = round(score * 100, 2)
      print(f'\nPass {id}: {neighbors} neighbor(s), {metric}: {score}')

      if best_score is None or score > best_score:
        best_neighbors, best_score, best_id = neighbors, score, id
      
    if metric == 'MSE':

      score = mean_squared_error(y_val, predictions)
      score = round(score, 2)
      print(f'\nPass {id}: {neighbors} neighbor(s), {metric}: {score}')

      if best_score is None or score < best_score:
        best_neighbors, best_score, best_id = neighbors, score, id

  print(f'\nBest pass {best_id}: {best_neighbors} neighbor(s), {metric}: {best_score}')
  
  return model(n_neighbors=best_neighbors).fit(X_train, y_train)


def eta_squared(x,y):
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT


def anova(dataframe:pd.DataFrame,by:str):

  dataframe_eta = dataframe.select_dtypes([int,float])
  dataframe_eta = pd.DataFrame(QuantileTransformer().fit_transform(dataframe_eta), columns=dataframe_eta.columns, index=dataframe_eta.index)
  dataframe_eta.loc['Eta²'] = dataframe_eta.apply(lambda column: eta_squared(dataframe[by], column.values), axis=0)
  dataframe_eta = pd.DataFrame(dataframe_eta.loc['Eta²',:]).rename_axis('Feature',axis=0).reset_index()
  fig = px.bar(dataframe_eta, x='Feature', y='Eta²', title=f'{by} Anova (on the normalized distributions)')
  fig.show()


def dist_plot(dataframe:pd.DataFrame, feature:str, by=None, bin_size=0.5):

  subsets = list()
  labels = list()
  if by is not None:
    labels = list(set(dataframe[by].values))
    labels.sort()
    for filter in labels:
      subsets.append(dataframe[dataframe[by]==filter][feature].values)
  else:
    labels = [feature]
    subsets = [dataframe[feature].values]

  fig = ff.create_distplot(subsets, group_labels=labels, bin_size=bin_size,
                          curve_type='normal', show_rug=False
                          )
  fig.update_layout(title_text=f'{feature} vs normal distribution', height=750)
  fig.show()


def heatmap(matrix:pd.DataFrame, title='', extra=None):
  
  if extra is not None:
    extra = extra.values
  fig = ff.create_annotated_heatmap(matrix.values, x=matrix.columns.to_list(), y=matrix.index.to_list(), annotation_text=extra)
  fig.update_layout(title=title)
  fig.show()


def corr_matrix(dataframe:pd.DataFrame, title='', extra=None):

    return heatmap(dataframe.select_dtypes([int,float]).corr().round(2),title,extra)

    
def pie_plot(dataframe:pd.DataFrame, feature:str, title=''):

  fig_df = pd.DataFrame(pd.Series((','.join(dataframe[feature].astype(str).to_list())).split(',')).value_counts(), columns=['population']).rename_axis(mapper='tag', axis=0)
  fig = px.pie(fig_df.reset_index(), names='tag', values='population', title=title)
  # fig.update_layout(template=template)
  fig.show()


def bar_plot(dataframe:pd.DataFrame, feature:str):

  fig_df = pd.DataFrame(pd.Series((','.join(dataframe[feature].astype(str).to_list())).split(',')).value_counts(), columns=['population']).rename_axis(mapper='tag', axis=0)
  fig = px.bar(fig_df.reset_index(), x='tag', y='population', title=f'{feature} population')
  # fig.update_layout(template=template)
  fig.show()


def box_plots(x_data,y_data, outliers=False):

  flattened_y = np.hstack(np.array(y_data))
  max_min = np.max(flattened_y) - np.hstack(flattened_y).min()
  range = int('1' + len(str(max_min)) * '0')
  dtick = max_min // range
  colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)',
            'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)']
      
  fig = go.Figure()

  for xd, yd, cls in zip(x_data, y_data, colors):
          fig.add_trace(go.Box(
              y=yd,
              name=str(xd),
              boxpoints=outliers,
              jitter=0.5,
              whiskerwidth=0.2,
              fillcolor=cls,
              marker_size=1,
              line_width=1)
          )

  fig.update_layout(
      height=700,
      yaxis=dict(
          autorange=True,
          showgrid=True,
          zeroline=True,
          dtick=dtick,
          gridcolor='rgb(255, 255, 255)',
          gridwidth=1,
          zerolinecolor='rgb(255, 255, 255)',
          zerolinewidth=2,
      ),
      margin=dict(
          l=40,
          r=30,
          b=80,
          t=100,
      ),
      paper_bgcolor='rgb(243, 243, 243)',
      plot_bgcolor='rgb(243, 243, 243)',
      showlegend=False
  )

  return fig

  
def batch_box_plots(dataframe:pd.DataFrame, by=None, title='', outliers=False):

  dataframe_num = dataframe.select_dtypes([int,float])

  if by in dataframe_num.columns:
    dataframe_num.drop(by,axis=1,inplace=True)

  if by is None:
    x_data = dataframe_num.columns.to_list()
    y_data = np.array([dataframe_num[feature].values for feature in x_data])
    fig = go.Figure()

    fig = box_plots(x_data,y_data,outliers)
    fig.update_layout(title=title)
    fig.show()

  else:

    for feature in dataframe_num.columns:

      filters = list(set(dataframe[by].values))
      filters.sort()
      x_data = filters
      y_data = [dataframe[dataframe[by]==filter][feature].values for filter in filters]

      fig = box_plots(x_data,y_data,outliers)
      fig_title = f'Box plots by {by} ({feature})'
      fig.update_layout(title=fig_title)
      fig.show()


def dist_plot(dataframe:pd.DataFrame, feature:str, by:str, bin_size=1):

  subsets = list()
  labels = list(set(dataframe[by].values))
  for filter in labels:
    subsets.append(dataframe[dataframe[by]==filter][feature].values)
  fig = ff.create_distplot(subsets, group_labels=labels, bin_size=bin_size,
                          curve_type='normal', show_rug=False)

  fig.update_layout(title_text=f'{feature} vs normal distribution', height=750)
  fig.show()


def sum_dtypes(dataframe:pd.DataFrame):

  dtypes = dataframe.dtypes.value_counts()
  dtypes.index = dtypes.index.astype(str)
  dtypes = pd.DataFrame(data=dtypes, columns=['population']).rename_axis(mapper='dtype', axis=0)
  dtypes_fig = px.pie(dtypes.reset_index(), names='dtype', values='population', title="dtypes repartition")
  # dtypes_fig.update_layout(template=template)
  dtypes_fig.show()

  return dtypes


def sum_nans(dataframe:pd.DataFrame):

  nans = dataframe.isna().sum()
  nans = pd.DataFrame(data=nans, columns=['nans']).rename_axis(mapper='feature', axis=0).sort_values(by='nans', ascending=False)
  nans['nans_%'] = ((nans['nans'] / dataframe.shape[0]) * 100).round(2)
  nans = nans[nans['nans_%'] > 0]

  return nans


def sum_uniques(dataframe:pd.DataFrame):

  samples, features = dataframe.shape[0], dataframe.shape[1]
  uniques = dataframe.nunique()
  uniques = pd.DataFrame(data=uniques, columns=['uniques']).rename_axis(mapper='feature', axis=0).sort_values(by='uniques', ascending=False)
  uniques['uniques_%'] = ((uniques['uniques'] / dataframe.shape[0]) * 100).round(2)
  uniques = uniques[uniques['uniques_%'] > 0]

  return uniques


def join(series):

  return series.to_list()


def sample(*series):

  df = pd.DataFrame()
  
  for serie in series:
    if serie.name in df.columns:
      suffix = '_1'
    else:
      suffix = ''
    uniques = serie.unique()
    if len(uniques) >= 10:
      sample = pd.Series(uniques).sample(10)
      df[f'{serie.name}{suffix}'] = sample.values
    else:
      sample = serie.sample(10)
      df[f'{serie.name}{suffix}'] = sample.values
    df[f'{serie.name}_index{suffix}'] = sample.index
  
  df = df.reset_index().drop('index', axis=1).rename_axis(mapper='sample', axis=0)
  
  return df


def filter_tags(dataframe:pd.DataFrame, filters:dict):

  dataframe_features = dataframe.columns.tolist()
  features_df = pd.DataFrame(dataframe.columns, columns=['features'], index=dataframe.columns).rename_axis(mapper='index', axis=0)
  features_df['dtype'] = dataframe.dtypes.astype(str).values
  features_df['cat'] = features_df['dtype'].str.contains('object')
  features_df['num'] = features_df['dtype'].str.contains('float64')
  features_df['startswith'] = features_df['features'].str.split('_').str[0]
  features_df['splits'] = features_df['features'].str.count('_')
  features_df['processed'] = features_df['features']

  for filter in filters:
    if filter == 'endswith':
      for tag in filters[filter]:
        features_df[f'...{tag}'] = features_df['features'].str.endswith(tag)
        features_df['processed'] = features_df['processed'].str.replace(tag + r'$', '')

  filters_endswith = {f'...{filter}':sum for filter in filters['endswith']}
  misc = {feature:sum for feature in ['cat','num']}
  # dataframe qui filtre les startwith pat tag pour trouver les noms uniques
  features_filtered_df = features_df.groupby(by='startswith').agg({**misc, **filters_endswith, **{'splits': max, 'features': join, 'processed':join}}).rename_axis(mapper='index', axis=0)
  features_filtered_df['startswith_filtered'] = features_filtered_df.index
  features_filtered_df['total'] = features_filtered_df['cat'] + features_filtered_df['num']
  features_filtered_df_cols = features_filtered_df.columns.to_list()
  features_filtered_df = features_filtered_df[[features_filtered_df_cols[-1]]+features_filtered_df_cols[:-1]]
  features_filtered_df['processed'] = features_filtered_df['processed'].apply(lambda cell: set(cell))
  features_filtered_df = features_filtered_df.sort_values(by='splits', ascending=False)
  # recroisement avec la liste de features du dataframe
  features_names = [name for names in features_filtered_df['processed'].to_list() for name in names]
  features_final = list()

  for filter in filters:
    if filter == 'endswith':
      for feature_name in features_names:
          for tag in filters[filter] + ['']:
            temp_feature_name = f'{feature_name}{tag}'
            if temp_feature_name in dataframe_features:
              features_final.append(temp_feature_name)
              break

  print(f'\n{len(dataframe_features) - len(features_final)} features dropped\n')

  return features_final, features_filtered_df.drop('startswith_filtered', axis=1)


def filter_cat_feature(dataframe:pd.DataFrame, by:str, minimum_coverage=100):

  #filter top features with minimum cov and plot top features and others
  feature = dataframe[by].astype(str)
  features_df = pd.DataFrame(pd.Series((','.join(feature.to_list())).split(',')).value_counts(), columns=['population']).rename_axis(mapper='tag', axis=0)
  features_df['population_%'] = round((features_df['population'] / features_df['population'].values.sum()) * 100, 2)
  features_df['cumulative_uniques_%'] = features_df['population_%'].values.cumsum()
  features_n = features_df.shape[0]
  top_features_n = 0
  
  if minimum_coverage == 100:
    top_features = features_df.index.to_list()
    others = None

  else:
    for feature_index, coverage in enumerate(features_df['cumulative_uniques_%'].to_list()):
      if coverage >= minimum_coverage:
        top_features_n = feature_index +1
        break
    top_features = features_df.index.to_list()[:top_features_n]
    others = features_df[top_features_n:]
    
  top_features_df = features_df

  if others is not None:
    top_features_df = features_df.copy().head(top_features_n)
    top_features_df.loc['others',:] = [others['population'].sum(), others['population_%'].sum(), others['cumulative_uniques_%'].to_list()[-1]]
    top_features = top_features + ['others']
  # details
  filtered_percent = round((top_features_n / features_n) * 100, 2)
  print(f'\nMinimum coverage: {minimum_coverage}%\nFiltered "{by}": {top_features_n}/{features_n} ({filtered_percent}%)\nSelected: {top_features}\n')
  # fig 1
  if top_features_n > 0: top_string = f' (top {top_features_n} and others)'
  else: top_string = ''
  # filters dataframe with each feature to aggregate stats into top_features_df
  for feature in top_features:
    if feature == 'others':
        filter_df = others
    else:
      filter_df = dataframe.copy()
      filter_df['/filter'] = dataframe[by].str.contains(feature)
      filter_df = filter_df[filter_df['/filter'] == True].drop('/filter', axis=1)
    top_features_df.loc[feature, 'size'] = filter_df.shape[0] * filter_df.shape[1]
    top_features_df.loc[feature, 'nans'] = filter_df.isna().sum().sum()
    top_features_df.loc[feature, 'unique'] = filter_df.nunique().sum().sum()

  top_features_df['fill'] = top_features_df['size'] - top_features_df['nans']
  top_features_df['nans_%'] = ((top_features_df['nans'] / top_features_df['size']) * 100).round(2)
  top_features_df['fill_%'] = 100 - top_features_df['nans_%']
  top_features_df['uniques_%'] = ((top_features_df['unique'] / top_features_df['size']) * 100).round(2)
  top_features_df = top_features_df[['population', 'population_%', 'cumulative_uniques_%', 'fill', 'fill_%', 'nans', 'nans_%', 'unique', 'uniques_%', 'size']]
  top_features_fig = top_features_df[['population_%', 'fill_%', 'nans_%', 'uniques_%']].transpose()
  top_features_fig = go.Figure(data=[go.Bar(name=str(top_features_fig.index[index]), x=list(top_features_fig.columns.values), y=list(top_features_fig.iloc[index,:].values)) for index in range(top_features_fig.shape[0])])
  top_features_fig.update_layout(title=f'"{by}" charateristics per category' + top_string) #width=1200, height=600, 
  top_features_fig.show()

  return top_features_df
