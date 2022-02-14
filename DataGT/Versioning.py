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

  
class Dataset():


  def __init__(self, dataframe:pd.DataFrame):

    init_name = 'original'
    self.versions_index = {init_name:0}
    self.versions = {0:self.dict_constructor(dataframe=dataframe, step=init_name, index=0)}
    self.current_version = 0

    print(f'Version {self.current_version}: "{init_name}" initialized') 


  def last_index(self):

    return list(self.versions_index.values())[-1]


  def add_index(self, step, index):

    if step in list(self.versions_index.keys()) or step=='current':
      print('\nKey already in index, choose another key.\n')
      return False

    else:
      self.versions_index[step] = index
      return True


  def dict_constructor(self, dataframe, step, index):

    if index == 0:
      samples_diff = dataframe.shape[0]
      features_diff = dataframe.shape[1]
      index_diff = dataframe.index
      columns_diff = dataframe.columns

    else:
      last_version = self.versions.get(index-1)
      samples_diff = last_version.get('samples') - dataframe.shape[0]
      features_diff = last_version.get('features') - dataframe.shape[1]
      index_diff = last_version.get('index').difference(dataframe.index)
      columns_diff = last_version.get('columns').difference(dataframe.columns)

    dataframe_dict = {
      'name': step,
      'dataframe': dataframe,
      'samples': dataframe.shape[0],
      'samples_diff': samples_diff,
      'features': dataframe.shape[1],
      'features_diff': features_diff,
      'index': dataframe.index,
      'index_diff': index_diff,
      'columns': dataframe.columns,
      'columns_diff': columns_diff
    }

    return dataframe_dict


  def save_version(self, updated_dataframe:pd.DataFrame, step:str):

    next_version = self.last_index() + 1
    key_pass = self.add_index(step=step, index=next_version)
    if key_pass == True:
      self.versions[next_version] = self.dict_constructor(dataframe=updated_dataframe, step=step, index=next_version)
      self.current_version = next_version
      print(f'\nVersion {next_version}: "{step}" saved\n')    


  def parse_step(self, step):

    if type(step) == str:
      version_index = self.versions_index.get(step)
      version_name = step
    if type(step) == int:
      version_index = step
      version_name = self.versions.get(version_index).get('name')
    return version_index, version_name


  def get_version(self, step):

    version_index, version_name = self.parse_step(step)

    return self.versions.get(version_index)


  def delete_version(self, step):

    version_index, version_name = self.parse_step(step)
    del self.versions_index[version_name]
    del self.versions[version_index]

    print(f'\nVersion {version_index}: "{version_name}" deleted\n')


  def pull_features(self, features, step=0):

    return self.get_version(self.current_version).get('dataframe').join(self.get_version(step).get('dataframe')[features], how='inner')


  def get(self, item='dataframe', step='current'):

    if step == 'current': version = self.versions.get(self.current_version)
    elif step == 'latter': version = self.versions.get(self.current_version-1)
    else: version = self.get_version(step)

    try: 
      item = version.get(item)
      return item
    except KeyError: 
      print('\nKey error, try generating the item first.\n')


  def num_report(self, fig_fill_min=100):

    dataframe = self.get()

    samples = dataframe.shape[0]
    report = dataframe.select_dtypes([int, float, 'datetime']).describe().transpose()
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

    report_fig.update_layout(template=template,title=title)
    report_fig.show()

    self.versions[self.current_version]['numericals'] = report


  def cat_report(self, fig_fill_min=100):

    dataframe = self.get()

    samples = dataframe.shape[0]
    report = dataframe.select_dtypes('object').describe().transpose()
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

    report_fig.update_layout(template=template,title=title)
    report_fig.show()

    self.versions[self.current_version]['categoricals'] = report


  def report(self, fig_fill_min=0):

    dataframe = self.get()

    if self.current_version > 0:

      version_old = self.versions.get(self.current_version-1)
      samples_old, features_old = version_old.get('samples'), version_old.get('features')
      samples_diff = samples_old - dataframe.shape[0]
      samples_percent = round((samples_diff / samples_old) * 100, 2)
      features_diff = features_old - dataframe.shape[1]
      features_percent = round((features_diff / features_old) * 100, 2)
      print(f'\nSamples dropped: {samples_diff}/{samples_old} ({samples_percent}%)\nFeatures dropped: {features_diff}/{features_old} ({features_percent}%)\n')

    # fig 1
    num_df = self.num_report(fig_fill_min)
    # fig 2
    cat_df = self.cat_report(fig_fill_min)

  
  def help(self):

    print('This is the help.')
