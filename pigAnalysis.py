# distance series analysis of corrosion rate (dataILI)

import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report
import os

matplotlib.use('Agg')
target = {'regression': 'Wall_loss', 'classification': 'dimension'}

# ----------------------------------------------------------------------------------------------------------------------
# Variables
# ----------------------------------------------------------------------------------------------------------------------
param = dict(group_dist=50, test_size=0.25, cv=5,
             scoring={'regression': 'neg_mean_squared_error', 'classification': 'accuracy'},
             replica={'regression': 0, 'classification': 1},
             grid_search={'regression': False, 'classification': False})


# ----------------------------------------------------------------------------------------------------------------------
# Implemented Functions
# ----------------------------------------------------------------------------------------------------------------------
def clean_data(df, _info):
    df = df[df['Wall_loss'] > 0.0]
    df = df.join(_info.set_index(['pipeline', 'Year']), on=['pipeline', 'Year'])
    df['Area'] = df['length_mm'] * df['width_mm']
    df = df.drop(['depth_perc', 'length_mm', 'width_mm', 'ML_g'], axis=1)
    df = df[df['Distance'] <= 3200.0]
    df = df.dropna(axis=0, how='any').reset_index(drop=True)
    return df


def read_data(file_name, new):
    _char = pd.read_excel('{}.xlsx'.format(file_name), sheet_name='char').set_index('char')
    _info = pd.read_excel('{}.xlsx'.format(file_name), sheet_name='info')
    _flow = pd.read_excel('{}.xlsx'.format(file_name), sheet_name='flow')
    _orig = pd.read_excel('{}.xlsx'.format(file_name), sheet_name='flowOriginal')
    # ---------------------------------
    if new:
        _pipelines = ['pipe1_2002', 'pipe1_2010', 'pipe2_2000', 'pipe3_2002', 'pipe3_2010']
        df = pd.DataFrame()
        for sheet_name in _pipelines:
            if sheet_name == _pipelines[0]:
                df = pd.read_excel('{}.xlsx'.format(file_name), sheet_name=sheet_name)
            else:
                temp = pd.read_excel('{}.xlsx'.format(file_name), sheet_name=sheet_name)
                df = pd.concat([df, temp], ignore_index=True)
        df = clean_data(df, _info)
        _n_pipe = df['pipeline'].value_counts()
        excel_output(df, _root='', file_name='{}Cleaned'.format(file_name), csv=True)
    else:
        df = pd.read_csv('{}Cleaned.csv'.format(file_name))
        _n_pipe = len(_info['pipeline'].unique())
    return df, _info, _char, _flow, _n_pipe


def grouped_by_dist(df, file_name, group_dist):
    _root = 'dataSummary'
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    df = df.groupby(['pipeline', 'Year', 'grouping']).max().reset_index(drop=False)
    temp1 = []
    for row in range(len(df)):
        dist = 0
        temp1.append(0)
        while df['grouping'][row] > dist:
            dist += group_dist
        temp1[row] = dist
    df['distance_grouped'] = temp1
    # ---------------------------------
    df2 = df.groupby(['pipeline', 'Year', 'distance_grouped']).mean().reset_index(drop=False)
    temp2 = df.groupby(['pipeline', 'Year', 'distance_grouped'])['Wall_loss'].agg(pd.Series.max).to_frame()
    temp2 = temp2.reset_index(drop=False)
    df2 = df2.drop(['distance_grouped', 'grouping', 'Wall_loss'], axis=1)
    _scale = [temp2['Wall_loss'].min(), temp2['Wall_loss'].max()]
    df2['Wall_loss'] = (temp2['Wall_loss'] - _scale[0]) / (_scale[1] - _scale[0])
    # ---------------------------------
    excel_output(df2, _root, file_name='{}GroupedByDistance'.format(file_name), csv=False)
    return df2, _scale


def columns_stats(df):
    _root = 'dataSummary'
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    statistics = pd.DataFrame()
    for column in df.columns:
        if (column != 'Distance') and (column != 'grouping') and (column != 'Area') and (column != 'Wall_loss'):
            if column == df.columns[0]:
                statistics = pd.DataFrame(df[column].value_counts()).reset_index(drop=False)
                statistics.rename(columns={'index': column, column: 'samples_num'}, inplace=True)
            else:
                temp = pd.DataFrame(df[column].value_counts()).reset_index(drop=False)
                temp.rename(columns={'index': column, column: 'samples_num'}, inplace=True)
                statistics = pd.concat([statistics, temp], axis=1)
    excel_output(statistics, _root, file_name='columnsStats', csv=True)
    return statistics


def view_data(df, grouped, _char, _scale):
    _root = 'dataSummary/WallLossVsDistance'
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    _pipelines = df['pipeline'].unique()
    for pipe in _pipelines:
        df1 = df.loc[df['pipeline'] == pipe].reset_index(drop=True)
        fig, ax = plt.subplots(1, figsize=(12, 9))
        if grouped:
            name = 'groupedPipe_{}'.format(pipe)
            years = df1['Year'].unique()
            for year in years:
                df2 = df1.loc[df1['Year'] == year].reset_index(drop=True)
                _X = df2['Distance']
                _y = df2['Wall_loss'] * (_scale[1] - _scale[0]) + _scale[0]
                plt.scatter(_X, _y, label='year = {}'.format(year))
        else:
            name = 'dimensionPipe_{}'.format(pipe)
            dimension = np.sort(df1['dimension'].unique())
            dimension = dimension[::-1]
            for dim in dimension:
                df2 = df1.loc[df1['dimension'] == dim].reset_index(drop=True)
                _X = df2['Distance']
                _y = df2['Wall_loss']
                plt.scatter(_X, _y, label=dim)
        # ---------------------------------
        _X = df1['Distance']
        b = _char.loc['steelThick_mm', 'pipeline_{}'.format(pipe)]
        plt.plot(_X, 0 * _X + b, '-', color='gray', linewidth=5.0, label='Wall Thickness = {:.1f} mm'.format(b))
        # ---------------------------------
        if not grouped:
            years = df1['Year'].unique()
            if len(years) == 1:
                _info = 'year = {}'.format(years[0])
            else:
                _info = 'years = {} & {}'.format(years[0], years[1])
            plt.text(0.03, 0.96, _info,
                     ha='left', va='top', transform=ax.transAxes,
                     fontdict={'color': 'k', 'size': 20},
                     bbox={'boxstyle': 'round', 'fc': 'snow', 'ec': 'gray', 'pad': 0.5})
        # ---------------------------------
        plt.grid(linewidth=0.5)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Distance (m)', fontsize=27)
        plt.ylabel('Wall Loss (mm)', fontsize=27)
        ax.set_xlim(0, 3500)
        ax.set_ylim(0, 25)
        plt.legend(loc='upper right', fontsize=18, fancybox=True, shadow=True)
        plt.title(label='Pipeline #{}'.format(pipe), fontsize=30)
        plt.savefig('{}/{}.png'.format(_root, name))
        plt.close()


def grouped_by_col(df, column):
    df1 = df.groupby([column]).mean().reset_index(drop=False)
    df2 = df.groupby([column]).count().reset_index(drop=False)
    return df1, df2


def dimensions(df):
    _root = 'dataSummary/WallLossVsDimension'
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    df2, df3 = grouped_by_col(df, column='dimension')
    for column in df2.columns:
        if column != 'dimension':
            fig, ax = plt.subplots(1, figsize=(12, 9))
            _X = df2['dimension']
            _y = df2[column]
            plt.plot(_X, _y, 'ok-', linewidth=5.0, label='Raw data')
            # ---------------------------------
            _info = 'grouped by "corrosion type" \ny-axis represents average values'
            plt.text(0.03, 0.97, _info,
                     ha='left', va='top', transform=ax.transAxes,
                     fontdict={'color': 'k', 'size': 20},
                     bbox={'boxstyle': 'round', 'fc': 'snow', 'ec': 'gray', 'pad': 0.5})
            # ---------------------------------
            plt.grid(linewidth=0.5)
            plt.xticks(fontsize=16, rotation=45)
            plt.yticks(fontsize=20)
            temp = [df2[column].min(), df2[column].max(), df2[column].mean(), df2[column].std()]
            ax.set_ylim(temp[0] - temp[3], temp[1] + temp[3])
            plt.ylabel('{} (normalized)'.format(column), fontsize=27)
            plt.legend(loc='upper right', fontsize=18, fancybox=True, shadow=True)
            plt.tight_layout()
            plt.savefig('{}/{}.png'.format(_root, column))
            plt.close()


def excel_output(_object, _root, file_name, csv):
    if csv:
        if _root != '':
            _object.to_csv('{}/{}.csv'.format(_root, file_name))
        else:
            _object.to_csv('{}.csv'.format(file_name))
    else:
        _object.to_excel('{}/{}.xls'.format(_root, file_name))


# ----------------------------------------------------------------------------------------------------------------------
def select_features(df, _target):
    if _target == 'regression':
        df = df[['pipeline', 'Year', 'Distance', 'Pipe_age',
                 'Gas_FR', 'Water_FR', 'Condensate_FR', 'Temperature', 'Pressure', 'CO2_content', 'Wall_loss']]
    else:
        df = df[['pipeline', 'Year', 'Pipe_age', 'Area',
                 'Gas_FR', 'Water_FR', 'Condensate_FR', 'Temperature', 'Pressure', 'CO2_content', 'dimension']]
    return df


def encode_data(df, _flow, _target):
    df2 = pd.DataFrame()
    _pipelines = pd.Series(dtype=float)
    for column in df.columns:
        if column == 'pipeline':
            _pipelines = df['pipeline']
        elif column in _flow.columns:
            _scale = [_flow[column].min(), _flow[column].max()]
            df2[column] = abs(df[column] - _scale[0]) / (_scale[1] - _scale[0])
        elif column == 'Distance':
            _scale = [0, 3200]
            df2[column] = abs(df[column] - _scale[0]) / (_scale[1] - _scale[0])
        elif column == 'Area':
            _scale = [df[column].min(), df[column].max()]
            df2[column] = abs(df[column] - _scale[0]) / (_scale[1] - _scale[0])
        elif column == 'Wall_loss':
            if _target != 'regression':
                _scale = [df[column].min(), df[column].max()]
                df2[column] = abs(df[column] - _scale[0]) / (_scale[1] - _scale[0])
            else:
                df2[column] = df[column]
        else:
            df2[column] = df[column]
    return df2, _pipelines


def split_data(df, _pipelines, _param, _approach, _target):
    if (_approach[0] == 'exp.') and (_target == 'regression'):
        df = df.copy(deep=True)
        df['pipeline'] = _pipelines
        df_train = df[(df['pipeline'] != _approach[1]) | (df['Year'] != 1)].reset_index(drop=True)
        df_train = df_train.drop(['pipeline'], axis=1)
        df_test = df[(df['pipeline'] == _approach[1]) & (df['Year'] == 1)].reset_index(drop=True)
        df_test = df_test.drop(['pipeline'], axis=1)
    else:
        df = shuffle(df)
        head = int((1 - _param['test_size']) * len(df))
        tail = len(df) - head
        df_train = df.head(head).reset_index(drop=True)
        df_test = df.tail(tail).reset_index(drop=True)
    return df_train, df_test


def split_xy(df):
    df = df.copy(deep=True)
    df = df.drop(['Year'], axis=1)
    df = shuffle(df)
    _X = df.iloc[:, 0:-1].reset_index(drop=True)
    _y = df.iloc[:, -1].to_numpy()
    return _X, _y


def grid_search_reg(model):
    models = []
    hp1 = {'MLP': [(2,), (4,), (6,), (8,), (10,), (12,), (16,), (32,), (50,), (100,),
                   (2, 2), (4, 4), (6, 6), (8, 8), (10, 10), (12, 12), (16, 16),
                   (2, 2, 2), (4, 4, 4), (6, 6, 6), (8, 8, 8), (10, 10, 10), (12, 12, 12), (16, 16, 16)],
           'SVM': ['auto', 'scale', 1, 0.1, 0.01, 0.001],
           'RF': [10, 20, 50, 70, 100, 200],
           'KNN': [1, 2, 3, 4, 5, 6, 7, 8]}
    hp2 = {'MLP': ['constant'],
           'SVM': [5, 10, 100, 1000],
           'RF': ['auto', 0.5, 0.7, 1.0],
           'KNN': ['auto', 'ball_tree', 'kd_tree']}
    for n in hp1[model]:
        for m in hp2[model]:
            if model == 'MLP':
                models.append(('MLP_{}_{}'.format(n, m), MLPRegressor(max_iter=10000, random_state=5,
                                                                      hidden_layer_sizes=n, learning_rate=m)))
            elif model == 'SVM':
                models.append(('SVM_{}_{}'.format(n, m), SVR(gamma=n, C=m)))
            elif model == 'RF':
                models.append(('RF_{}_{}'.format(n, m), RandomForestRegressor(random_state=5,
                                                                              n_estimators=n, max_features=m)))
            elif model == 'KNN':
                models.append(('KNN_{}_{}'.format(n, m), KNeighborsRegressor(n_neighbors=n, algorithm=m)))
    return models


def grid_search_cls(model):
    models = []
    hp1 = {'MLP': [(2,), (4,), (6,), (8,), (10,)],  # , (12,), (16,), (32,), (50,), (100,),
           # (2, 2), (4, 4), (6, 6), (8, 8), (10, 10), (12, 12), (16, 16),
           # (2, 2, 2), (4, 4, 4), (6, 6, 6), (8, 8, 8), (10, 10, 10), (12, 12, 12), (16, 16, 16)],
           'SVM': ['auto', 'scale', 1, 0.1, 0.01, 0.001],
           'RF': [10, 20, 50, 70, 100, 200],
           'KNN': [1, 2, 3, 4, 5, 6, 7, 8]}
    hp2 = {'MLP': ['constant'],
           'SVM': [5, 10, 100, 1000],
           'RF': ['auto', 0.5, 0.7, 1.0],
           'KNN': ['auto', 'ball_tree', 'kd_tree']}
    for n in hp1[model]:
        for m in hp2[model]:
            if model == 'MLP':
                models.append(('MLP_{}_{}'.format(n, m), MLPClassifier(max_iter=10000, random_state=5,
                                                                       hidden_layer_sizes=n, learning_rate=m)))
            elif model == 'SVM':
                models.append(('SVM_{}_{}'.format(n, m), SVC(gamma=n, C=m)))
            elif model == 'RF':
                models.append(('RF_{}_{}'.format(n, m), RandomForestClassifier(random_state=5,
                                                                               n_estimators=n, max_features=m)))
            elif model == 'KNN':
                models.append(('KNN_{}_{}'.format(n, m), KNeighborsClassifier(n_neighbors=n, algorithm=m)))
    return models


def compare_models(df, models, _param, _target):
    scoring, cv, replicas = _param['scoring'][_target], _param['cv'], _param['replica'][_target]
    # ---------------------------------
    results = pd.DataFrame()
    for i in range(replicas):
        _X_train, _y_train = split_xy(df)
        temp = []
        for name, model in models:
            # print(i, name)
            cv_results = cross_val_score(model, _X_train, _y_train, cv=cv, scoring=scoring)
            cv_results = np.mean(cv_results)
            temp.append(cv_results)
        if i == 0:
            results = pd.DataFrame(temp)
        else:
            results = pd.concat([results, pd.DataFrame(temp)], axis=1, ignore_index=True)
    results['mean'] = results.mean(axis=1)
    results['std'] = results.std(axis=1)
    # ---------------------------------
    _names, _models = [], []
    for name, model in models:
        _names.append(name)
        _models.append(model)
    results['name'] = pd.Series(_names)
    results['model'] = pd.Series(_models)
    # ---------------------------------
    id_best = results['mean'].idxmax()
    _best = results.loc[id_best, 'model']
    return results, _best


def prediction(df, estimator, _pipelines, _param, _approach, _target):
    replicas = _param['replica'][_target]
    errors = pd.DataFrame()
    for i in range(replicas):
        df_training, df_testing = split_data(df, _pipelines, _param, _approach, _target)
        _X_train, _y_train = split_xy(df_training)
        estimator.fit(_X_train, _y_train)
        _X_test, _y_test = split_xy(df_testing)
        _y_pred = estimator.predict(_X_test)
        errors.loc[i, 'r2'] = r2_score(_y_test, _y_pred)
        errors.loc[i, 'mse'] = mean_squared_error(_y_test, _y_pred)
        errors.loc[i, 'mae'] = mean_absolute_error(_y_test, _y_pred)
        errors.loc[i, 'rmse'] = np.sqrt(mean_squared_error(_y_test, _y_pred))
    _scores = [('R2', np.mean(errors['r2']), np.std(errors['r2'])),
               ('MSE', np.mean(errors['mse']), np.std(errors['mse'])),
               ('MAE', np.mean(errors['mae']), np.std(errors['mae'])),
               ('RMSE', np.mean(errors['rmse']), np.std(errors['rmse']))]
    return _scores


def create_production(_flow):
    df2 = pd.DataFrame()
    temp = pd.DataFrame()
    distance = np.arange(50, 3250, 50)
    for n in range(len(_flow)):
        if n == 0:
            df2['pipeline'] = pd.Series([_flow.loc[n, 'pipeline'] for i in range(len(distance))])
            df2['Year'] = pd.Series([_flow.loc[n, 'Year'] for i in range(len(distance))])
            df2['Distance'] = pd.Series(distance)
        else:
            temp['pipeline'] = pd.Series([_flow.loc[n, 'pipeline'] for i in range(len(distance))])
            temp['Year'] = pd.Series([_flow.loc[n, 'Year'] for i in range(len(distance))])
            temp['Distance'] = pd.Series(distance)
            df2 = pd.concat([df2, temp], ignore_index=True)
    df2 = df2.join(_flow.set_index(['pipeline', 'Year']), on=['pipeline', 'Year'])
    return df2


def sensitivity(df, estimator, _pipelines, features, _param, _approach, _target):
    scoring, replicas = _param['scoring'][_target], _param['replica'][_target]
    df2 = pd.DataFrame()
    for i in range(replicas):
        df_training, df_testing = split_data(df, _pipelines, _param, _approach, _target)
        for name, feature in features:
            if name == 'All_in':
                training = df_training
                testing = df_testing
            else:
                training = df_training.drop([feature], axis=1)
                testing = df_testing.drop([feature], axis=1)
            _X_train, _y_train = split_xy(training)
            estimator.fit(_X_train, _y_train)
            _X_test, _y_test = split_xy(testing)
            _y_pred = estimator.predict(_X_test)
            if scoring == 'accuracy':
                _score = accuracy_score(_y_test, _y_pred)
            elif scoring == 'r2':
                _score = r2_score(_y_test, _y_pred)
            else:
                _score = mean_squared_error(_y_test, _y_pred)
            df2.loc[i, name] = _score
    return df2


# ----------------------------------------------------------------------------------------------------------------------
def smooth(y_array, window):
    if window != 0:
        y_smoothed = pd.Series(y_array).rolling(window, center=False).mean().shift(-int(0.5 * window)).to_numpy()
    else:
        y_smoothed = y_array
    return y_smoothed


def compare_models_plot(df, _target):
    _root = '{}/gridSearchModels'.format(_target)
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    for model_name in ['MLP', 'SVM', 'RF', 'KNN']:
        x_axis_index = [i + 1 for i in np.arange(len(df))]
        if _target == 'regression':
            _y = [-i for i in df['{}_mean'.format(model_name)]]
        else:
            _y = df['{}_mean'.format(model_name)].tolist()
        _y_err = df['{}_std'.format(model_name)].tolist()
        bar_width = 0.45
        colors = {'MLP': 'mistyrose', 'SVM': 'cornsilk', 'RF': 'lightgray', 'KNN': 'lightcyan'}
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.bar(x_axis_index, _y, width=bar_width, color=colors[model_name], edgecolor='black', zorder=3,
               yerr=_y_err, capsize=5, align='center', ecolor='black', alpha=0.5, label=model_name)
        # ---------------------------------
        letter = {'MLP': 'A', 'RF': 'B', 'KNN': 'C', 'SVM': 'D'}
        plt.text(0.02, 0.98, '{}'.format(letter[model_name]),
                 ha='left', va='top', transform=ax.transAxes,
                 fontdict={'color': 'k', 'weight': 'bold', 'size': 50})
        # ---------------------------------
        ax.grid(axis='y', linewidth=0.35, zorder=0)
        ax.set_xticks(x_axis_index)
        ax.set_xticklabels(x_axis_index, fontsize=20, rotation=45)
        ax.set_xlabel('Grid serach combination', fontsize=27)
        y_axis_lim = {'regression': [0.12, 0.02], 'classification': [1.2, 0.2]}
        y_axis_index = np.arange(0, y_axis_lim[_target][0], y_axis_lim[_target][1])
        ax.set_yticks(y_axis_index)
        ax.set_yticklabels(['{:.2f}'.format(i) for i in y_axis_index], fontsize=20)
        y_axis_label = {'regression': 'Mean Squared Error (MSE)', 'classification': 'Accuracy'}
        ax.set_ylabel(y_axis_label[_target], fontsize=27)
        plt.legend(loc='upper right', fontsize=18, fancybox=True, shadow=True)
        # plt.tight_layout()
        plt.savefig('{}/{}.png'.format(_root, model_name))
        plt.close()


def compare_models_box_plot(df, _param, _target):
    cv, replicas = _param['cv'], _param['replica'][_target]
    _root = '{}/gridSearchModels'.format(_target)
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    x_axis_names = [name for name in df['name']]
    df = df.drop(['name', 'mean', 'std', 'model'], axis=1)
    if _target == 'regression':
        df = df.transform(lambda x: -x)
    _y_matrix = df.values.tolist()
    fig, ax = plt.subplots(1, figsize=(12, 9))
    plt.boxplot(_y_matrix, labels=x_axis_names, sym='',
                medianprops=dict(color='lightgrey', linewidth=1.0),
                meanprops=dict(linestyle='-', color='black', linewidth=1.5), meanline=True, showmeans=True)
    # ---------------------------------
    _info = '{}-fold cross validation analysis \n{} replications per algorithm'.format(cv, replicas)
    plt.text(0.03, 0.96, _info,
             ha='left', va='top', transform=ax.transAxes,
             fontdict={'color': 'k', 'size': 18},
             bbox={'boxstyle': 'round', 'fc': 'snow', 'ec': 'gray', 'pad': 0.5})
    # ---------------------------------
    ax.grid(axis='y', linewidth=0.35, zorder=0)
    x_axis_index = [i + 1 for i in np.arange(len(x_axis_names))]
    ax.set_xticks(x_axis_index)
    ax.set_xticklabels(x_axis_names, fontsize=30)
    y_axis_lim = {'regression': [0, 0.06, 0.01], 'classification': [0.4, 1.2, 0.2]}
    y_axis_index = np.arange(y_axis_lim[_target][0], y_axis_lim[_target][1], y_axis_lim[_target][2])
    ax.set_yticks(y_axis_index)
    ax.set_yticklabels(['{:.2f}'.format(i) for i in y_axis_index], fontsize=20)
    y_label = {'regression': 'Mean Squared Error (MSE)', 'classification': 'Accuracy'}
    ax.set_ylabel(y_label[_target], fontsize=28)
    # plt.tight_layout()
    plt.savefig('{}/comparison.png'.format(_root))
    plt.close()


def importance_correlation_plot(df, estimator, _target):
    _root = '{}/bestModelPerformance'.format(_target)
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    df = df.copy(deep=True)
    df = df.drop(['Year'], axis=1)
    # ---------------------------------
    names = df.columns
    imp = estimator.feature_importances_
    indices = np.argsort(imp)
    fig, ax = plt.subplots(1, figsize=(12, 9))
    plt.barh(range(len(indices)), imp[indices], color='black', align='center')
    # ---------------------------------
    x_axis_lim = {'regression': [0.8, 0.1], 'classification': [0.8, 0.1]}
    x_axis_index = np.arange(0, x_axis_lim[_target][0], x_axis_lim[_target][1])
    ax.set_xticks(x_axis_index)
    ax.set_xticklabels(x_axis_index, fontsize=20)
    ax.set_xticklabels(['{:.1f}'.format(i) for i in x_axis_index], fontsize=20)
    ax.set_xlabel('Relative Importance', fontsize=30)
    plt.yticks(range(len(indices)), [names[i] for i in indices], fontsize=20)
    plt.tight_layout()
    plt.savefig('{}/featuresImp.png'.format(_root))
    plt.close()
    # ---------------------------------
    if _target == 'regression':
        corr = df.corr()
        plt.subplots(figsize=(12, 12))
        sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap='coolwarm', square=True)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig('{}/corrMatrix.png'.format(_root))
        plt.close()


def parity_plot(_y_test, _y_pred, _scale, _scores, _approach, _target):
    _root = '{}/bestModelPerformance'.format(_target)
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    _info = '{} = {:.2f} +/- {:.2f}\n{} = {:.2f} +/- {:.2f}\n{} = {:.2f} +/- {:.2f}\n{} = {:.2f} +/- {:.2f}'. \
        format(_scores[0][0], _scores[0][1], _scores[0][2],
               _scores[1][0], _scores[1][1], _scores[1][2],
               _scores[2][0], _scores[2][1], _scores[2][2],
               _scores[3][0], _scores[3][1], _scores[3][2])
    # ---------------------------------
    fig, ax = plt.subplots(1, figsize=(12, 9))
    _y_test = _y_test * (_scale[1] - _scale[0]) + _scale[0]
    _y_pred = _y_pred * (_scale[1] - _scale[0]) + _scale[0]
    plt.scatter(_y_pred, _y_test, c='black', label='Testing set')
    a, b = min(_y_test.min(), _y_pred.min()), max(_y_test.max(), _y_pred.max())
    plt.plot([a, b], [a, b], '-', c='lightcoral', linewidth=7.0, label='y = x')
    # ---------------------------------
    plt.text(0.03, 0.96, _info,
             ha='left', va='top', transform=ax.transAxes,
             fontdict={'color': 'k', 'size': 20},
             bbox={'boxstyle': 'round', 'fc': 'snow', 'ec': 'gray', 'pad': 0.5})
    # ---------------------------------
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    plt.xlabel('Wall loss (mm) - Predicted', fontsize=25)
    plt.ylabel('Wall loss (mm) - True', fontsize=25)
    plt.legend(loc='upper right', fontsize=20, fancybox=True, shadow=True)
    # plt.tight_layout()
    if _approach[0] == 'exp.':
        plt.savefig('{}/parityPlotTestPipe_{}.png'.format(_root, _approach[1]))
    else:
        plt.savefig('{}/parityPlotTestRandom.png'.format(_root))
    plt.close()
    # ---------------------------------
    df = pd.DataFrame(columns=['True_value', 'Predicted_value'])
    df['True_value'] = y_test
    df['Predicted_value'] = y_pred
    if _approach[0] == 'exp.':
        excel_output(df, _root, file_name='parityPlotDataTestPipe_{}'.format(_approach[1]), csv=True)
    else:
        excel_output(df, _root, file_name='parityPlotDataTestRandom', csv=True)


def prediction_plot(df_train, _y_train, df_test, _y_test, _y_pred, _scale, _char, _info, _approach, _target):
    if _approach[0] == 'exp.':
        _root = '{}/bestModelPerformance/testPipe_{}'.format(_target, _approach[1])
    else:
        _root = '{}/bestModelPerformance/testRandom'.format(_target)
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    df_train['y_train'] = pd.Series(_y_train)
    df_test['y_test'] = pd.Series(_y_test)
    df_test['y_pred'] = pd.Series(_y_pred)
    # ---------------------------------
    scale_age = [_info['Pipe_age'].min(), _info['Pipe_age'].max()]
    temp = (_info['Pipe_age'] - scale_age[0]) / (scale_age[1] - scale_age[0])
    pipe_ages = np.sort(temp.unique())
    for i, age in enumerate(pipe_ages):
        fig, ax = plt.subplots(1, figsize=(12, 9))
        df = df_train.loc[df_train['Pipe_age'] == age].reset_index(drop=True)
        if len(df) > 0:
            _X = df['Distance'] * 3200
            _y = df['y_train'] * (_scale[1] - _scale[0]) + _scale[0]
            ax.scatter(_X, _y, c='black', label='Training set')
        # ---------------------------------
        df = df_test.loc[df_test['Pipe_age'] == age].reset_index(drop=True)
        if len(df) > 0:
            _X = df['Distance'] * 3200
            _y = df['y_test'] * (_scale[1] - _scale[0]) + _scale[0]
            ax.scatter(_X, _y, c='blue', label='Testing set (True value)')
            # ---------------------------------
            _X = df['Distance'] * 3200
            _y = df['y_pred'] * (_scale[1] - _scale[0]) + _scale[0]
            ax.scatter(_X, _y, c='red', label='Testing set (Predicted value)')
        # ---------------------------------
        if i == 0:
            pipe, year = 2, 2000
        elif i == 1:
            pipe, year = 1, 2002
        elif i == 2:
            pipe, year = 3, 2002
        elif i == 3:
            pipe, year = 1, 2010
        else:
            pipe, year = 3, 2010
        # ---------------------------------
        _X = df_train['Distance'] * 3200
        b = _char.loc['steelThick_mm', 'pipeline_{}'.format(pipe)]
        plt.plot(_X, 0 * _X + b, '-', color='gray', linewidth=5.0, label='Wall Thickness = {:.1f} mm'.format(b))
        # ---------------------------------
        plt.text(0.03, 0.96, 'year = {}\nage = {:.0f}'.format(year, age * 13 + 1),
                 ha='left', va='top', transform=ax.transAxes,
                 fontdict={'color': 'k', 'size': 20},
                 bbox={'boxstyle': 'round', 'fc': 'snow', 'ec': 'gray', 'pad': 0.5})
        # ---------------------------------
        plt.grid(linewidth=0.5)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Distance (m)', fontsize=30)
        plt.ylabel('Wall Loss (mm)', fontsize=30)
        ax.set_xlim(0, 3500)
        ax.set_ylim(0, 25)
        plt.legend(loc='upper right', fontsize=20, fancybox=True, shadow=True)
        # plt.title(label='Pipeline #{}'.format(pipe), fontsize=21)
        # plt.tight_layout()
        plt.savefig('{}/Pipe{}_Year{}.png'.format(_root, pipe, year))
        plt.close()


def production_plot(df, _char, _scale, window, _target):
    _root = '{}/bestModelPerformance/predOtherYears'.format(_target)
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    _pipelines = df['pipeline'].unique()
    for pipe in _pipelines:
        df1 = df.loc[df['pipeline'] == pipe].reset_index(drop=True)
        fig, ax = plt.subplots(1, figsize=(12, 9))
        # ---------------------------------
        _X = df1['Distance'] * 3200
        b = _char.loc['steelThick_mm', 'pipeline_{}'.format(pipe)]
        plt.plot(_X, 0 * _X + b, '-', color='gray', linewidth=5.0, label='Wall Thickness = {:.1f} mm'.format(b))
        # ---------------------------------
        ages = df1['Pipe_age'].unique()
        for age in ages:
            df2 = df1.loc[df1['Pipe_age'] == age].reset_index(drop=True)
            _X = df2['Distance'] * 3200
            _y = df2['y_prod'] * (_scale[1] - _scale[0]) + _scale[0]
            _y = smooth(_y, window)
            plt.plot(_X, _y, 'o-', label='pipe age = {:.0f}'.format(age * 13 + 1))
        # ---------------------------------
        _info = 'Model Production'
        plt.text(0.03, 0.96, _info,
                 ha='left', va='top', transform=ax.transAxes,
                 fontdict={'color': 'k', 'size': 20},
                 bbox={'boxstyle': 'round', 'fc': 'snow', 'ec': 'gray', 'pad': 0.5})
        # ---------------------------------
        plt.grid(linewidth=0.5)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Distance (m)', fontsize=30)
        plt.ylabel('Wall Loss (mm)', fontsize=30)
        ax.set_xlim(0, 3500)
        ax.set_ylim(0, 10)
        plt.legend(loc='upper right', fontsize=18, fancybox=True, shadow=True)
        # plt.title(label='Pipeline #{}'.format(pipe), fontsize=21)
        # plt.tight_layout()
        plt.savefig('{}/pipe_{}.png'.format(_root, pipe))
        plt.close()


def sensitivity_plot(df, _target):
    _root = '{}/bestModelPerformance'.format(_target)
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    x_axis_names, _y, _y_err = [], [], []
    for column in df.columns:
        # print(column)
        x_axis_names.append(column)
        _y.append(np.mean(df[column]))
        _y_err.append(np.std(df[column]))
    bar_width = 0.5
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.bar(x_axis_names, _y, width=bar_width, color='k', edgecolor='black',
           yerr=_y_err, capsize=5, align='center', ecolor='black', alpha=0.5,
           label='Missed feature in the training set')
    # ---------------------------------
    ax.grid(axis='y', linewidth=0.35, zorder=0)
    x_axis_index = [i for i in np.arange(len(x_axis_names))]
    ax.set_xticks(x_axis_index)
    ax.set_xticklabels(x_axis_names, fontsize=20, rotation=45)
    y_axis_lim = {'regression': [0, 0.06, 0.01], 'classification': [0, 1.2, 0.2]}
    y_axis_index = np.arange(y_axis_lim[_target][0], y_axis_lim[_target][1], y_axis_lim[_target][2])
    ax.set_yticks(y_axis_index)
    ax.set_yticklabels(['{:.2f}'.format(i) for i in y_axis_index], fontsize=20)
    y_label = {'regression': 'Mean Squared Error (MSE)', 'classification': 'Accuracy'}
    ax.set_ylabel(y_label[_target], fontsize=25)
    plt.legend(loc='upper right', fontsize=20, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig('{}/featuresSens.png'.format(_root))
    plt.close()


def confusion_matrix(estimator, _x_test, _y_test, _target, normalize):
    _root = '{}/bestModelPerformance'.format(_target)
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    fig, ax = plt.subplots(1, figsize=(12, 12))
    plt.rcParams.update({'font.size': 14})
    colors = {'true': 'Blues', 'pred': 'Reds', 'all': 'Greens', 'none': 'Greys'}
    if normalize == 'none':
        plot_confusion_matrix(estimator, _x_test, _y_test,
                              cmap=colors[normalize], values_format='.0f', xticks_rotation='vertical', ax=ax)
    else:
        plot_confusion_matrix(estimator, _x_test, _y_test, normalize=normalize,
                              cmap=colors[normalize], values_format='.3f', xticks_rotation='vertical', ax=ax)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Predicted label', fontsize=21)
    plt.ylabel('True label', fontsize=21)
    if normalize == 'true':
        metric = 'Recall'
    elif normalize == 'pred':
        metric = 'Precision'
    elif normalize == 'all':
        metric = 'Confusion matrix (normalized)'
    else:
        metric = 'Confusion matrix'
    plt.title('{}'.format(metric), fontsize=25)
    plt.tight_layout()
    plt.savefig('{}/confMatrixTestRandom_{}.png'.format(_root, normalize))
    plt.close()


def class_report(estimator, _x_test, _y_test, _target):
    _root = '{}/bestModelPerformance'.format(_target)
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    _y_pred = estimator.predict(_x_test)
    report = classification_report(_y_test, _y_pred, output_dict=True)
    sns.heatmap(pd.DataFrame(report).T, annot=True)
    print('average accuracy = {:.5f}'.format(report['accuracy']))
    excel_output(pd.DataFrame(report), _root, file_name='classificationReport', csv=True)
    return pd.DataFrame(report)


# --------------------------------------------------------------------------------------------------------------------
# BEGIN
# --------------------------------------------------------------------------------------------------------------------

# reading data
dataCleaned, info, char, flow, n_pipe = read_data('dataILI', new=False)
inline, scale = grouped_by_dist(dataCleaned, 'dataILI', param['group_dist'])

# data summary
columns_stats(dataCleaned)
view_data(dataCleaned, False, char, scale)
view_data(inline, True, char, scale)

# --------------------------------------------------------------------------------------------------------------------
# REGRESSION PROBLEM
# --------------------------------------------------------------------------------------------------------------------

if param['replica']['regression'] != 0:

    # encoding data
    regression = select_features(inline, 'regression')
    regression, pipelines = encode_data(regression, flow, 'regression')

    # grid-search to find the best model of each regressor algorithm (one-time output)
    if param['grid_search']['regression']:
        root = 'regression/gridSearchModels'
        if not os.path.exists(root):
            os.makedirs(root)
        # ---------------------------------
        best_models = {}
        df_scores = pd.DataFrame()
        for algorithm in ['MLP', 'SVM', 'RF', 'KNN']:
            print(algorithm)
            algorithms = grid_search_reg(algorithm)
            scores, best = compare_models(regression, algorithms, param, 'regression')
            best_models[algorithm] = best
            df_scores['{}_mean'.format(algorithm)] = scores['mean']
            df_scores['{}_std'.format(algorithm)] = scores['std']
            printOut = pd.DataFrame(algorithms)
            printOut['mean'], printOut['std'] = [-x for x in scores['mean']], scores['std']
            excel_output(printOut, root, file_name='{}'.format(algorithm), csv=False)
        compare_models_plot(df_scores, 'regression')
        models_reg = [('MLP', best_models['MLP']),
                      ('SVM', best_models['SVM']),
                      ('RF', best_models['RF']),
                      ('KNN', best_models['KNN'])]
    else:
        models_reg = [('MLP', MLPRegressor(max_iter=10000, random_state=5)),
                      ('SVM', SVR(C=1000, gamma=1)),
                      ('RF', RandomForestRegressor(n_estimators=200, random_state=5)),
                      ('KNN', KNeighborsRegressor(n_neighbors=5))]

    # comparing different models
    scores_reg, _best_reg = compare_models(regression, models_reg, param, 'regression')
    compare_models_box_plot(scores_reg, param, 'regression')
    excel_output(scores_reg, 'regression/gridSearchModels', file_name='comparison', csv=False)
    best_reg = models_reg[2][1]

    # features importance
    X, y = split_xy(regression)
    models_reg[2][1].fit(X, y)
    importance_correlation_plot(regression, models_reg[2][1], 'regression')

    # parity + prediction plots
    testing_approaches = [['random', 0], ['exp.', 1], ['exp.', 3]]
    for approach in testing_approaches:
        training_reg, testing_reg = split_data(regression, pipelines, param, approach, 'regression')
        X_train, y_train = split_xy(training_reg)
        best_reg.fit(X_train, y_train)
        X_test, y_test = split_xy(testing_reg)
        y_pred = best_reg.predict(X_test)
        scores_pred_reg = prediction(regression, best_reg, pipelines, param, approach, 'regression')
        parity_plot(y_test, y_pred, scale, scores_pred_reg, approach, 'regression')
        excel_output(X_train, 'regression/bestModelPerformance',
                     file_name='trainFeatureMatrixNorm_{}'.format(approach[1]), csv=True)
        prediction_plot(X_train, y_train, X_test, y_test, y_pred, scale, char, info, approach, 'regression')

    # production plots (years in between)
    regression2 = create_production(flow)
    X_prod, pipelines2 = encode_data(regression2, flow, 'regression')
    X_prod = X_prod.drop(['Year'], axis=1)
    X_train, y_train = split_xy(regression)
    best_reg.fit(X_train, y_train)
    y_prod = best_reg.predict(X_prod)
    X_prod['pipeline'] = pipelines2
    X_prod['y_prod'] = pd.Series(y_prod)
    production_plot(X_prod, char, scale, 10, 'regression')

    # sensitivity analysis
    features_reg = [('All_in', ''),
                    ('Distance', 'Distance'),
                    ('Pipe_age', 'Pipe_age'),
                    ('Gas_FR', 'Gas_FR'),
                    ('Water_FR', 'Water_FR'),
                    ('Condensate_FR', 'Condensate_FR'),
                    ('Temperature', 'Temperature'),
                    ('Pressure', 'Pressure'),
                    ('CO2_content', 'CO2_content')]
    scores_sens_reg = sensitivity(regression, best_reg, pipelines, features_reg, param, ['random', 0], 'regression')
    sensitivity_plot(scores_sens_reg, 'regression')

# --------------------------------------------------------------------------------------------------------------------
# CLASSIFICATION PROBLEM
# --------------------------------------------------------------------------------------------------------------------

if param['replica']['classification'] != 0:

    # encoding data
    classification = select_features(dataCleaned, 'classification')
    classification, pipelines = encode_data(classification, flow, 'classification')
    dimensions(classification)

    # grid-search to find the best model of each classifier algorithm (one-time output)
    if param['grid_search']['classification']:
        root = 'classification/gridSearchModels'
        if not os.path.exists(root):
            os.makedirs(root)
        # ---------------------------------
        best_models = {}
        df_scores = pd.DataFrame()
        for algorithm in ['MLP', 'SVM', 'RF', 'KNN']:
            print(algorithm)
            algorithms = grid_search_cls(algorithm)
            scores, best = compare_models(classification, algorithms, param, 'classification')
            best_models[algorithm] = best
            df_scores['{}_mean'.format(algorithm)] = scores['mean']
            df_scores['{}_std'.format(algorithm)] = scores['std']
            printOut = pd.DataFrame(algorithms)
            printOut['mean'], printOut['std'] = scores['mean'], scores['std']
            excel_output(printOut, root, file_name='{}'.format(algorithm), csv=False)
        compare_models_plot(df_scores, 'classification')
        models_cls = [('MLP', best_models['MLP']),
                      ('SVM', best_models['SVM']),
                      ('RF', best_models['RF']),
                      ('KNN', best_models['KNN'])]
    else:
        models_cls = [('MLP', MLPClassifier(max_iter=10000, random_state=5)),
                      ('SVM', SVC()),
                      ('RF', RandomForestClassifier(random_state=5)),
                      ('KNN', KNeighborsClassifier())]

    # comparing different models
    scores_cls, _best_cls = compare_models(classification, models_cls, param, 'classification')
    compare_models_box_plot(scores_cls, param, 'classification')
    excel_output(scores_cls, 'classification/gridSearchModels', file_name='comparison', csv=False)
    best_cls = models_cls[2][1]

    # features importance
    X, y = split_xy(classification)
    models_cls[2][1].fit(X, y)
    importance_correlation_plot(classification, models_cls[2][1], 'classification')

    # confusion matrix
    training_cls, testing_cls = split_data(classification, pipelines, param, ['random', 0], 'classification')
    X_train, y_train = split_xy(training_cls)
    best_cls.fit(X_train, y_train)
    X_test, y_test = split_xy(testing_cls)
    confusion_matrix(best_cls, X_test, y_test, 'classification', 'true')
    confusion_matrix(best_cls, X_test, y_test, 'classification', 'pred')
    confusion_matrix(best_cls, X_test, y_test, 'classification', 'all')
    confusion_matrix(best_cls, X_test, y_test, 'classification', 'none')
    cls_report = class_report(best_cls, X_test, y_test, 'classification')
    excel_output(X_train, 'classification/bestModelPerformance', file_name='trainFeatureMatrixNorm', csv=True)

    # sensitivity analysis
    features_cls = [('All_in', ''),
                    ('Area', 'Area'),
                    ('Pipe_age', 'Pipe_age'),
                    ('Gas_FR', 'Gas_FR'),
                    ('Water_FR', 'Water_FR'),
                    ('Condensate_FR', 'Condensate_FR'),
                    ('Temperature', 'Temperature'),
                    ('Pressure', 'Pressure'),
                    ('CO2_content', 'CO2_content')]
    scores_sens_cls = sensitivity(classification, best_cls, pipelines, features_cls,
                                  param, ['random', 0], 'classification')
    sensitivity_plot(scores_sens_cls, 'classification')

# --------------------------------------------------------------------------------------------------------------------
# THE END
# --------------------------------------------------------------------------------------------------------------------
print('=> DONE!')
