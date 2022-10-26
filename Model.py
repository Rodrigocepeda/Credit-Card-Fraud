#%% Libraries -----------------------------------------------------------------
import pandas as pd
import numpy as np
from datetime import datetime, date
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from sklearn.linear_model import LinearRegression
pio.renderers.default = "browser"
from plotly.subplots import make_subplots
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Input, Dense
from scipy.stats.contingency import crosstab

#%% Initial data treatment ----------------------------------------------------
_train = pd.read_csv('data/fraudTrain.csv')
_test = pd.read_csv('data/fraudTest.csv')
data = pd.concat([_train, _test])


def initial_changes(dataset, columns_good):
    dataset = dataset.iloc[:, 1:]
    dataset.set_index('trans_num', inplace = True)
    dataset['trans_date_trans_time'] = dataset['trans_date_trans_time'].\
        apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    dataset['dob'] = dataset['dob'].\
        apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    dataset['age_when_trans'] = (dataset['trans_date_trans_time'] 
                                 - dataset['dob']).dt.days / 365
    dataset['hour'] = dataset['trans_date_trans_time'].dt.hour
    dataset['wday'] = dataset['trans_date_trans_time'].dt.weekday
    dataset['month'] = dataset['trans_date_trans_time'].dt.month
    #dataset_Y = dataset['is_fraud']
    dataset = dataset[columns_good]
    return dataset

columns_good = ['category', 'amt', 'gender', 'age_when_trans', 'lat', 'long',
                'city_pop', 'merch_lat', 'merch_long',
                'hour', 'wday', 'month','is_fraud']

def get_num_mapping(dataset_1, columns):
    map_dictionary = {}
    for c in columns:
        temp = list(np.unique(dataset_1[c]))
        temp_df = pd.DataFrame(temp)
        temp_df.columns = [c]
        temp_df['Map'] = np.arange(0, len(temp))
        map_dictionary[c] = temp_df
    return map_dictionary

def numerize_var(dataset_1, map_dictionary):
    columns_order = dataset_1.columns
    for c in map_dictionary.keys():
        dataset_1 = dataset_1.merge(map_dictionary[c], left_on = c, right_on = c)
        dataset_1.drop(columns = [c], inplace = True)
        dataset_1.rename(columns = {'Map':str(c)}, inplace = True)
        dataset_1 = dataset_1[columns_order]
    return dataset_1

def one_hot_encode_col(data2, columns):
    for c in columns:
        col = data2[c].copy()
        data2.drop(columns = [c], inplace = True)
        one_h = pd.get_dummies(col).astype(float)
        one_h.columns = [c + '_' + str(i) for i in range(one_h.shape[1])]
        data2 = pd.concat([data2, one_h], axis = 1)
    return data2

data = initial_changes(data, columns_good)
mapping = get_num_mapping(data, ['category', 'gender'])
data = numerize_var(data, mapping)
data = one_hot_encode_col(data, ['category', 'gender'])

#%% PCA -----------------------------------------------------------------------
def sep_data(dataset):
    data_Y = dataset['is_fraud']
    data_X = dataset.drop(columns = ['is_fraud'])
    return data_X, data_Y

_X, y = sep_data(data)

pca = PCA(n_components = 20)
X_pca = pd.DataFrame(pca.fit_transform(_X))

def train_test_val_split(x, y, train_size, val_size, test_size, random_state, shuffle):
    x_train, x_rem, y_train, y_rem = train_test_split(
        x, y, test_size = (val_size + test_size), random_state = random_state, shuffle = shuffle)
    x_val, x_test, y_val, y_test = train_test_split(
        x_rem, y_rem, test_size = test_size / (val_size + test_size),
        random_state = random_state, shuffle = shuffle)
    return x_train, x_val, x_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = train_test_val_split(
    X_pca, y, train_size = 0.7, val_size = 0.15, test_size = 0.15,
    random_state = 2, shuffle = True)

#%% Imbalance treatment -------------------------------------------------------
def _reduce(dataset,d_y, size):
    dataset['is_fraud'] = d_y
    dataset_f = dataset[dataset['is_fraud'] == 1].copy()
    dataset_nf = dataset[dataset['is_fraud'] == 0].copy().sample(size)
    dataset_r = pd.concat([dataset_f, dataset_nf])
    dataset_y = dataset_r['is_fraud']
    dataset_r.drop(columns = ['is_fraud'], inplace = True)
    return dataset_r, dataset_y

X_train, y_train = _reduce(X_train,y_train, 50000)

def _smote(dataset, dataset_Y):
    balance = SMOTE()
    dataset, dataset_Y = balance.fit_resample(dataset, dataset_Y)
    return dataset, dataset_Y

X_train, y_train = _smote(X_train, y_train)

#%%

# def _quantile_removal(dataset, dataset_y, columns_r, thres):
#     dataset_y.columns = ['is_fraud']
#     dataset_full = pd.concat([dataset, dataset_y], axis = 1)
    
#     dataset_nf = dataset_full[dataset_full['is_fraud'] == 0].copy()
#     for c in columns_r:
#         low_in = dataset_nf[dataset_nf['is_fraud'] == 0][c].quantile([0.05, 0.95]).iloc[0]
#         high_in = dataset_nf[dataset_nf['is_fraud'] == 0][c].quantile([0.05, 0.95]).iloc[1]
#         dataset_nf = dataset_nf[(dataset_nf[c] >= (low_in * (1 - thres))) & (dataset_nf[c] <= (high_in * (1 + thres)))]
    
#     dataset_f = dataset_full[dataset_full['is_fraud'] == 1].copy()
#     for c in columns_r:
#         low_in = dataset_f[dataset_f['is_fraud'] == 1][c].quantile([0.05, 0.95]).iloc[0]
#         high_in = dataset_f[dataset_f['is_fraud'] == 1][c].quantile([0.05, 0.95]).iloc[1]
#         dataset_f = dataset_f[(dataset_f[c] >= (low_in * (1 - thres))) & (dataset_f[c] <= (high_in * (1 + thres)))].copy()
    
#     dataset = pd.concat([dataset_nf, dataset_f])
#     dataset_y = dataset['is_fraud']
#     dataset.drop(columns = ['is_fraud'], inplace = True)
#     return dataset, dataset_y

# X_train, y_train = _quantile_removal(pd.DataFrame(X_train), pd.DataFrame(y_train), pd.DataFrame(X_train).columns, 0.001)


#%% Machine Learning ----------------------------------------------------------

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform (X_test)

model = Sequential()
model.add(Input(shape = X_train.shape[1]))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(256, activation = 'sigmoid'))
model.add(Dense(128, activation = 'sigmoid'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',
              metrics=['acc']) #Utilizamos 'binary_crossentropy' porque tenemos solo dos clases, 0 y 1.

_nepochs = 4
history = model.fit(X_train, 
                    y_train, 
                    epochs = _nepochs,
                    #batch_size = _batch_size,
                    verbose = True)

y_pred = model.predict(X_test)
y_pred[y_pred >= 0.5] = 1
y_pred[y_pred != 1] = 0

confussion_matrix_1 = crosstab(y_test, y_pred)[1]


from sklearn.tree import DecisionTreeClassifier
dcstree = DecisionTreeClassifier(random_state=42)
dcstree.fit(X_train, y_train)

importances = dcstree.feature_importances_
px.bar(importances)

y_pred = dcstree.predict(X_test)
y_pred[y_pred >= 0.5] = 1
y_pred[y_pred != 1] = 0
confussion_matrix = crosstab(y_test, y_pred)[1]
