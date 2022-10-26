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

#%% Data ----------------------------------------------------------------------
_train = pd.read_csv('data/fraudTrain.csv')
_test = pd.read_csv('data/fraudTest.csv')

#_train = _train.iloc[0:100000, :]
#_test = _test.iloc[0:100000, :]

def initial_changes(dataset, columns_good):
    dataset = dataset.iloc[:, 1:]
    dataset.set_index('trans_num', inplace = True)
    dataset['trans_date_trans_time'] = dataset['trans_date_trans_time'].\
        apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    dataset['dob'] = dataset['dob'].\
        apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    dataset['age_when_trans'] = (dataset['trans_date_trans_time'] 
                                 - dataset['dob']).dt.days / 365
    #dataset_Y = dataset['is_fraud']
    dataset = dataset[columns_good]
    return dataset

columns_good = ['trans_date_trans_time', 'category', 'amt', 'gender', 'age_when_trans', 'city_pop', 'merch_lat',
                'merch_long', 'merchant', 'is_fraud']

test_X, train_X = initial_changes(_test, columns_good), initial_changes(_train, columns_good)

def get_num_mapping(dataset_1, dataset_2, columns):
    map_dictionary = {}
    for c in columns:
        temp_1 = list(np.unique(dataset_1[c]))
        temp_2 = list(np.unique(dataset_2[c]))
        temp = list(set(temp_1 + temp_2)) #Get unique
        temp_df = pd.DataFrame(temp)
        temp_df.columns = [c]
        temp_df['Map'] = np.arange(0, len(temp))
        map_dictionary[c] = temp_df
    return map_dictionary

def numerize_var(dataset_1, dataset_2, map_dictionary):
    columns_order = dataset_1.columns
    for c in map_dictionary.keys():
        dataset_1 = dataset_1.merge(map_dictionary[c], left_on = c, right_on = c)
        dataset_1.drop(columns = [c], inplace = True)
        dataset_1.rename(columns = {'Map':str(c)}, inplace = True)
        dataset_1 = dataset_1[columns_order]
    for c in map_dictionary.keys():
        dataset_2 = dataset_2.merge(map_dictionary[c], left_on = c, right_on = c)
        dataset_2.drop(columns = {c}, inplace = True)
        dataset_2.rename(columns = {'Map':str(c)}, inplace = True)
        dataset_2 = dataset_2[columns_order]
    return dataset_1, dataset_2

mapping = get_num_mapping(train_X, test_X, ['category', 'gender', 'merchant'])  
train_X, test_X = numerize_var(train_X, test_X, mapping)

#%% EDA -----------------------------------------------------------------------
#1) First, lets study about the location -------------------------------------
us_map = gpd.read_file('us/USA_States.shp')
map_plot = _train[['merch_lat', 'merch_long', 'is_fraud']]
geometry = [Point(xy) for xy in zip(map_plot['merch_long'],
                                    map_plot['merch_lat'])]
crs = {'init' : 'epsg:4326'}
geo_df = gpd.GeoDataFrame(map_plot, crs = crs,geometry = geometry)
fig, ax = plt.subplots(figsize = (40, 40))
us_map.plot(ax = ax, alpha = 0.4, color = 'grey')
geo_df.plot(ax = ax, markersize = 40, edgecolor='black', linewidth=1,
            color = 'lightblue')
fig.savefig('All.png',dpi = 450)
# -----------------------------------------------------------------------------
#2) Second, lets study the fraud by amount ------------------------------------
eda_amount = train_X[['amt', 'is_fraud']]
amt_bins = []
amt = np.arange(0, 1500, 25)
for i in amt:
    amt_bins.append(i)
eda_amount_df = pd.DataFrame(amt_bins)
eda_amount_df.columns = ['Amount_bins']
eda_amount_df['Total_trans'] = 0
eda_amount_df['Total_fraud'] = 0
eda_amount_df['Perc'] = 0
for i in range(len(amt)):
    temp = eda_amount[(eda_amount['amt'] >= amt[i]) & (eda_amount['amt'] < (amt[i] + 25))].copy()
    eda_amount_df.loc[i ,'Total_trans'] = temp.shape[0]
    eda_amount_df.loc[i ,'Total_fraud'] = temp['is_fraud'].sum()
eda_amount_df['Perc'] = eda_amount_df['Total_fraud'] / eda_amount_df['Total_trans']
eda_amount_df['Perc'] = eda_amount_df['Perc'] * 100
eda_amount_df.fillna(0, inplace = True)
eda_amount_df_2 = eda_amount_df.copy()
eda_amount_df_2.set_index('Amount_bins', drop = True, inplace = True)
eda_amount_df_2 = eda_amount_df_2[['Perc']]
eda_amount_df_2['Perc_2'] = 100 - eda_amount_df_2['Perc']
eda_amount_df_2.columns = ['Fraud', 'No fraud']

fig1 = px.bar(eda_amount_df_2, color_discrete_sequence = ['#780400','#011e4c'])
fig1.update_layout(
margin=dict(l=0, r=0, t=0, b=0))
fig1.update_layout(template="simple_white")

fig1.update_layout( # customize font and legend orientation & position
    font_family="Georgia",
    font_size = 20,
    legend=dict(
        font_size = 20, title=None, orientation = 'h', yanchor="bottom",
        y=1, x = 0.01, xanchor="left"))
fig1.update_layout(margin = dict(l = 0, r = 0, t = 0, b = 0))
fig1.update_yaxes(title=None)
fig1.update_xaxes(title=None)
fig1.update_layout(xaxis_tickprefix = '$', yaxis_tickformat = ',.')
fig1.update_layout(yaxis_ticksuffix = '%', yaxis_tickformat = ',.')
fig1.update_layout(paper_bgcolor='white', plot_bgcolor='white')
fig1.update_layout(legend = dict(bgcolor = 'rgba(255,255,255,0.1)'))
# -----------------------------------------------------------------------------

#3) 4 plots probability--------------------------------------------------------
four_plot = train_X[['trans_date_trans_time','age_when_trans', 'amt',
                     'gender', 'city_pop', 'is_fraud', 'category']].copy()
four_plot['hour'] = four_plot['trans_date_trans_time'].dt.hour
four_plot['Amount'] = four_plot['amt'].apply(lambda x: 500 * np.floor(x / 500))
four_plot['Age'] = four_plot['age_when_trans'].apply(lambda x: 5 * np.floor(x / 5))
four_plot['City Population'] = four_plot['city_pop'].apply(lambda x: 50000 * np.floor(x / 50000))
four_plot['wday'] = four_plot['trans_date_trans_time'].dt.weekday
four_plot['month'] = four_plot['trans_date_trans_time'].dt.month
from plotly.subplots import make_subplots
import plotly.graph_objects as go
fig1 = px.bar(four_plot.groupby('Amount')['is_fraud'].sum())
fig2 = px.bar(four_plot.groupby('Amount')['is_fraud'].mean() * 100)
fig3 = px.bar(four_plot.groupby('Age')['is_fraud'].sum())
fig4 = px.bar(four_plot.groupby('Age')['is_fraud'].mean() * 100)
fig5 = px.bar(four_plot.groupby('City Population')['is_fraud'].sum())
fig6 = px.bar(four_plot.groupby('City Population')['is_fraud'].mean() * 100)
fig7 = px.bar(four_plot.groupby('gender')['is_fraud'].sum())
fig8 = px.bar(four_plot.groupby('gender')['is_fraud'].mean() * 100)
fig9 = px.bar(four_plot.groupby('hour')['is_fraud'].sum())
fig10 = px.bar(four_plot.groupby('hour')['is_fraud'].mean() * 100)
fig11 = px.bar(four_plot.groupby('wday')['is_fraud'].sum())
fig12 = px.bar(four_plot.groupby('wday')['is_fraud'].mean() * 100)
fig13 = px.bar(four_plot.groupby('month')['is_fraud'].sum())
fig14 = px.bar(four_plot.groupby('month')['is_fraud'].mean() * 100)
fig15 = px.bar(four_plot.groupby('category')['is_fraud'].sum())
fig16 = px.bar(four_plot.groupby('category')['is_fraud'].mean() * 100)

fig = make_subplots(rows=8, cols=2, subplot_titles=("Amount (total)", "Amount (perc)", "Age (total)", "Age (perc)",
                                                    "City Pop (total)", "City Pop (perc)", "Gender (total)", "Gender (perc)",
                    "Hour (total)", "Hour (perc)", "Week Day (total)", "Week Day (perc)","Month (total)", "Month (perc)",
                    "Category (total)", "Category (perc)"),
                    vertical_spacing = 0.085)
for d in fig1.data:
    fig.add_trace((go.Bar(x=d['x'], y=d['y'], name = d['name'], marker=dict(color = '#01224e'))), row=1, col=1)        
for d in fig2.data:
    fig.add_trace((go.Bar(x=d['x'], y=d['y'],  name = d['name'], marker=dict(color = '#c45666'))), row=1, col=2)   
for d in fig3.data:
    fig.add_trace((go.Bar(x=d['x'], y=d['y'],  name = d['name'], marker=dict(color = '#01224e'))), row=2, col=1)   
for d in fig4.data:
    fig.add_trace((go.Bar(x=d['x'], y=d['y'],  name = d['name'], marker=dict(color = '#c45666'))), row=2, col=2)   
for d in fig5.data:
    fig.add_trace((go.Bar(x=d['x'], y=d['y'],  name = d['name'], marker=dict(color = '#01224e'))), row=3, col=1)    
for d in fig6.data:
    fig.add_trace((go.Bar(x=d['x'], y=d['y'],  name = d['name'], marker=dict(color = '#c45666'))), row=3, col=2)   
for d in fig7.data:
    fig.add_trace((go.Bar(x=d['x'], y=d['y'],  name = d['name'], marker=dict(color = '#01224e'))), row=4, col=1)
for d in fig8.data:
    fig.add_trace((go.Bar(x=d['x'], y=d['y'],  name = d['name'], marker=dict(color = '#c45666'))), row=4, col=2)
for d in fig9.data:
    fig.add_trace((go.Bar(x=d['x'], y=d['y'],  name = d['name'], marker=dict(color = '#01224e'))), row=5, col=1)
for d in fig10.data:
    fig.add_trace((go.Bar(x=d['x'], y=d['y'],  name = d['name'], marker=dict(color = '#c45666'))), row=5, col=2)
for d in fig11.data:
    fig.add_trace((go.Bar(x=d['x'], y=d['y'],  name = d['name'], marker=dict(color = '#01224e'))), row=6, col=1)
for d in fig12.data:
    fig.add_trace((go.Bar(x=d['x'], y=d['y'],  name = d['name'], marker=dict(color = '#c45666'))), row=6, col=2)
for d in fig13.data:
    fig.add_trace((go.Bar(x=d['x'], y=d['y'],  name = d['name'], marker=dict(color = '#01224e'))), row=7, col=1)
for d in fig14.data:
    fig.add_trace((go.Bar(x=d['x'], y=d['y'],  name = d['name'], marker=dict(color = '#c45666'))), row=7, col=2)
for d in fig15.data:
    fig.add_trace((go.Bar(x=d['x'], y=d['y'],  name = d['name'], marker=dict(color = '#01224e'))), row=8, col=1)
for d in fig16.data:
    fig.add_trace((go.Bar(x=d['x'], y=d['y'],  name = d['name'], marker=dict(color = '#c45666'))), row=8, col=2)
    
for row_ in range(1,9):
    for col_ in range(1,9):
        fig.update_xaxes(showgrid=False, row=row_, col=col_)
        fig.update_yaxes(showgrid=False, row=row_, col=col_)
        fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#f2f4f4', mirror = True, row=row_, col=col_)
        fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#f2f4f4', mirror = True, row=row_, col=col_)
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', row=row_, col=col_)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', row=row_, col=col_)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

fig.update_layout(showlegend = False)
fig.update_layout(paper_bgcolor='white', plot_bgcolor='white')   
fig.update_layout(
    font_family="Georgia",
    font_size = 18,
    font_color = 'black',
    margin=dict(l=10, r=10, t=10, b=10))

fig.update_layout(yaxis2_ticksuffix = '%', yaxis_tickformat = ',.')
fig.update_layout(yaxis4_ticksuffix = '%', yaxis_tickformat = ',.')
fig.update_layout(yaxis6_ticksuffix = '%', yaxis_tickformat = ',.')
fig.update_layout(yaxis8_ticksuffix = '%', yaxis_tickformat = ',.')
fig.update_layout(yaxis10_ticksuffix = '%', yaxis_tickformat = ',.')
fig.update_layout(yaxis12_ticksuffix = '%', yaxis_tickformat = ',.')
fig.update_layout(yaxis14_ticksuffix = '%', yaxis_tickformat = ',.')
fig.update_layout(yaxis16_ticksuffix = '%', yaxis_tickformat = ',.')
fig.show()
#------------------------------------------------------------------------------
# Contour ---------------------------------------------------------------------
train_X_cont = train_X[['amt','city_pop','is_fraud']]
train_X_cont = train_X_cont[train_X_cont['amt'] <=1200].copy()
train_X_cont = train_X_cont[train_X_cont['city_pop'] <= 500000].copy()
X = np.array(train_X_cont[['amt', 'city_pop']])
y = np.array(train_X_cont['is_fraud'])
# Parameters
mesh_size = 100
mesh_size_2 = 1000
margin = 0.01
# Create a mesh grid on which we will run our model
x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
xrange = np.arange(x_min, x_max, mesh_size)
yrange = np.arange(y_min, y_max, mesh_size_2)
xx, yy = np.meshgrid(xrange, yrange)

# Model 
knn = KNeighborsClassifier(n_neighbors = 10, weights = 'uniform')
knn.fit(X, y)

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# Plot 
fig = go.Figure()
fig.add_trace(go.Contour(
    x = xrange, y = yrange, z = Z,
    colorscale = 'RdYlBu', showscale = False, opacity = 0.75))
fig.update_layout(
    font_family="Georgia",
    font_size = 20,
    font_color = 'black',
    margin=dict(l=10, r=10, t=10, b=10))
fig.update_layout(xaxis_tickprefix = '$', yaxis_tickformat = ',.')
fig.update_layout(showlegend = False, margin=dict(l=0, r=0, t=0, b=0),
                  font_family = 'Georgia', paper_bgcolor = 'white',
                  plot_bgcolor = 'white')
fig.update_coloraxes(showscale = False)
# -----------------------------------------------------------------------------
train_X_cut = train_X.copy()
train_X_cut = train_X[(train_X['amt'] <= 1300) & (train_X['amt'] >= 250)].copy()
train_X_cut['hour'] = train_X_cut['trans_date_trans_time'].dt.hour

train_X_cut.groupby('category')['is_fraud'].mean()


_train[_train['state'] == 'DE']
state = _train.groupby('state')['is_fraud'].mean()

# PCA -------------------------------------------------------------------------
pca_data = train_X.copy()
pca_data['hour'] = pca_data['trans_date_trans_time'].dt.hour
pca_data['wday'] = pca_data['trans_date_trans_time'].dt.weekday
pca_data['month'] = pca_data['trans_date_trans_time'].dt.month

pca_data = pca_data[['category','amt','gender','age_when_trans','city_pop','merch_lat',
                     'merch_long','merchant','hour','wday','month']]


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(pca_data)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PCA1', 'PCA2'])

px.scatter(x = principalDf['PCA1'], y = principalDf['PCA2'], color = train_X['is_fraud'])

# Categoty and hour ---------------------------------------------------------------------
train_X_cat_hour = train_X[['category','trans_date_trans_time','is_fraud']].copy()
train_X_cat_hour['hour'] = train_X_cat_hour['trans_date_trans_time'].dt.hour
train_X_cat_hour.drop(columns = ['trans_date_trans_time'], inplace = True)

fig = make_subplots(rows=7, cols=2, subplot_titles=('Fitness', 'Shopping Online',
                                                    'Misc Online','Dinning',
                                                    'Misc', 'Self care',
                                                    'Trave', 'Gas Transport',
                                                    'Home', 'Grocery',
                                                    'Entertainment', 'Grocery Online',
                                                    'Kids', 'Shopping'), vertical_spacing = 0.085)
_contador = _row = _col = 0
for c in list(np.unique(train_X_cat_hour['category'])):
    _contador = _contador + 1
    if _contador == 1: _row = _row + 1; _col = 1
    if _contador == 2: _contador = 0; _col = 2
    
    _temp = train_X_cat_hour[train_X_cat_hour['category'] == c]
    fig1 = px.bar(_temp.groupby('hour')['is_fraud'].sum())

    for d in fig1.data:
        fig.add_trace((go.Bar(x = d['x'],
                              y = d['y'],
                              name = d['name'], 
                              marker=dict(color = '#01224e'))),
                      row = _row, col = _col)
for row_ in range(1,8):
    for col_ in range(1,8):
        fig.update_xaxes(showgrid=False, row=row_, col=col_)
        fig.update_yaxes(showgrid=False, row=row_, col=col_)
        fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#f2f4f4', mirror = True, row=row_, col=col_)
        fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#f2f4f4', mirror = True, row=row_, col=col_)
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', row=row_, col=col_)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', row=row_, col=col_)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

fig.update_layout(showlegend = False)
fig.update_layout(paper_bgcolor='white', plot_bgcolor='white')   
fig.update_layout(
    font_family="Georgia",
    font_size = 18,
    font_color = 'black',
    margin=dict(l=10, r=10, t=10, b=10))


# Categoty and amount ---------------------------------------------------------------------
train_X_cat_amt = train_X[['category','amt','is_fraud', 'age_when_trans']].copy()

fig = make_subplots(rows=7, cols=2, subplot_titles=('Fitness', 'Shopping Online',
                                                    'Misc Online','Dinning',
                                                    'Misc', 'Self care',
                                                    'Trave', 'Gas Transport',
                                                    'Home', 'Grocery',
                                                    'Entertainment', 'Grocery Online',
                                                    'Kids', 'Shopping'), vertical_spacing = 0.085)
_contador = _row = _col = 0
for c in list(np.unique(train_X_cat_amt['category'])):
    _contador = _contador + 1
    if _contador == 1: _row = _row + 1; _col = 1
    if _contador == 2: _contador = 0; _col = 2
    
    _temp = train_X_cat_amt[train_X_cat_amt['category'] == c]
    
    _llen = len(_temp[_temp['is_fraud'] == 1])
    _temp_nf = _temp[_temp['is_fraud'] == 0].sample(_llen)
    
    
    _temp = _temp[_temp['is_fraud'] == 1].copy()
    _temp = pd.concat([_temp, _temp_nf])
    fig1 = px.scatter(_temp, x = 'age_when_trans', y = 'amt', color = 'is_fraud')

    # for d in fig1.data:
    fig.add_trace((go.Scatter(x = fig1.data[0]['x'],
                            y = fig1.data[0]['y'],
                            name = fig1.data[0]['name'], 
                            mode = 'markers',
                            marker=dict(color = fig1.data[0]['marker']['color'],
                                        size=6,
                                        line=dict(
                                            color='black', width=1),
                                        colorscale = ['red', 'green']))),
                    row = _row, col = _col)
for row_ in range(1,8):
    for col_ in range(1,8):
        fig.update_xaxes(showgrid=False, row=row_, col=col_)
        fig.update_yaxes(showgrid=False, row=row_, col=col_)
        fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#f2f4f4', mirror = True, row=row_, col=col_)
        fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#f2f4f4', mirror = True, row=row_, col=col_)
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', row=row_, col=col_)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', row=row_, col=col_)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

fig.update_layout(showlegend = False)
fig.update_layout(paper_bgcolor='white', plot_bgcolor='white')   
fig.update_layout(
    font_family="Georgia",
    font_size = 18,
    font_color = 'black',
    margin=dict(l=10, r=10, t=10, b=10))

# Correlation
train_X_corr = train_X.copy()
train_X_corr['hour'] = train_X_corr['trans_date_trans_time'].dt.hour
train_X_corr['wday'] = train_X_corr['trans_date_trans_time'].dt.weekday
train_X_corr['month'] = train_X_corr['trans_date_trans_time'].dt.month

train_X_corr = train_X_corr[['amt','age_when_trans','city_pop','merch_lat',
                     'merch_long','hour','wday','month']]
train_X_corr =  train_X_corr.corr()

px.imshow(train_X_corr)
fig = px.imshow(train_X_corr.corr(), aspect="auto",
                color_continuous_scale = 'Bluered_r')
fig.update_layout( # customize font and legend orientation & position
    font_family="Georgia")
fig.update_layout(
    font_family="Georgia",
    font_size = 21,
    font_color = 'black',
    margin=dict(l=10, r=10, t=10, b=10))
fig.update_yaxes(title = None)
fig.update_xaxes(title= None)
fig.update_layout(paper_bgcolor='white',
               plot_bgcolor='white')



#%% Check simple rules --------------------------------------------------------
test_X['simple_fraud'] = 0
test_X['hour'] = test_X['trans_date_trans_time'].dt.hour

test_X.loc[test_X[((test_X['amt'] <= 1300) & (test_X['amt'] >= 600)) & ((test_X['hour'] <= 4) | (test_X['hour'] >= 22)) ].index, 'simple_fraud'] = 1
test_X.loc[test_X[((test_X['amt'] <= 1300) & (test_X['amt'] >= 250)) & ((test_X['hour'] <= 4) | (test_X['hour'] >= 22)) & test_X['category'] == 12 ].index, 'simple_fraud'] = 1

from scipy.stats.contingency import crosstab
a = crosstab(test_X['is_fraud'], test_X['simple_fraud'])


