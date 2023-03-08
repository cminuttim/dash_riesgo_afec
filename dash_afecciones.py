#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Carlos Minutti Martinez <carlos.minutti@iimas.unam.mx>
@author: Miguel Félix Mata Rivera <mmatar@ipn.mx >


Resumen:
    
El sistema desarrollado permite modelar el riesgo de defunción y severidad hospitalaria asociada a diferentes afecciones médicas en pacientes hospitalizados. El modelo estadístico toma en cuenta diferentes factores, como la edad, peso, procedencia, primera hospitalización y las afecciones presentadas por el paciente para estimar el riesgo de defunción y la severidad de la condición del paciente.

Valores superiores a 50% indican un mayor riesgo de defunción y mayor severidad, mientras que valores menores al 50% indican mayor posibilidad de sobrevivencia y menor severidad. Es importante mencionar que el sistema no se utiliza para determinar los riesgos de pacientes específicos, sino que es un modelado estadístico para estudiar cómo interactúan conjuntamente los diferentes factores en cada enfermedad, en hospitalizaciones que ocurren dentro de la Zona del Valle de México.

El modelo final es un modelo ponderado de Gradient Boosting Models (GBM), Árboles de Regresión y Regresión Logística. El archivo PICKLE original de los modelos no se incluye por razones de privacidad de los datos, en cambio se incluye un archivo PICKLE generado a partir de datos sintéticos.
"""

import pickle
import pandas as pd
import numpy as np
import plotly.express as px 
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree



# Load data (deserialize)
with open('risk_models-rnd.pickle', 'rb') as handle:
    models_dict = pickle.load(handle)
    

models = models_dict['models']
models_c = models_dict['models_c']


# model variables    
model_vars = pd.Series(models_dict['model_vars'])

model_vars = model_vars[~model_vars.str.contains(':')]
model_vars = model_vars[~model_vars.str.contains('I\(')]
model_vars = model_vars[~model_vars.isin(['EDAD', 'PESO', 'SEXO_M', 'TALLA', 'TIMEID'])]

otr_vars = ['PROCED_2', 'PROCED_3', 'PROCED_4', 'PROCED_9', 'VEZ_2', 'VEZ_9']
afec_vars = model_vars[~model_vars.isin(otr_vars)]
base_vars = ['EDAD', 'PESO', 'TALLA', 'SEXO_M']

var_names = np.concatenate([base_vars,afec_vars.values, otr_vars]) 

X_d = pd.DataFrame(0, index=np.arange(1), columns=var_names)
X_d[base_vars] = [24, 60, 156, 0]
X_d['PROCED_2'] = 0
X_d['TIMEID'] = 48

proced_list = ['CONSULTA EXTERNA', 'URGENCIAS', 'REFERIDO', 'OTRO', '(N.E.)']
vez_list = ['PRIMERA VEZ', 'SUBSECUENTE', '(N.E.)']

affec_name = models_dict['affec_name']
models_name = models_dict['models_name'] 
models_select = ['Todos', 'Crónicas'] + models



def min_max_scaler(X, xmin=None, xmax=None):
    
    if type(xmin)==type(None):
        xmin = X.min()
    if type(xmax)==type(None):
        xmax = X.max()
    
    X_tmp = X.copy()
        
    # constant values to 0
    idx = xmin.index[xmin==xmax]
    idx = idx[idx.isin(X_tmp.columns)]
    if idx.shape[0]>0:
        X_tmp[idx] = 0
    
    idx = xmin.index[(xmin<xmax) * ((xmin!=0) + (xmax!=1))]
    idx = idx[idx.isin(X_tmp.columns)] # only available columns
    
    X_tmp[idx] = (X_tmp[idx]-xmin[idx])/(xmax[idx]-xmin[idx])

    return (X_tmp, xmin, xmax)


def risk_bar_plot(models_sel):
    
    y_est = []
    auc = []
    #cie_desc = []
    #print(models_sel)
    for cie in models_sel:

        #---------------
        # Tree model
        #---------------
        tree_model = models_dict[cie+'_TREE']
        X_d_tree = X_d.reindex(labels=models_dict[cie+'_TREE_VARS'],axis=1)
        X_d_tree.fillna(0, inplace = True)
        y_est_tree = tree_model.predict(X_d_tree)


        #---------------
        # GBM
        #---------------
        gbm = models_dict[cie+'_GBM']
        y_est_gbm = gbm.predict(X_d_tree)

        #---------------
        # Logistic model
        #---------------
        xmin = models_dict[cie+'_xmin']
        xmax = models_dict[cie+'_xmax']

        # dataset
        X_s, xmin, xmax = min_max_scaler(X_d, xmin, xmax)
        # X_s.assign(rt=0.5)
        # X_s['rt'] = y_est_tree
        
        # prediction
        cie_model = models_dict[cie+'_LOGISTIC']        
        y_est.append(0.45*cie_model.predict(X_s)[0] + 0.45*y_est_gbm[0] + 0.1*y_est_tree[0])
        auc.append(np.round(models_dict[cie+'_AUC'],2))
        
        

    df = {'cie': models_sel, 'risk': y_est, 'desc': models_name[models_sel], 'auc':auc}
    risk = pd.DataFrame.from_dict(df)
    risk.set_index(risk['cie'])
    risk['risk'] = np.round(risk['risk']*100,2)
    
    fig = px.bar(risk, x='cie', y='risk', title="Riesgo para diferentes CIE como causa base",
                 hover_data=['cie', 'desc', 'risk', 'auc'],
                 color='risk', range_y=[0,100], color_continuous_scale='peach', template="plotly_white") #ylorrd, bluered, peach
    
    
    
    return fig



app = dash.Dash(__name__)


edad_list = range(121)
sexo_list = ['M', 'F']
talla_list = range(20, 251)
peso_list = range(201)

app.layout = html.Div([
    html.H1("Modelos de riesgo", style={'text-align': 'center', 'color': '#666666'}),
    
    html.Div("Edad:", style={'color': '#333333', 'width':'150px', 'float':'left', 'margin-left':'20px'}),
    html.Div("Peso:", style={'color': '#333333', 'width':'150px', 'float':'left', 'margin-left':'10px'}),
    html.Div("Sexo:", style={'color': '#333333', 'width':'150px', 'float':'left', 'margin-left':'10px'}),
    html.Br(),

    
    dcc.Dropdown(
        id='edad-select', 
        options=[{'label': i, 'value': i} for i in edad_list],
        value=X_d['EDAD'][0],
        style={'width':'150px','float':'left', 'margin-left':'10px'}
    ),
    
    dcc.Dropdown(
        id='peso-select', 
        options=[{'label': i, 'value': i} for i in peso_list],
        value=X_d['PESO'][0], 
        style={'width':'150px','float':'left', 'margin-left':'10px'}
    ),

    dcc.Dropdown(
        id='sexo-select', 
        options=[{'label': i, 'value': i} for i in sexo_list],
        value=('F' if X_d['SEXO_M'][0]==0 else 'M'), 
        style={'width':'150px','float':'left', 'margin-left':'10px'}
    ),

    
    html.Br(),
    html.P(' ', style={'height':'20px'}),
    html.Div("Procedencia:", style={'color': '#333333', 'width':'250px', 'float':'left', 'margin-left':'20px'}),
    html.Div("Primera hospitalización:", style={'color': '#333333', 'width':'250px', 'float':'left', 'margin-left':'10px'}),
    html.Br(),

    dcc.Dropdown(id='proced-select', options=[{'label': i, 'value': i} for i in proced_list],
                           value=proced_list[0], style={'width': '250px','float':'left', 'margin-left':'10px'}),

    dcc.Dropdown(id='vez-select', options=[{'label': i, 'value': i} for i in vez_list],
                           value=vez_list[0], style={'width': '250px','float':'left', 'margin-left':'10px'}),

    html.Br(),
    html.P(' ', style={'height':'20px'}),
    html.Div("Afecciones:", style={'color': '#333333', 'margin-left':'20px'}),

    
    dcc.Dropdown(
        id='afec-list',
        options=[{'label': f"{i} - {affec_name[i]}", 'value': i} for i in afec_vars],
        value=[],  # default value to show
        multi=True,
        searchable=True,
        style={'width':'800px', 'margin-left':'10px'}
    ),

    html.P(' ', style={'height':'20px'}),
    html.Div("Modelos:", style={'color': '#333333', 'margin-left':'20px'}),
    dcc.Dropdown(
        id='models-list',
        options=[{'label': (f"{i} - {models_name[i]}" if (i!=models_select[0] and i!=models_select[1]) else i), 'value': i} for i in models_select],
        value=[models_select[0]],  # default value to show
        multi=True,
        searchable=True,
        style={'width':'800px', 'margin-left':'10px'}
    ),
    html.Br(),    
        
    dcc.Graph(id='risk-graph', figure={})
])




@app.callback(
    Output('risk-graph', 'figure'),
    [Input('edad-select', 'value'),
    Input('peso-select', 'value'),
    Input('sexo-select', 'value'),
    #Input('talla-select', 'value'),
    Input('proced-select', 'value'),
    Input('vez-select', 'value'),
    Input('afec-list', 'value'),
    Input('models-list', 'value')]
)
def update_graph_p(edad, peso, sexo, proced, vez, afec, mod):
    X_d['EDAD'] = edad
    X_d['PESO'] = peso
    X_d['SEXO_M'] = 1 if sexo=='M' else 0
    #X_d['TALLA'] = talla
    
    X_d[['PROCED_2', 'PROCED_3', 'PROCED_4', 'PROCED_9']] = [0,0,0,0]
    if proced==proced_list[1]:
        X_d[['PROCED_2']] = 1
    elif proced==proced_list[2]:
        X_d[['PROCED_3']] = 1
    elif proced==proced_list[3]:
        X_d[['PROCED_4']] = 1
    elif proced==proced_list[4]:
        X_d[['PROCED_9']] = 1
        
    X_d[['VEZ_2', 'VEZ_9']] = [0,0]
    if vez==vez_list[1]:
        X_d[['VEZ_2']] = 1
    elif vez==vez_list[2]:
        X_d[['VEZ_9']] = 1
    #print(X_d[['VEZ_2', 'VEZ_9']])
    
    X_d[afec_vars] = 0
    X_d[afec] = 1
    #print(afec)

    #print(mod)
    if pd.Series(models_select[0]).isin(mod)[0]:
        models_sel = models
    elif pd.Series(models_select[1]).isin(mod)[0]:
        models_sel = models_c        
    else:
        models_sel = mod

    
    return risk_bar_plot(models_sel)




# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=False)