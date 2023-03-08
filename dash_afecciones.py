#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Carlos Minutti Martinez <carlos.minutti@iimas.unam.mx>
@author: Miguel Félix Mata Rivera <mmatar@ipn.mx >


Resumen:
    
El sistema desarrollado permite modelar el riesgo de defunción y severidad hospitalaria asociada a diferentes afecciones médicas en pacientes hospitalizados. El modelo estadístico toma en cuenta diferentes factores, como la edad, peso, procedencia, primera hospitalización y las afecciones presentadas por el paciente para estimar el riesgo de defunción y la severidad de la condición del paciente.

Valores superiores a 50% indican un mayor riesgo de defunción y mayor severidad, mientras que valores menores al 50% indican mayor posibilidad de sobrevivencia y menor severidad. Es importante mencionar que el sistema no se utiliza para determinar los riesgos de pacientes específicos, sino que es un modelado estadístico para estudiar cómo interactúan conjuntamente los diferentes factores en cada enfermedad, en hospitalizaciones que ocurren dentro de la Zona del Valle de México.

El modelo final es un modelo ponderado de Gradient Boosting Models (GBM), Árboles de Regresión y Regresión Logística. El archivo PICKLE original de los modelos no se incluye por razones de privacidad de los datos, en cambio se incluye un archivo PICKLE generado a partir de datos sintéticos.


Licencia:    
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS


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