from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

from database import *
from prepare_curvs import *

DFList=get_list()
keys=list(DFList['name'][pd.notna(DFList['Equipment'])].values)
Name=keys[0]
DF=get_DF(Name=Name)
app = Dash(__name__)


fig=plot_df_dim(DF,Name=Name)

app.layout = html.Div(children=[
    html.H1(children='Т - 100'),

    html.Div(children='Характеристики цифрового двойника: список поверхностей.'),
    dcc.Dropdown(keys, keys[0], id='dropdown-selection'),
    
    dcc.Graph(
        id='graph-content',
        figure=fig
    )
])
@callback(
    Output('graph-content', 'figure'),
    Input('dropdown-selection', 'value')
)
def update_graph(value):
    DF=get_DF(Name=value)
    return plot_df_dim(DF,Name=value)

#if __name__ == '__main__
app.run(debug=True)
