from dash import dcc, html, Input, Output, no_update, Dash
import plotly.graph_objects as go
import pandas as pd
import pathlib

app = Dash(__name__)
server = app.server

file_path = "/rds/general/user/cb1319/home/GEOM3D/Geom3D/fragment_experiment_2/interactive_results.pkl"
morgan_keys = pd.read_pickle(file_path)

fig = go.Figure(data=[
    go.Scatter(
        x=morgan_keys['pca1'],
        y=morgan_keys['pca2'],
        mode="markers",
        marker=dict(
            color='blue',  # You can set the color as per your requirement
            size=10,
            opacity=0.8,
        ),
    )
])

fig.update_layout(
    xaxis=dict(title='PCA1'),
    yaxis=dict(title='PCA2'),
    plot_bgcolor='rgba(255,255,255,0.1)',
    width=800,
    height=800,
)

app.title = "PCA Plot with Molecular Images"
app.layout = html.Div([
    dcc.Graph(id="graph", figure=fig, clear_on_unhover=True),
    dcc.Tooltip(id="graph-tooltip"),
])

@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("graph", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    num = hoverData["points"][0]["pointNumber"]

    im_url = morgan_keys['img'][num]

    children = [
        html.Div(children=[
            html.Img(src=im_url, style={"width": "100%"}),
        ],
        style={'width': '200px', 'white-space': 'normal'})
    ]

    return True, no_update, children

if __name__ == "__main__":
    app.run_server(debug=True, port=8057)
