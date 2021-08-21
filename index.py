
"""
Created - July 2021
Author - caffsean

This is the main index of the app.
THIS IS THE PAGE YOU NEED TO RUN TO MAKE THE APP WORK
"""


import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input

### Don't remove the callbacks even though PyCharm says they're unused imports!
from app import app
import explorer_layouts
import explorer_callbacks
import vector_space_layout
import vector_space_callbacks
import topic_callbacks
import topic_layout
import hashtag_layout
import network_final_dash



dropdown_style = {'color': 'blue','background-color': '#212121'}
graph_background = {'backgroundColor': '#22303D','padding': '10px 10px 10px 10px','border':'4px solid', 'border-radius': 10}
graph_background_blue = {'backgroundColor': '#2D455C','padding': '10px 10px 10px 10px','border':'4px solid','border-color':'black','border-radius': 10}


SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "25rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}


CONTENT_STYLE = {
    "margin-left": "25rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "background-color": '#2D455C'
}

sidebar = html.Div(
    [
        html.H2("International Relations of Twitter", className="display-4"),
        html.Hr(),
        html.P(
            "A web application for exploring diplomatic messaging around the world.", className="lead"
        ),
        dbc.Nav(
            [
                html.Br(),
                dbc.NavLink("Explore Dashboard - Network Level", href="/", active="exact"),
                dbc.NavLink("Explore Dashboard - Embassy Level", href="/page-2", active="exact"),
                dbc.NavLink("Explore Dashboard - Comparative", href="/page-3", active="exact"),
                dbc.NavLink("Hashtag Dynamics Dashboard", href="/page-4", active="exact"),
                dbc.NavLink("Topic Modeling Dashboard", href="/page-5", active="exact"),
                dbc.NavLink("Network Analysis Dashboard", href="/page-6", active="exact"),
                dbc.NavLink("Document Embeddings Dashboard", href="/page-7", active="exact"),
                dbc.NavLink("Word Embeddings Dashboard", href="/page-8", active="exact"),

            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)
location = dcc.Location(id="url")

app.validation_layout = html.Div([
    sidebar,
    content,
])

app.layout = html.Div([location, sidebar, content])


@app.callback(Output("page-content", "children"),
            [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return explorer_layouts.layout
    elif pathname == "/page-2":
        return explorer_layouts.layout_2
    elif pathname == "/page-3":
        return explorer_layouts.layout_3
    elif pathname == "/page-4":
        return hashtag_layout.hashtag_layout
    elif pathname == "/page-5":
        return topic_layout.redo_topic_layout
    elif pathname == "/page-6":
        return network_final_dash.network_layout
    elif pathname == "/page-7":
        return vector_space_layout.vector_layout
    elif pathname == "/page-8":
        return vector_space_layout.vector_layout_2
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

if __name__ == '__main__':
    app.run_server(debug=True,port=4088)
