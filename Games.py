import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output, State
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
import json

steam = pd.read_csv('./data/steam.csv')
steamspy_tag_data = pd.read_csv('./data/steamspy_tag_data.csv')
steam_description_data = pd.read_csv('./data/steam_description_data.csv')
steam_media_data = pd.read_csv('./data/steam_media_data.csv')
steam_requirements_data = pd.read_csv('./data/steam_requirements_data.csv')
steam_support_info = pd.read_csv('./data/steam_support_info.csv')

steam['pos_rating_ratio'] = steam['positive_ratings']/(steam['positive_ratings']+steam['negative_ratings'])
steam['owners_range_max'] = steam['owners'].str.split('-').map(lambda x: int(x[1]))
steam['max_income'] = steam['owners_range_max'] * steam['price']
features = steamspy_tag_data.values[:, 1:]

norm_const = features.sum(1).reshape(-1,1)
norm_const[norm_const==0]=1
features = features/norm_const
similarities = cosine_similarity(features, features)
appids = steamspy_tag_data.values[:, 0]

external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    {
        'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
        'crossorigin': 'anonymous'
    }
]


def yearcounts():
    year_game_count = steam.release_date.astype("datetime64").map(lambda x: x.year).value_counts().sort_index()
    return px.bar(year_game_count, x=year_game_count.index, y=year_game_count.values,
                  title='Games per year', labels=dict(index="Year", y="Count"))


def developercountstop10():
    developergamescount = steam.developer.value_counts().head(10)
    return px.bar(developergamescount, x=developergamescount.index, y=developergamescount.values,
                  title='Games by developer', labels=dict(index="Developer", y="Count"))

def genrestop10():
    genres = steam.genres.str.split(';').apply(pd.Series).unstack().T.dropna().reset_index(1, drop=True).value_counts().head(10)
    return px.bar(genres, x=genres.index, y=genres.values, title='Games by genre', labels=dict(index="Genre", y="Count"))


def game_names_dict():
    un_games = steam[['appid', 'name']].dropna().apply(lambda x: {'value': x['appid'], 'label': x['name']}, axis=1)
    return list(un_games)

def platforms():
    platforms = steam.platforms.str.split(';').apply(pd.Series).unstack().T.dropna().reset_index(1,drop=True).value_counts()
    return px.bar(platforms, x=platforms.index, y=platforms.values, title='Games count by platform', labels=dict(index="Platform", y="Count"))

app = dash.Dash(external_stylesheets=external_stylesheets)

@app.callback(
       Output(component_id = 'developercount', component_property = 'figure'),
        [Input(component_id = 'plot', component_property = 'n_clicks')],
          [State(component_id = 'stat', component_property = 'value')]
)
def plotdevcount(n_clicks,stat):
    if stat == 'Count':
        developergamescount = steam.developer.value_counts().head(10)
        fig = px.bar(developergamescount, x = developergamescount.index, y = developergamescount.values,
                  title='Games by developer', labels=dict(index="Developer", y="Count"))
        fig.update_xaxes(visible = False)
        return fig
    else:
        df = steam.groupby("developer")[stat].mean().sort_values(ascending = False).head(10)
        fig = px.bar(df, x = df.index, y = df.values, labels = dict(y = stat.replace("_"," ")))
        fig.update_xaxes(visible = False)
        return fig

def top_similar_games(app_id):
    similar_games = np.argsort(similarities[list(appids).index(app_id),:])[::-1]
    similar_games_similarities = np.sort(similarities[list(appids).index(app_id),:])[::-1]
    sim_df = pd.DataFrame({'appid':appids[similar_games], 'sim':similar_games_similarities})
    sim_dat = pd.merge(steam, sim_df, on='appid', how='inner')
    return sim_dat.sort_values('sim', ascending=False).head(10)

@app.callback(
    Output(component_id='game_content', component_property='children'),
    Input(component_id='game_name', component_property='value')
)
def update_output_div(app_id):
    app_id = 10 if app_id is None else app_id
    game = steam[steam.appid == app_id]
    game_desc = steam_description_data[steam_description_data.steam_appid == app_id]
    img = steam_media_data[steam_media_data.steam_appid == app_id].header_image.values[0]
    game_name = game.name.values[0]
    game_about_text = game_desc.about_the_game.values[0]
    game_about_text = BeautifulSoup(game_about_text, "lxml").text

    top_sim_games = top_similar_games(app_id)
    sim_games_fig = px.bar(top_sim_games, x=top_sim_games.name, y=top_sim_games.sim, title=f'Top 10 similar games to {game_name}', labels=dict(name="Games", sim="Similarities"))
    ret_div = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H1(game_name, id="game_name_text"),
                            html.H4(game_about_text, id="game_about"),
                        ], className="six columns"),
                    html.Img(src=img, id="body-image", className="six columns")
                ]
            ),
            html.Div([dcc.Graph(id='top_games', figure=sim_games_fig)], className='twelfe columns')
        ]
    )
    return ret_div

app.layout = html.Div(
    [
        dcc.Tabs(id='tabs', value='tab1', children=
        [
            dcc.Tab(label='Overview', value='Games1', children = 
            [
                html.Div(
                [
                    html.Div([dcc.Graph(id='yearcounts', figure = yearcounts())], className='six columns'),
                    html.Div([dcc.Graph(id='genres', figure = genrestop10())], className = 'six columns')
                ]),
                
                html.Div(
                [
                    html.Div([dcc.Graph(id='developercount')], className='six columns'),
                    html.Div([dcc.Graph(id='platforms', figure = platforms())], className = 'six columns')
                ]),
                
                                
                html.Div(
                [ 
                    html.Div([dcc.Dropdown(id = 'stat', options = 
                            [
                              {'label': "Count",   'value':"Count"},
                              {'label': "Positive ratings",  'value':"positive_ratings"},
                              {'label': "Negative ratings",  'value':"negative_ratings"},
                              {'label': "Average playtime",  'value':"average_playtime"},
                              {'label': "Price",             'value':"price"},
                              {'label': "Max income",'value':"max_income"}], value = 'Count')], className = 'four columns'),
                    html.Div([html.Button(id = 'plot', n_clicks = 0, children = 'Show')], className = 'one column')
                ], className = 'row')     
            ]),
            dcc.Tab(label='Similar Games', value='Games2', children=
            [
                html.Div(
                [
                    dcc.Dropdown(options=game_names_dict(), placeholder="Select a city", id='game_name', value=10)
                ]),
                html.Div(id='game_content')
            ])
        ])
        
        #html.Div([html.H1('Dash App')], className='row'),

    ], className='container')


# @app.callback(
#     Output('game_name', 'value'),
#     Input('top_games', 'clickData'))
# def display_click_data(clickData):
#     print(clickData['points']['label'])
#     name = clickData['points']['label']
#     new_appid = steam[steam.name == name].appid.values[0]
#     return new_appid

if __name__ == '__main__':
    app.run_server(debug=True)
