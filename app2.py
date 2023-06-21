import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# Load the CSV data
df_final_20_22 = pd.read_csv("C:/Users/marie/Documents/Projet ENEDIS/df_final_20_22.csv")

# Define X and y variables
y = df_final_20_22["Consommation_moyenne"]
X = df_final_20_22.select_dtypes(include='number').drop(
    ["Consommation_moyenne", "Nb points soutirage", "Total énergie soutirée (Wh)"], axis=1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, shuffle=False)

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(max_depth=None, max_features=1.0,
                              min_samples_leaf=1, min_samples_split=2, n_estimators=100)
model.fit(X_train_scaled, y_train)

# Create the Dash application
app = dash.Dash(__name__)

# Define the application layout
app.layout = html.Div(
    style={
        'background-color': '#FFFFFF',
        'color': 'black',
        'textAlign': 'center',
        'justify-content': 'center'
    },
    children=[
        html.Br(),
        html.Img(
            src="/assets/logo_enedis.png",
            style={'height': '200px'}
        ),
        html.Br(),
        html.Br(),
        html.Br(),
        html.H1("Projet ENEDIS : prédiction de la consommation",
                style={'margin': 'auto', 'text-align': 'center'}),
        html.Br(),
        html.Br(),
        html.Br(),
        
        
        html.Div(
                [
                    html.Div(
                        [
                html.P("Choisissez une température (°C): "),
                                dcc.Slider(
                                    id='temperature-slider',
                                    min=-15,
                                    max=40,
                                    step=1,
                                    value=0,
                                    marks={i: str(i) for i in range(-16, 41, 2)},
                                ),
                                html.Div(id='temperature-output'),
                            ],
                            className="col",
                            style={'margin': 'auto', 'width': '80%'}  
                        ),


            html.Div(
                [
                    html.P("Quantité de pluie (mm)"),
                    dcc.Input(id='rainfall-input', type='number', value='', min=0),
                ],
                className="col",
                style={'margin': 'auto'}
            ),
        ],
        className="row",
        style={'width': '300px', 'textAlign': 'center',
                           'color': 'black', 'margin': '0 auto'}


        ),
        html.Br(),
        html.Br(),
        
        
        html.Div([
            html.P("Statut du jour de la semaine (obligatoire)"),
            dcc.Dropdown(
                id='style-jour-dropdown',
                options=[
                    {'label': 'Ouvré', 'value': 'Ouvré'},
                    {'label': 'Week-end', 'value': 'Week-end'},
                    {'label': 'Férié', 'value': 'Férié'},
                ],
                value='',
                style={'width': '250px', 'textAlign': 'center',
                       'color': 'black', 'margin': '0 auto'}
            )
        ], style={'display': 'inline-block', 'margin-right': '10px'}),
        html.Div([
            html.P("Description"),
            dcc.Dropdown(
                id='description-dropdown',
                options=[
                    {'label': "Sans objet", 'value': 'Aucune'},
                    {'label': 'Vacances', 'value': 'Vacances'},
                    {'label': 'Confinement', 'value': 'Confinement'},

                ],
                value='',
                style={'width': '250px', 'textAlign': 'center',
                       'color': 'black', 'margin': '0 auto'}
            )
        ], style={'display': 'inline-block', 'margin-right': '10px'}),
        html.Br(),
        html.P("Date (mois)"),
        dcc.Dropdown(
            id='date-input',
            options=[
                {'label': 'Janvier', 'value': '1'},
                {'label': 'Février', 'value': '2'},
                {'label': 'Mars', 'value': '3'},
                {'label': 'Avril', 'value': '4'},
                {'label': 'Mai', 'value': '5'},
                {'label': 'Juin', 'value': '6'},
                {'label': 'Juillet', 'value': '7'},
                {'label': 'Aout', 'value': '8'},
                {'label': 'Septembre', 'value': '9'},
                {'label': 'Octobre', 'value': '10'},
                {'label': 'Novembre', 'value': '11'},
                {'label': 'Décembre', 'value': '12'}
            ],
            value='',
            style={'width': '250px', 'textAlign': 'center',
                   'color': 'black', 'margin': '0 auto'}
        ),
        html.Br(),
        
        html.Br(),
        html.Div([
        html.P("Profil consommateur"),
        dcc.Dropdown(
            id='profil-consommateur-dropdown',
            options=[
                {'label': 'Professionnel', 'value': 'Professionnel'},
                {'label': 'Résidentiel', 'value': 'Résidentiel'}
            ],
            value='',
            style={'width': '250px', 'textAlign': 'center',
                   'color': 'black', 'margin': '0 auto'}
        ),
    ], style={'display': 'inline-block', 'margin-right': '10px'}),
    
        html.Div(
            [
                html.P("Région"),
                dcc.Dropdown(
                    id='region-dropdown',
                    options=[
                        {'label': 'Centre Val de Loire', 'value': 'Centre-Val de Loire'},
                        {'label': 'Hauts de France', 'value': 'Hauts-de-France'}
                    ],
                    value='',
                    style={'width': '250px', 'textAlign': 'center',
                           'color': 'black', 'margin': '0 auto'}
                ),
            ],
            style={'display': 'inline-block', 'margin-right': '10px'}
        ),


        html.Br(),
        html.Br(),
        html.Button('Prédire', id='predict-button', n_clicks=0),
        html.Br(),
        html.H3("Consommation électrique prédite :"),
        html.Div(id='prediction-output')
    ])


@ app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    State("temperature-slider", "value"),
    State("rainfall-input", "value"),
    State("style-jour-dropdown", "value"),
    State("description-dropdown", "value"),
    State("date-input", "value"),
    State("profil-consommateur-dropdown", "value"),
    State("region-dropdown", "value")
)
def update_prediction_output(n_clicks, temperature, rainfall, style_jour, description, date, profil_consommateur, region):
    if n_clicks > 0:

        input_data = pd.DataFrame(columns=X_train.columns)

        input_data.loc[0, 'Moyenne_temperature'] = temperature
        input_data.loc[0, 'PRECIP_TOTAL_DAY_MM'] = rainfall
        input_data['Statut_férié'] = 0
        input_data['Statut_ouvré'] = 0
        input_data['Statut_week-end'] = 0

        if style_jour == 'Férié':
            input_data['Statut_férié'] = 1
        elif style_jour == 'Ouvré':
            input_data['Statut_ouvré'] = 1
        elif style_jour == 'Week-end':
            input_data['Statut_week-end'] = 1

        input_data['Profil_consommateur_Professionnel'] = 0
        input_data['Profil_consommateur_Résident'] = 0

        if profil_consommateur == 'Professionnel':
            input_data['Profil_consommateur_Professionnel'] = 1
        elif profil_consommateur == 'Résidentiel':
            input_data['Profil_consommateur_Résident'] = 1

        input_data['Région_Centre-Val de Loire'] = 0
        input_data['Région_Hauts-de-France'] = 0

        if region == 'Centre-Val de Loire':
            input_data['Région_Centre-Val de Loire'] = 1
        elif region == 'Hauts-de-France':
            input_data['Région_Hauts-de-France'] = 1

        input_data['Mois'] = date

        input_data['Description_y_Confinement'] = 0
        input_data['Description_y_Vacances'] = 0

        if description == 'Vacances':
            input_data['Description_y_Vacances'] = 1
        elif description == 'Confinement':
            input_data['Description_y_Confinement'] = 1

        month_averages = X_train.groupby("Mois").mean()
        input_data.fillna(month_averages.loc[int(date)], inplace=True)

        input_data_scaled = scaler.transform(input_data)

        predicted_consumption = model.predict(input_data_scaled)

        return f"Prédiction de la consommation électrique : {predicted_consumption[0]:.2f} Wh"

    return ""


if __name__ == '__main__':
    app.run_server(debug=True)