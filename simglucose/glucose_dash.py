import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import pkg_resources

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')

def getPaitntNames():
    params = pd.read_csv(PATIENT_PARA_FILE)
    return params['Name'].tolist()

def getPatientOptions():
    names = getPaitntNames()
    return [{'label': n, 'value': n} for n in names]

app.layout = html.Div([
    html.H1('Simglucose Dashboard'),

    html.Div(children="A web-based Type 1 Diabetes simulator"),

    html.Div(
        [
            dcc.Dropdown(id='patient-selector',
                         options=getPatientOptions(),
                         multi=True,
                         placeholder="Select patients for simulation ..."),

            html.Button('Run Simulation', id='run-sim')
        ]
    ),

    html.Div(
        [
            html.H3('Glucose Level'),
            dcc.Graph(id="glucose-level")
        ]
    )
])


if __name__ == '__main__':
    app.run_server(debug=True)
