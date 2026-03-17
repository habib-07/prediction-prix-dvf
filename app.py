
import pandas as pd
import numpy as np
import joblib
import os
import urllib.request
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ============================================================
# TELECHARGEMENT ET ENTRAINEMENT AU DEMARRAGE
# ============================================================
dept_idf = ["75","77","78","91","92","93","94","95"]
dept_labels = {
    "75":"Paris","77":"Seine-et-Marne","78":"Yvelines",
    "91":"Essonne","92":"Hauts-de-Seine",
    "93":"Seine-Saint-Denis","94":"Val-de-Marne","95":"Val-d Oise"
}

def telecharger_et_preparer():
    os.makedirs("data", exist_ok=True)
    dfs = []
    for dept in dept_idf:
        path = f"data/{dept}_2024.csv.gz"
        if not os.path.exists(path):
            url = f"https://files.data.gouv.fr/geo-dvf/latest/csv/2024/departements/{dept}.csv.gz"
            print(f"Telechargement {dept}...")
            urllib.request.urlretrieve(url, path)
        df_d = pd.read_csv(path, compression="gzip", low_memory=False)
        dfs.append(df_d)
    return pd.concat(dfs, ignore_index=True)

def nettoyer(df_raw):
    df = df_raw[df_raw["type_local"].isin(["Appartement","Maison"])].copy()
    cols = ["id_mutation","valeur_fonciere","code_departement","nom_commune",
            "type_local","surface_reelle_bati","nombre_pieces_principales",
            "longitude","latitude"]
    df = df[cols].copy()
    df["valeur_fonciere"]           = pd.to_numeric(df["valeur_fonciere"], errors="coerce")
    df["surface_reelle_bati"]       = pd.to_numeric(df["surface_reelle_bati"], errors="coerce")
    df["nombre_pieces_principales"] = pd.to_numeric(df["nombre_pieces_principales"], errors="coerce")
    df = df.groupby("id_mutation").first().reset_index()
    df = df.dropna(subset=["valeur_fonciere","surface_reelle_bati","longitude","latitude"])
    df = df[(df["valeur_fonciere"]>10000)&(df["valeur_fonciere"]<5000000)]
    df = df[(df["surface_reelle_bati"]>9)&(df["surface_reelle_bati"]<500)]
    df["prix_m2"] = (df["valeur_fonciere"]/df["surface_reelle_bati"]).round(0)
    df = df[(df["prix_m2"]>500)&(df["prix_m2"]<25000)]
    df["code_departement"] = df["code_departement"].astype(str).str.zfill(2)
    df["dept_lib"] = df["code_departement"].map(dept_labels)
    df["dist_paris_km"] = (
        ((df["latitude"]-48.8530)*111)**2 +
        ((df["longitude"]-2.3499)*74)**2
    )**0.5
    return df

def entrainer(df):
    le_type = LabelEncoder()
    le_dept = LabelEncoder()
    df["type_enc"] = le_type.fit_transform(df["type_local"])
    df["dept_enc"] = le_dept.fit_transform(df["code_departement"])
    features = ["surface_reelle_bati","nombre_pieces_principales",
                "dist_paris_km","latitude","longitude",
                "mois","trimestre","type_enc","dept_enc"]
    df["mois"] = 6
    df["trimestre"] = 2
    X = df[features].dropna()
    y = df.loc[X.index, "prix_m2"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
    print("Entrainement du modele...")
    model.fit(X_train, y_train)
    print("Modele entraine !")
    return model, le_type, le_dept

print("Chargement des donnees...")
df_raw = telecharger_et_preparer()
df = nettoyer(df_raw)
print(f"Donnees pretes : {len(df):,} transactions")
model, le_type, le_dept = entrainer(df)

# ============================================================
# APP DASH
# ============================================================
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

simulateur = dbc.Card([
    html.H5("Simulateur de prix", className="text-primary"),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.Label("Type de bien"),
            dcc.Dropdown(id="sim_type",
                options=[{"label":"Appartement","value":"Appartement"},
                         {"label":"Maison","value":"Maison"}],
                value="Appartement", clearable=False)
        ], width=6),
        dbc.Col([
            html.Label("Departement"),
            dcc.Dropdown(id="sim_dept",
                options=[{"label":f"{dept_labels[d]} ({d})","value":d} for d in dept_idf],
                value="75", clearable=False)
        ], width=6),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            html.Label("Surface (m2)"),
            dcc.Slider(id="sim_surface", min=15, max=200, value=65, step=5,
                       marks={15:"15",50:"50",100:"100",150:"150",200:"200"})
        ], width=12),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            html.Label("Nombre de pieces"),
            dcc.Slider(id="sim_pieces", min=1, max=8, value=3, step=1,
                       marks={i:str(i) for i in range(1,9)})
        ], width=12),
    ], className="mb-3"),
    html.Div(id="sim_resultat", className="mt-3")
], body=True)

tabs = dbc.Tabs([
    dbc.Tab(label="Marche IDF", children=[
        dbc.Row([
            dbc.Col(dbc.Card(id="kpi_ventes",   color="primary", inverse=True), width=3),
            dbc.Col(dbc.Card(id="kpi_prix_app", color="danger",  inverse=True), width=3),
            dbc.Col(dbc.Card(id="kpi_prix_mai", color="warning", inverse=True), width=3),
            dbc.Col(dbc.Card(id="kpi_r2",       color="dark",    inverse=True), width=3),
        ], className="mt-3 mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="p_prix_dept"), width=7),
            dbc.Col(dcc.Graph(id="p_dist"),      width=5),
        ])
    ]),
    dbc.Tab(label="Simulateur ML", children=[
        dbc.Row([
            dbc.Col(simulateur, width=5),
            dbc.Col(dcc.Graph(id="p_sim_gauge"), width=7),
        ], className="mt-3"),
        dbc.Row([dbc.Col(dcc.Graph(id="p_sim_hist"), width=12)])
    ]),
    dbc.Tab(label="Carte des prix", children=[
        dbc.Row([
            dbc.Col([
                html.Label("Type de bien"),
                dcc.Dropdown(id="carte_type",
                    options=[{"label":"Appartement","value":"Appartement"},
                             {"label":"Maison","value":"Maison"}],
                    value="Appartement", clearable=False)
            ], width=3),
        ], className="mt-3 mb-2"),
        dbc.Row([dbc.Col(dcc.Graph(id="carte_prix", style={"height":"550px"}), width=12)])
    ]),
])

app.layout = dbc.Container([
    dbc.Row([html.H2("DVF — Marche Immobilier en Ile-de-France 2024",
                     className="text-primary my-3")]),
    dbc.Row([dbc.Col(tabs, width=12)])
], fluid=True)

@app.callback(
    Output("kpi_ventes","children"), Output("kpi_prix_app","children"),
    Output("kpi_prix_mai","children"), Output("kpi_r2","children"),
    Input("carte_type","value")
)
def kpis(_):
    nb    = len(df)
    p_app = df[df["type_local"]=="Appartement"]["prix_m2"].median()
    p_mai = df[df["type_local"]=="Maison"]["prix_m2"].median()
    def k(t,v): return dbc.CardBody([html.H6(t), html.H3(str(v))])
    return (k("Transactions 2024", f"{nb:,}"),
            k("Mediane Appart.",   f"{p_app:,.0f} EUR/m2"),
            k("Mediane Maison",    f"{p_mai:,.0f} EUR/m2"),
            k("R2 modele ML",      "0.669"))

@app.callback(Output("p_prix_dept","figure"), Input("carte_type","value"))
def p_prix_dept(_):
    d = df.groupby(["dept_lib","type_local"])["prix_m2"].median().reset_index()
    fig = px.bar(d, x="dept_lib", y="prix_m2", color="type_local", barmode="group",
                 color_discrete_map={"Appartement":"#A32D2D","Maison":"#185FA5"},
                 title="Prix/m2 median par departement",
                 labels={"prix_m2":"EUR/m2","dept_lib":"","type_local":"Type"})
    fig.update_layout(template="plotly_white", xaxis_tickangle=-30)
    return fig

@app.callback(Output("p_dist","figure"), Input("carte_type","value"))
def p_dist(_):
    sample = df.sample(min(5000,len(df)), random_state=42)
    fig = px.scatter(sample, x="dist_paris_km", y="prix_m2", color="type_local",
                     opacity=0.3,
                     color_discrete_map={"Appartement":"#A32D2D","Maison":"#185FA5"},
                     title="Prix vs Distance Paris",
                     labels={"dist_paris_km":"Distance (km)","prix_m2":"EUR/m2","type_local":"Type"})
    fig.update_traces(marker_size=4)
    fig.update_layout(template="plotly_white")
    return fig

@app.callback(
    Output("sim_resultat","children"),
    Output("p_sim_gauge","figure"),
    Output("p_sim_hist","figure"),
    Input("sim_type","value"), Input("sim_dept","value"),
    Input("sim_surface","value"), Input("sim_pieces","value")
)
def simuler(type_b, dept, surface, pieces):
    coords = {
        "75":(48.8566,2.3522),"77":(48.8500,2.9167),"78":(48.8000,1.9833),
        "91":(48.6333,2.4500),"92":(48.8900,2.2500),"93":(48.9167,2.4333),
        "94":(48.7833,2.4667),"95":(49.0500,2.0833)
    }
    lat, lon = coords.get(dept,(48.8566,2.3522))
    dist = (((lat-48.8530)*111)**2+((lon-2.3499)*74)**2)**0.5
    type_enc = int(le_type.transform([type_b])[0])
    dept_enc = int(le_dept.transform([dept])[0])
    X_pred = pd.DataFrame([{
        "surface_reelle_bati":surface,"nombre_pieces_principales":pieces,
        "dist_paris_km":dist,"latitude":lat,"longitude":lon,
        "mois":6,"trimestre":2,"type_enc":type_enc,"dept_enc":dept_enc
    }])
    prix_pred     = model.predict(X_pred)[0]
    valeur_totale = prix_pred * surface
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prix_pred,
        title={"text":f"Prix/m2 predit — {dept_labels.get(dept,dept)}"},
        number={"suffix":" EUR/m2"},
        gauge={
            "axis":{"range":[500,20000]},
            "bar":{"color":"#A32D2D"},
            "steps":[
                {"range":[500,3000],"color":"#C8E6C9"},
                {"range":[3000,6000],"color":"#FFF9C4"},
                {"range":[6000,10000],"color":"#FFE0B2"},
                {"range":[10000,20000],"color":"#FFCDD2"}
            ]
        }
    ))
    fig_gauge.update_layout(template="plotly_white")
    d_ref = df[(df["type_local"]==type_b)&(df["code_departement"]==dept)]
    fig_hist = px.histogram(d_ref, x="prix_m2", nbins=40,
                            title=f"Distribution — {type_b} — {dept_labels.get(dept,dept)}",
                            labels={"prix_m2":"Prix/m2 (EUR)"},
                            color_discrete_sequence=["#185FA5"])
    fig_hist.add_vline(x=prix_pred, line_dash="dash", line_color="#A32D2D",
                       annotation_text=f"Estimation: {prix_pred:,.0f} EUR/m2")
    fig_hist.update_layout(template="plotly_white")
    resultat = dbc.Alert([
        html.H4(f"Prix estime : {prix_pred:,.0f} EUR/m2", className="alert-heading"),
        html.P(f"Valeur totale estimee : {valeur_totale:,.0f} EUR"),
        html.P(f"{surface} m2 | {pieces} pieces | {dept_labels.get(dept,dept)}")
    ], color="success")
    return resultat, fig_gauge, fig_hist

@app.callback(Output("carte_prix","figure"), Input("carte_type","value"))
def carte_prix(type_b):
    d = df[df["type_local"]==type_b].dropna(subset=["longitude","latitude"])
    d = d.sample(min(8000,len(d)), random_state=42)
    fig = px.scatter_mapbox(d, lat="latitude", lon="longitude",
                            color="prix_m2", size_max=8,
                            color_continuous_scale="RdYlGn_r",
                            range_color=[1000,15000],
                            hover_name="nom_commune",
                            zoom=9, center={"lat":48.85,"lon":2.35},
                            title=f"Carte des prix/m2 — {type_b}",
                            mapbox_style="carto-positron")
    fig.update_layout(template="plotly_white", margin={"r":0,"t":40,"l":0,"b":0})
    return fig

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT",8056)))
