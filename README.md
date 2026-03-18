# prediction-prix-dvf

Dashboard ML de prediction des prix immobiliers en Ile-de-France (2024).

## Lien de l'application

L'application est déployée sur huggingface et consultable ici : 👉 [https://huggingface.co/spaces/KPHabib/dvf-idf]

## Stack
- Python : pandas, scikit-learn, xgboost, dash, plotly
- Donnees : DVF 2024 — data.gouv.fr (114 882 transactions)
- Modele : Random Forest (MAE=1 181 EUR/m2, R2=0.669)

## Fonctionnalites
- Vue d ensemble du marche IDF 2024
- Simulateur ML de prix au m2 interactif
- Carte geographique des prix par transaction

## Auteur
Habib Laskin KPENGOU — Data Analyst DRIEAT Ile-de-France
github.com/habib-07
