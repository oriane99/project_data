import streamlit as st 
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as xp
import warnings
import plotly.figure_factory as ff


# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)


# Titre de l'application
#st.title('Data analyst')
#df = pd.read_csv("DataAnalyst.csv")
#df.drop(['Unnamed: 0'], axis=1, inplace=True)

#data=pd.DataFrame(df)

with st.sidebar:

    st.logo("Logo_Efrei_PAris_2.png")
    selected = option_menu(
        menu_title="Menu",
        options= ["Initial data","Data preprocessing","Visualization","Clustering","Learning Evaluation","Objective"]
    
    )  
    uploaded_file=st.file_uploader("Cliquez ici pour choisir un fichier CSV", type="csv")   

    

  
# Bouton de téléchargement de fichier CSV



# Initialiser la variable de session pour le DataFrame
if 'df' not in st.session_state:
    st.session_state.df = None


print("-------------------------part 1")

if selected == "Initial data":
        # Titre de l'application
    st.title("Application d'analyse de données")



    if uploaded_file is not None:
        # Lecture du fichier CSV
        #df = pd.read_csv(uploaded_file, sep=" ")
        #df = pd.read_csv(uploaded_file, sep=',')
        df = pd.read_csv(uploaded_file,delim_whitespace=True)

        # Ajouter les noms des colonnes manuellement
        noms_colonnes = ['sequence_name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
        df.columns = noms_colonnes

        st.session_state.df = df

        # Affichage des premières et dernières lignes du fichier
        st.subheader("Aperçu des données")
        st.write("Premières lignes:")
        st.write(df.head())
        st.write("Dernières lignes:")
        st.write(df.tail())
        # Résumé statistique
        st.subheader("Résumé statistique")
        st.write("Nombre de lignes et de colonnes:")
        st.write(df.shape)
        st.write("Noms des colonnes:")
        st.write(df.columns)
        st.write("Nombre de valeurs manquantes par colonne:")
        st.write(df.isnull().sum())
        st.write("Description des variables")
        st.write(df.describe())
    
print("----------------------------------")
if selected == "Data preprocessing":
    if st.session_state.df is not None:
        df = st.session_state.df #session enregistrée
        st.subheader("Prétraitement des données")
        st.write("Données actuelles:")
        st.write(df)

    st.subheader("Traitement des valeurs manquantes")
                    # Choix de la méthode de traitement       
    method = st.selectbox("Choisissez une méthode pour traiter les valeurs manquantes", 
                              ["Choisir","Supprimer les lignes/colonnes", "Remplacer par la moyenne", 
                               "Remplacer par la médiane", "Remplacer par le mode", 
                               "Imputation KNN"])

        # Option pour supprimer les valeurs manquantes
    if method=="Supprimer les lignes/colonnes":
            st.session_state.df = df
            # Afficher les valeurs manquantes avant suppression
            missing_values = df[df.isnull().any(axis=1)]
            st.write("Valeurs manquantes avant suppression:")
            st.write(missing_values)

            # Supprimer les lignes avec des valeurs manquantes
            df_cleaned=df.dropna(inplace=True)
            
            # Afficher le DataFrame nettoyé
            st.write("DataFrame après suppression des valeurs manquantes:")
            st.write(df)
    
    elif method == "Remplacer par la moyenne":
        # Afficher les valeurs manquantes avant suppression
        missing_values = df[df.isnull().any(axis=1)]
        st.write("Valeurs manquantes avant suppression:")
        st.write(missing_values)

        # Remplacer par la moyenne
        df.fillna(df.mean(), inplace=True)
        st.session_state.df = df
        st.write("Valeurs manquantes remplacées par la moyenne.")
        st.write(df)
    
    elif method == "Remplacer par la médiane":
        # Afficher les valeurs manquantes avant suppression
        missing_values = df[df.isnull().any(axis=1)]
        st.write("Valeurs manquantes avant suppression:")
        st.write(missing_values)

        # Remplacer par la mediane
        df_remplaced_mode= df.fillna(df.median(), inplace=True)
        st.session_state.df = df
        st.write("Valeurs après remplacement par la médiane.")
        st.write(df)

    elif method == "Remplacer par le mode":
        st.session_state.df = df
        # Afficher les valeurs manquantes avant suppression
        missing_values = df[df.isnull().any(axis=1)]
        st.write("Valeurs manquantes avant suppression:")
        st.write(missing_values)

        for column in df.columns:
            df[column].fillna(df[column].mode()[0], inplace=True)

        # Mettre à jour la session state avec le DataFrame modifié
        st.session_state.df = df

        # Afficher un message de confirmation et le DataFrame modifié
        st.write("Valeurs manquantes remplacées par le mode.")
        st.write(df)
    
    elif method == "Imputation KNN":
        st.session_state.df = df
        # Afficher les valeurs manquantes avant suppression
        missing_values = df[df.isnull().any(axis=1)]
        st.write("Valeurs manquantes avant suppression:")
        st.write(missing_values)

        # Remplacer par l imputation
        imputer = KNNImputer()
        df[df.columns] = imputer.fit_transform(df)
        st.session_state.df = df
        st.write("Valeurs manquantes imputées avec KNN.")
        st.write(df)
    
    
print("--------------------------------")
if selected == "Visualization":
    if st.session_state.df is not None:
        df = st.session_state.df
        st.subheader("Visualisation des données")
        st.write(df)

        # Checklist pour choisir les types de visualisations
        options = st.multiselect("Choisissez les types de visualisation", 
                                 ["Univariée", "Bivariée", "Multivariée"])

        # Visualisation univariée des données
        if "Univariée" in options:
            st.write("Visualisation univariée des données:")
            selected_column = st.selectbox("Sélectionnez une colonne pour visualiser l'histogramme", df.columns)
            if selected_column:
    # Créer l'histogramme interactif avec plotly
                fig = xp.histogram(df, x=selected_column, nbins=10, title=f"Histogramme de {selected_column}")
                fig.update_layout(showlegend=False)
                
                # Afficher le graphique dans Streamlit
                st.plotly_chart(fig)
            else:
                st.warning("Veuillez sélectionner une colonne pour visualiser l'histogramme.")

        # Visualisation bivariée des données
        elif "Bivariée" in options:
            st.write("Visualisation bivariée des données avec histogrammes conditionnés:")
            selected_columns = st.multiselect("Sélectionnez deux colonnes pour visualiser leur relation", df.columns)

# Sélecteur de type de graphique
            plot_type = st.selectbox("Sélectionnez le type de graphique", ["Histogramme", "Boxplot"])

            if len(selected_columns) == 2:
                if plot_type == "Histogramme":
                    # Créer l'histogramme bivarié interactif avec plotly
                    fig = xp.histogram(df, x=selected_columns[0], title=f"Histogramme de {selected_columns[0]} par {selected_columns[1]}")
                    fig.update_layout(bargap=0.1, showlegend=False)  # Enlever la légende
                elif plot_type == "Boxplot":
                    # Créer le boxplot bivarié interactif avec plotly
                    fig = xp.box(df, x=selected_columns[0], y=selected_columns[1], title=f"Boxplot de {selected_columns[0]} par {selected_columns[1]}")
                    fig.update_layout(showlegend=False)  # Enlever la légende

                # Afficher le graphique dans Streamlit
                st.plotly_chart(fig)
            else:
                st.warning("Veuillez sélectionner deux colonnes pour visualiser leur relation.")
            
        elif "Multivariée" in options:
            # Calculer la matrice de corrélation
            # Sélectionner uniquement les colonnes avec des types de données numériques
            df_numeric = df.select_dtypes(include=['number'])

            # Vérifier le résultat
            corr_matrix = df_numeric.corr()

            # Afficher la matrice de corrélation
            print(corr_matrix)
                            # Créer un heatmap interactif avec plotly
            fig = xp.imshow(corr_matrix.values,
                x=list(corr_matrix.columns),
                    y=list(corr_matrix.index),
                    color_continuous_scale='Viridis')

            # Mise à jour du layout
            fig.update_layout(title="Matrice de Corrélation avec Légende des Couleurs",
                            xaxis_title="Variables",
                            yaxis_title="Variables",
                            coloraxis_colorbar=dict(title='Corrélation', tickvals=[-1, 0, 1], ticktext=['Faible', 'Moyenne', 'Élevée'])) #afficher la legende

            # Afficher le graphique dans Streamlit
            st.plotly_chart(fig)
        
if selected=="Objective":
    st.write("Ce projet a pour but de :")