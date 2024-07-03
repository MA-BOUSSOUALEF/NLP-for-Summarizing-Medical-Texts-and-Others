import pandas as pd
import spacy
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from matplotlib import pyplot as plt


# Charger le modèle de langue française de spaCy
nlp = spacy.load('fr_core_news_sm', disable=['parser', 'ner'])

# Charger les données depuis le fichier Excel
df = pd.read_excel('IA/Data_.xls')
df['COMMENTAIRE'] = df['COMMENTAIRE'].astype(str)
df['CRAP'] = df['CRAP'].astype(str)
df['RCP_TXT'] = df['RCP_TXT'].astype(str)

# Supprimer les NaN values
df.dropna(subset=['CRAP', 'RCP_TXT', 'COMMENTAIRE'], inplace=True)

# Définir une fonction de prétraitement du texte
def preprocess_text(text):
    # Convertir le texte en minuscules
    text = text.lower()
    # Supprimer les caractères non alphabétiques et les chiffres
    text = re.sub(r'[^a-z]', ' ', text)
    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text)
    return text

# Appliquer le prétraitement sur les colonnes CRAP et RCP_TXT
df['CRAP'] = df['CRAP'].apply(preprocess_text)
df['RCP_TXT'] = df['RCP_TXT'].apply(preprocess_text)

# Grouper les commentaires selon le numéro de tumeur en évitant les doublons
df_grouped_comment = df.groupby('NUM_IDTUMEUR')['COMMENTAIRE'].apply(lambda x: ' '.join(set(x))).reset_index()

# Grouper les textes CRAP et RCP_TXT selon le numéro de tumeur
df_grouped_merged_text = df.groupby('NUM_IDTUMEUR').agg({'CRAP': ' '.join, 'RCP_TXT': ' '.join}).reset_index()

# Fusionner les DataFrames groupés
df_ = pd.merge(df_grouped_comment, df_grouped_merged_text, on='NUM_IDTUMEUR')
df['MERGED_CRAP_RCP_TXT'] = df['NUM_IDTUMEUR']+'\n' + df['RCP_TXT'] + '\n' + df['CRAP'] 


#_________
# Fusionner les colonnes CRAP et RCP_TXT en une seule colonne
df_['MERGED_CRAP_RCP_TXT'] = df_['CRAP'] + ' ' + df_['RCP_TXT']

# Supprimer les colonnes CRAP et RCP_TXT
df_.drop(['CRAP', 'RCP_TXT'], axis=1, inplace=True)

# # Renommer les colonnes pour plus de clarté
# df_.rename(columns={'COMMENTAIRE': 'COMMENTAIRES_ASSOCIES'}, inplace=True)


# Séparation des données en données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(df['MERGED_CRAP_RCP_TXT'], df['COMMENTAIRE'], test_size=0.1, random_state=42)

# Vectorisation TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Entraîner un modèle de forêt aléatoire
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_tfidf, y_train)

# Prédire les résumés pour les données de test
y_pred = rf_classifier.predict(X_test_tfidf)



#Afficher les scores et le rapport de classification

# for i in range(len(X_test)):
#     # print("Texte de test :",df_['NUM_IDTUMEUR'][i], X_test.iloc[i])
#     print("NUM_IDTUMEUR de texte de test :", X_test.iloc[i].split('\n')[0])
#     print("\n")
#     print("Résumé généré :", y_pred[i])
#   #print("Commentaire associé :", df[df['MERGED_CRAP_RCP_TXT'] == X_test.iloc[i]]['COMMENTAIRES_ASSOCIES'].values[0])
#     print("\n")

print("Score du modèle sur les données de test :", accuracy_score(y_test, y_pred))
# print("Rapport de classification :\n", classification_report(y_test, y_pred))
def generate_summary(text):
    # Prétraiter le texte de test
    preprocessed_text = preprocess_text(text)
    # Vectoriser le texte de test en utilisant le même vectoriseur utilisé pour l'entraînement
    text_tfidf = tfidf_vectorizer.transform([preprocessed_text])
    # Prédire le résumé
    summary = rf_classifier.predict(text_tfidf)
    cleaned_summary = re.sub(r'\b\d{2}/\d{2}/\d{2}\b', '', summary[0])
    return cleaned_summary

# #Texte de test
text_test = """


RENSEIGNEMENTS CLINIQUES :
Côté : DROIT (QI)
Motif du prélèvement : A1903645 : MC + HEA
Type du prélèvement :  zonectomie (A) + recoupe inférieure (B)
EXAMEN MACROSCOPIQUE :
Fixation : FORMOL
Taille du prélèvement : (A) 6 x 4 x 2.5 cm (inclusion en totalité)(B) 3 x 2.5 x 0.7 cm
Taille du lambeau cutané associé : sur (A) : 2.5 x 1.5 cm
Orientation du prélèvement :  oui
Encrage du prélèvement :   oui
Lésion macroscopique :   1
Taille macroscopique de la lésion principale : centimétrique
Examen Extemporané : Non
MICROSCOPIE :
Lésions bénignes : 0
Atypies épithéliales : 0
Carcinome lobulaire in situ (CLIS) sans lésion infiltrante associée :  0
Carcinome canalaire in situ (CCIS) sans lésion infiltrante associée :  un foyer de 5 mm de carcinome intracanalaire de bas grade, de type cribriforme sans nécrose
Micro-invasion :   0 (P63+)
Microcalcifications :   oui
Berges d'exérèse : ppm : 1 mm en limite externe
Prélèvements en périphérie de la pièce opératoire : PIECE (B) : RECOUPE INFERIEURE INTERNE ET EXTERNE absence d'image spécifique
EXTENSION :
Cutané : absence d'image spécifique
CONCLUSION :
UN FOYER DE 5 MM DE CIC BAS GRADE TYPE CRIBRIFORME SANS NECROSE
Limite d'exérèse : ppm : 1 mm en externe avant recoupe (B) donc au large pour la marge définitive
B. POLYPE INTRACVITAIRE UTERIN
L'aspect histologique est celui d'un POLYPE ENDOMETRIAL PAR HYPERPLASIE GLANDULO-KYSTIQUE.
Les glandes ectasiées ou dilatées en micro-kystes sont tapissées par un épithélium cubique. Elles sont entourées par un chorion cytogène densifié, remanié par des plages de fibrose et irrigué par des vaisseaux à la paroi hyalinisée.
La lésion ainsi définie réalise une végétation tapissée par un épithélium superficiel aminci ou parfois érodé.
Pas d'atypies.
AUCUN SIGNE DE MALIGNITE.
Signature validée électroniquement
Bastia, le 24/06/2019
Docteur 
@ZSignatureLec@
{Mode de découverte : Dépistage organisé.} 
{Circonstances de la découverte : foyer de micro qudarnt inferieur du sein droit : HCA
tumorectomie apres harpon.} 
{Latéralité : Droite.} 
{Commentaire : 1- Radiothérapie.} 
{ / Résultat anapath : CIS de 5 mm in sano.} 
{Plan de traitement : Radiothérapie.} 
{Synthèse : RT.} 
{Nature de la proposition : Mise en traitement.} 
.} 
 foyer de micro qudarnt inferieur du sein droit : HCA
tumorectomie apres harpon1- RadiothérapieCIS de 5 mm in sano01/07/201901/07/2019Oui2B0000012/CENTRE HOSPITALIER DE BASTIADroiteDépistage organiséProposition thérapeutiqueMise en traitementPhase initialeRadiothérapieCRAPNonNonDiscutéRTApplication référentiel

"""
#Générer le résumé
summary_generated = generate_summary(text_test)
# Afficher le résumé généré
print("Résumé généré pour le texte de test :\n", summary_generated)

