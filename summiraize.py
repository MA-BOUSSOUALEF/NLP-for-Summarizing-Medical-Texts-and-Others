import spacy
from spacy.lang.fr.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

# Charger le modèle de langue française
nlp = spacy.load('fr_core_news_sm')

# Texte à résumer
text = """L'informatique est un domaine d'activité scientifique, technique, et industriel concernant le traitement automatique de l'information numérique par l'exécution de programmes informatiques hébergés par des dispositifs électriques-électroniques : des systèmes embarqués, des ordinateurs, des robots, des automates, etc.

Ces champs d'application peuvent être séparés en deux branches :

théorique : concerne la définition de concepts et modèles ;
pratique : s'intéresse aux techniques concrètes de mise en œuvre.
Certains domaines de l'informatique peuvent être très abstraits, comme la complexité algorithmique, et d'autres peuvent être plus proches d'un public profane. Ainsi, la théorie des langages demeure un domaine davantage accessible aux professionnels formés (description des ordinateurs et méthodes de programmation), tandis que les métiers liés aux interfaces homme-machine (IHM) sont accessibles à un plus large public."""

# Créer une liste de mots vides et de ponctuation
stopwords = list(STOP_WORDS)
punctuation = punctuation + '\n'

# Calculer les fréquences des mots
word_frequencies = {}
for word in nlp(text):
    if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
        if word.text not in word_frequencies.keys():
            word_frequencies[word.text] = 1
        else:
            word_frequencies[word.text] += 1

# Normaliser les fréquences des mots
max_frequency = max(word_frequencies.values())
for word in word_frequencies.keys():
    word_frequencies[word] = word_frequencies[word]/max_frequency

# Calculer les scores de chaque phrase
sentence_tokens = [sent for sent in nlp(text).sents]
sentence_scores = {}
for sent in sentence_tokens:
    for word in sent:
        if word.text.lower() in word_frequencies.keys():
            if sent not in sentence_scores.keys():
                sentence_scores[sent] = word_frequencies[word.text.lower()]
            else:
                sentence_scores[sent] += word_frequencies[word.text.lower()]

# Sélectionner les phrases les plus importantes pour le résumé
select_length = int(len(sentence_tokens)*0.3)
summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)

# Construire le résumé final
final_summary = ' '.join([word.text for word in summary])
summary = ''.join(final_summary)
# Afficher le texte original et le résumé
print("Texte original:")
print(text)
print("\nRésumé:")
print(summary)

