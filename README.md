# Moteur de recherche et classification de décisions de justice en arabe
Projet de fouille de texte (Java, Eclipse).
- Prétraitement avec SAFAR (stemmer Light10)
- Matrices d'occurrences et TF-IDF, recherche par similarité cosinus
- Classifieur Naïve Bayes : 94 % d'exactitude sur 85 documents de test cachés

## Lancer le projet
1. Importer le projet dans Eclipse.
2. Ajouter SAFAR_v2.jar dans lib/ puis dans le Build Path.
3. Placer le corpus dans corpus_classes/immobilier et corpus_classes/statut_personnel.
4. Lancer la classe Classification.
5. Les résultats s’affichent dans la console
