# Identification des utilisateurs de Copilote

Ce code présente la résolution du problème du traitement des traces d’utilisation du logiciel Copilote. Ce logiciel, édité par la société Infologic, dans le secteur de l’agro-alimentaire, est utilisé par de nombreux acteurs industriels. L’objectif est de travailler sur des modèles d’analyse automatique des traces d’utilisation, qui permettent d’améliorer l’offre auprès des clients.


## 📋 Description

Le problème posé consiste à identifier automatiquement quel utilisateur interagit avec
un logiciel, simplement à partir des traces de son utilisation. Chez Infologic, les testeurs
utilisent différents profils pour évaluer diverses fonctionalités mais doivent indiquer manuellement quel profil ils ont utilisé.

L’objectif de ce projet est donc de concevoir un modèle de classification capable de
reconnaître un utilisateur à partir de ses actions sur le logiciel, en ayant à disposition les
données des précédentes sessions. Cette approche de machine learning permet donc de
simplifier les processus de test et de détecter d’éventuelles tentatives d’intrusion comme
un pirate qui utilise le compte d’un utilisateur réel.

Le jeu de données d’entraînement fourni contient un grand volume de traces d’utilisation, qu’il faut analyser et traiter pour entraîner un modèle performant. Une prédiction
est ensuite effectuée sur le jeu de donnée test et sera évaluée à l’aide du F1 score, qui
calcule la moyenne entre la précision et le rappel.

## 🚀 Fonctionnalités

- **Extraction de features avancées** :
  - Features temporelles (durée de session, écarts entre actions)
  - Features de diversité (entropie, ratio de diversité)
  - Features de fréquence (actions, écrans, configurations, chaînes)
  - Features binaires (présence d'actions spécifiques)

- **Preprocessing complet** :
  - Encodage One-Hot des variables catégorielles
  - Normalisation des features numériques (StandardScaler)
  - Gestion des données manquantes

- **Modèle de classification** :
  - Random Forest Classifier avec paramètres optimisés
  - Validation interne avec stratification
  - Métrique : F1-score macro

## 📦 Installation

1. Clonez le repository :
```bash
git clone <https://github.com/NBRS22/be-data-sciences>
cd BE-Data-Science-II
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt 
```

## 💻 Utilisation

1. Exécutez le script principal :
```bash
python main.py
```

2. Le fichier `submission.csv` sera généré avec les prédictions.

## 📁 Structure du projet

```
BE-Data-Science-II/
│
├── data/
│   ├── train.csv          # Dataset d'entraînement
│   └── test.csv           # Dataset de test
│
├── main.py                # Script principal
├── requirements.txt       # Dépendances Python
├── README.md             # Documentation
└── submission.csv        # Fichier de prédictions (généré)
```

## 🔍 Format des données

Le dataset attendu doit avoir le format suivant :
- **Colonne 0** : Identifiant utilisateur (train uniquement)
- **Colonne 1** : Type de navigateur
- **Colonnes 2+** : Actions et timestamps
  - Actions : Chaînes de caractères (format : `action(ecran)<config>$chaine$`)
  - Timestamps : Chaînes commençant par "t" (format : `t12345`)

## 🎯 Features extraites

### Features temporelles
- Durée de session
- Nombre d'actions
- Temps moyen entre actions
- Écart maximal entre actions
- Écart-type des écarts temporels
- Irrégularité temporelle

### Features de diversité
- Nombre d'actions uniques
- Entropie de Shannon
- Ratio de diversité
- Entropie normalisée
- Ratio de répétition
- Taux de répétitions consécutives

### Features de fréquence
- Fréquence des top actions
- Fréquence des écrans (top 20)
- Fréquence des configurations (top 20)
- Fréquence des chaînes (top 20)
- Features binaires (présence d'actions)

### Features catégorielles
- Navigateur
- Première action
- Dernière action

## ⚙️ Configuration du modèle

Le modèle utilise un **Random Forest Classifier** avec les paramètres suivants :
- `n_estimators`: 600
- `max_depth`: 20
- `min_samples_split`: 5
- `min_samples_leaf`: 2
- `max_features`: "sqrt"
- `class_weight`: "balanced"
- `random_state`: 44

## 📊 Métriques

Le modèle est évalué avec le **F1-score macro** sur un split de validation interne (80/20).

## 🛠️ Technologies utilisées

- **Python**
- **NumPy** : Calculs numériques
- **Pandas** : Manipulation de données
- **Scikit-learn** : Machine Learning

## 📝 Notes

- Les top 50 actions les plus fréquentes sont conservées pour limiter la dimensionnalité
- Les top 20 éléments sont conservés pour les écrans, configurations et chaînes
- Les variables catégorielles sont encodées avec One-Hot Encoding
- Toutes les features numériques sont normalisées avec StandardScaler


## 👤 Auteurs

Projet développé par : 
- Adrien Baraton
- Zélie Brachet
- Nour EL Bachari 

