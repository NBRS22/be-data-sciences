import numpy as np
import pandas as pd
import re
from collections import Counter
from math import log
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

# =========================
#  1. Lecture des datasets
# =========================

def read_ds(filename: str):

    with open(filename, "r", encoding="utf-8") as f:
        lignes = [line.strip().split(",") for line in f]
    max_cols = max(len(l) for l in lignes)
    for l in lignes:
        l += [np.nan] * (max_cols - len(l))
    return pd.DataFrame(lignes)

train_df = read_ds("data/train.csv")

test_df = read_ds("data/test.csv")

# ========================
#  2. Regex et constantes
# ========================

pattern_ecran = re.compile(r"\((.*?)\)")
pattern_conf_ecran = re.compile(r"<(.*?)>")
pattern_chaine = re.compile(r"\$(.*?)\$")

TOP_K_ACTIONS = 50
TOP_K_SUBITEMS = 20

# =========================================================
#  3. Analyse du train pour repérer les éléments fréquents
# =========================================================

all_actions, first_actions_all = [], []
all_ecrans, all_confs, all_chaines = [], [], []

for _, row in train_df.iterrows():
    actions = [a for a in row[2:] if isinstance(a, str) and not a.startswith("t")]
    if not actions:
        continue
    all_actions.extend(actions)
    first_actions_all.append(actions[0])

    for a in actions:
        all_ecrans.extend(pattern_ecran.findall(a))
        all_confs.extend(pattern_conf_ecran.findall(a))
        all_chaines.extend(pattern_chaine.findall(a))

top_actions = [a for a, _ in Counter(all_actions).most_common(TOP_K_ACTIONS)]
top_first_actions = [a for a, _ in Counter(first_actions_all).most_common(10)]
top_ecrans = [a for a, _ in Counter(all_ecrans).most_common(TOP_K_SUBITEMS)]
top_confs = [a for a, _ in Counter(all_confs).most_common(TOP_K_SUBITEMS)]
top_chaines = [a for a, _ in Counter(all_chaines).most_common(TOP_K_SUBITEMS)]

# ==========================
#  4. Fonctions utilitaires
# ==========================

def entropy(actions):
    counts = Counter(actions)
    total = len(actions)
    return -sum((c/total) * log(c/total, 2) for c in counts.values()) if total > 0 else 0

def extract_features(row, is_train=True):
    user = row[0] if is_train else None
    browser = row[1]
    actions = [a for a in row[2:] if isinstance(a, str) and not a.startswith("t")]
    timestamps = [int(a[1:]) for a in row[2:] if isinstance(a, str) and a.startswith("t")]

    session_duration = max(timestamps) if timestamps else 0
    num_actions = len(actions)
    avg_time_between = session_duration / num_actions if num_actions > 0 else 0

    if timestamps:
        time_diffs = np.diff(sorted(timestamps))
        max_gap = np.max(time_diffs) if len(time_diffs) > 0 else 0
        std_gap = np.std(time_diffs) if len(time_diffs) > 0 else 0
    else:
        max_gap = 0
        std_gap = 0

    unique_actions = len(set(actions))
    entropy_score = entropy(actions)
    first_action = actions[0] if actions else "none"
    first_action = first_action if first_action in top_first_actions else "OTHER"
    last_action = actions[-1] if actions else "none"
    last_action = last_action if last_action in top_actions else "OTHER"

    # sous-éléments
    ecrans, confs, chaines = [], [], []
    for a in actions:
        ecrans.extend(pattern_ecran.findall(a))
        confs.extend(pattern_conf_ecran.findall(a))
        chaines.extend(pattern_chaine.findall(a))

    freq_ecrans = {f"freq_ecran_{e}": ecrans.count(e) / num_actions if num_actions > 0 else 0 for e in top_ecrans}
    freq_confs = {f"freq_conf_{c}": confs.count(c) / num_actions if num_actions > 0 else 0 for c in top_confs}
    freq_chaines = {f"freq_chaine_{s}": chaines.count(s) / num_actions if num_actions > 0 else 0 for s in top_chaines}

    freq_actions = {f"freq_{a}": actions.count(a) / session_duration if session_duration > 0 else 0 for a in top_actions}
    binary_actions = {f"has_{a}": int(a in actions) for a in top_actions}

    # Nouvelles features globales
    diversity_ratio = unique_actions / num_actions if num_actions > 0 else 0
    normalized_entropy = entropy_score / log(num_actions + 1, 2) if num_actions > 0 else 0
    avg_duration_per_action = session_duration / (num_actions + 1)

    # Nouvelles features globales
    diversity_ratio = unique_actions / num_actions if num_actions > 0 else 0
    normalized_entropy = entropy_score / log(num_actions + 1, 2) if num_actions > 0 else 0
    avg_duration_per_action = session_duration / (num_actions + 1)

    # Features additionnelles
    repetition_ratio = 1 - diversity_ratio

    # Compte les répétitions consécutives
    consecutive_repeats = sum(
        1 for i in range(1, len(actions)) if actions[i] == actions[i - 1]
    )
    repeat_rate = consecutive_repeats / num_actions if num_actions > 0 else 0

    # Rapport du temps du dernier timestamp sur la durée totale
    first_to_last_ratio = (
        (timestamps[-1] / (max(timestamps) + 1)) if len(timestamps) > 0 else 0
    )

    # Coefficient de variation temporel
    time_irregularity = (std_gap / (np.mean(np.diff(sorted(timestamps))) + 1)) if len(timestamps) > 1 else 0


    features = {
        "browser": browser,
        "num_actions": num_actions,
        "session_duration": session_duration,
        "avg_time_between": avg_time_between,
        "max_gap": max_gap,
        "std_gap": std_gap,
        "unique_actions": unique_actions,
        "entropy": entropy_score,
        "diversity_ratio": diversity_ratio,
        "normalized_entropy": normalized_entropy,
        "avg_duration_per_action": avg_duration_per_action,
        "repetition_ratio": repetition_ratio,
        "repeat_rate": repeat_rate,
        "first_to_last_ratio": first_to_last_ratio,
        "time_irregularity": time_irregularity,
        "first_action": first_action,
        "last_action": last_action,
        **freq_ecrans, **freq_confs, **freq_chaines,
        **freq_actions, **binary_actions
    }

    if is_train:
        features["user"] = user
    return features

# ============================
#  5. Extraction des features
# ============================

train_features = train_df.apply(lambda r: extract_features(r, is_train=True), axis=1)
test_features = test_df.apply(lambda r: extract_features(r, is_train=False), axis=1)

train_data = pd.DataFrame(list(train_features))
test_data = pd.DataFrame(list(test_features))

# =========================================
#  6. Encodage des variables catégorielles
# =========================================

enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
categorical_cols = ["browser", "first_action", "last_action"]

enc.fit(train_data[categorical_cols])
train_encoded = enc.transform(train_data[categorical_cols])
test_encoded = enc.transform(test_data[categorical_cols])

encoded_cols = enc.get_feature_names_out(categorical_cols)

train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_cols)
test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_cols)

X_train = pd.concat([train_data.drop(columns=["user", *categorical_cols]), train_encoded_df], axis=1)
y_train = train_data["user"]
X_test = pd.concat([test_data.drop(columns=categorical_cols), test_encoded_df], axis=1)

# =============================
#  7. Split validation interne
# =============================
scaler = StandardScaler()
numeric_cols = X_train.select_dtypes(include=[np.number]).columns

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])


X_t, X_val, y_t, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=44, stratify=y_train)

# ===========================
#  8. Entraînement du modèle
# ===========================

model = RandomForestClassifier(
    n_estimators=600,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=44,
    n_jobs=-1,
    oob_score=True
)


model.fit(X_t, y_t)
y_pred = model.predict(X_val)
score = f1_score(y_val, y_pred, average="macro")
print(f"📈 Macro-F1 (validation interne) : {score:.4f}")

#from sklearn.model_selection import cross_val_score
#cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1)
#print(f"🔁 CV F1-macro moyenne : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ===========================
#  9. Prédiction sur le test
# ===========================

y_test_pred = model.predict(X_test)

submission = pd.DataFrame({
    "RowId": range(1,len(y_test_pred)+1),
    "prediction": y_test_pred.astype(str)
})

submission.to_csv("submission.csv", index=False)

print("Fichier submission.csv généré !")