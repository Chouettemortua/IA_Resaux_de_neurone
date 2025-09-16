#process du txt en csv

import re
import csv

def parse_line(line):
    # Exemple de ligne :  
    # ΔSleep Duration = -1.20 | ΔPhysical Activity Level = -100.00 | ΔStress Level = 0.50 | ΔBMI Category = -0.20 | ΔBlood Pressure = 40.00 → prédiction : 0.556 (impact : -0.111)
    
    # Sépare la partie avant la flèche → et la partie après
    try:
        features_part, pred_part = line.split('→')
    except ValueError:
        return None  # ligne mal formée

    # Récupère la prédiction et l'impact via regex
    m = re.search(r"prédiction\s*:\s*([\d\.]+)\s*\(impact\s*:\s*([-+]?\d*\.?\d+)\)", pred_part)
    if not m:
        return None

    prediction = float(m.group(1))
    impact = float(m.group(2))

    # On ne garde que les lignes où impact ≠ 0
    if impact == 0:
        return None

    # Parse les features et leurs deltas
    features = features_part.split('|')
    parsed_features = []
    for feat in features:
        feat = feat.strip()
        # Format attendu : ΔFeature Name = value
        m_feat = re.match(r"Δ(.+?)\s*=\s*([-+]?\d*\.?\d+)", feat)
        if m_feat:
            name = m_feat.group(1).strip()
            val = float(m_feat.group(2))
            parsed_features.append((name, val))

    return parsed_features, prediction, impact

def process_file(input_path, output_path):
    all_rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            parsed = parse_line(line)
            if parsed:
                features, pred, impact = parsed
                # Pour CSV, on va aplatir features en colonnes : Feature_1, Delta_1, Feature_2, Delta_2, ...
                row = {}
                for i, (fname, fval) in enumerate(features, start=1):
                    row[f'Feature_{i}'] = fname
                    row[f'Delta_{i}'] = fval
                row['Prediction'] = pred
                row['Impact'] = impact
                all_rows.append(row)

    # Trie par impact décroissant
    all_rows.sort(key=lambda x: x['Impact'], reverse=True)

    # Trouver le max de features pour créer les colonnes CSV
    max_features = max(len([k for k in row if k.startswith('Feature_')]) for row in all_rows)

    # Créer les noms de colonnes dynamiquement
    columns = []
    for i in range(1, max_features + 1):
        columns.append(f'Feature_{i}')
        columns.append(f'Delta_{i}')
    columns.extend(['Prediction', 'Impact'])

    # Écriture CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for row in all_rows:
            # Remplir les colonnes absentes avec ''
            for col in columns:
                if col not in row:
                    row[col] = ''
            writer.writerow(row)

if __name__ == "__main__":
    input_txt_file = "TIPE/Saves/recommandations_sleep_quality.txt"
    output_csv_file = "TIPE/Saves/recommandations_sleep_quality.csv"
    process_file(input_txt_file, output_csv_file)
    print(f"Fichier CSV généré : {output_csv_file}")


# Code recommender_améliorations :

def recommander_améliorations(
    model,
    input_row,
    features,
    non_modifiables,
    max_values=None,
    seuil_qualite=0.5,
    verbose=True,
    max_combinaison=2,
    log_path=None,
):
    import numpy as np
    from itertools import combinations, product
    import sys

    input_row = input_row.flatten()
    base_pred = model.predict(input_row.reshape(1, -1))[0]
    model_type = model.get_model_type()

    original_stdout = sys.stdout
    if log_path:
        log_file = open(log_path, 'w', buffering=1)
        sys.stdout = log_file

    if verbose:
        print("Type de modèle :", model_type)
        print("Prédiction initiale :", base_pred)

    if model_type == "regression" and base_pred >= seuil_qualite:
        if log_path:
            sys.stdout = original_stdout
            log_file.close()
        return ["Votre qualité de sommeil semble correcte. Aucune amélioration nécessaire."]

    deltas = [0.05, 0.1, 0.2, 0.5, -0.05, -0.1, -0.2, -0.5]
    impacts = {}

    # Test univarié
    for i, feature in enumerate(features):
        if feature in non_modifiables:
            continue

        val = input_row[i]
        if val < 0 or val > 1:
            continue

        for d in deltas:
            new_val = np.clip(val + d, 0, 1)
            modified = input_row.copy()
            modified[i] = new_val
            new_pred = model.predict(modified.reshape(1, -1))[0]

            impact = new_pred - base_pred if model_type == "regression" else base_pred - new_pred

            if verbose and impact != 0:  # affiche tous sauf impact nul
                delta_real = d * max_values[feature] if max_values and feature in max_values else d
                print(f"  Δ{feature} réel = {delta_real:.2f}")
                print(f"  Δ{feature} = {d:.2f} → prédiction : {new_pred:.3f} (impact : {impact:.3f})")

            if impact > 0.005:
                best_so_far = impacts.get((feature,), (0, 0))
                if impact > best_so_far[0]:
                    impacts[(feature,)] = (impact, new_val - val)

    # Test multivarié
    for r in range(2, max_combinaison + 1):
        if verbose:
            print(f"\nTest des combinaisons de {r} features :\n")
        for combo in combinations([i for i, f in enumerate(features) if f not in non_modifiables], r):
            for deltas_combo in product(deltas, repeat=r):
                modified = input_row.copy()
                delta_real_strs = []
                for idx, d in zip(combo, deltas_combo):
                    val = input_row[idx]
                    new_val = np.clip(val + d, 0, 1)
                    modified[idx] = new_val
                    f = features[idx]
                    delta_real = d * max_values[f] if max_values and f in max_values else d
                    delta_real_strs.append(f"Δ{f} = {delta_real:.2f}")

                new_pred = model.predict(modified.reshape(1, -1))[0]
                impact = new_pred - base_pred if model_type == "regression" else base_pred - new_pred

                if verbose and impact != 0:  # affiche tous sauf impact nul
                    print(f"  {' | '.join(delta_real_strs)} → prédiction : {new_pred:.3f} (impact : {impact:.3f})")

                if impact > 0.005:
                    feature_names = tuple(features[i] for i in combo)
                    impacts[feature_names] = (impact, [modified[i] - input_row[i] for i in combo])

    # Restaure sortie standard
    if log_path:
        sys.stdout = original_stdout
        log_file.close()

    sorted_impacts = sorted(impacts.items(), key=lambda x: x[1][0], reverse=True)
    recommandations = []

    for feature_tuple, (impact, delta_list) in sorted_impacts:
        if len(feature_tuple) == 1:
            feature = feature_tuple[0]
            delta = delta_list
            direction = "augmenter" if delta > 0 else "réduire"
            real_delta = abs(delta * max_values[feature]) if max_values and feature in max_values else abs(delta)
            recommandations.append(
                f"{direction.capitalize()} {feature} de {real_delta:.1f} unités pourrait améliorer votre score de {impact:.3f}"
            )
        else:
            variation_strs = []
            for i, feature in enumerate(feature_tuple):
                delta = delta_list[i]
                real_delta = abs(delta * max_values[feature]) if max_values and feature in max_values else abs(delta)
                direction = "augmenter" if delta > 0 else "réduire"
                variation_strs.append(f"{direction} {feature} de {real_delta:.1f}")
            recommandations.append(f"{' ; '.join(variation_strs)} → amélioration estimée : {impact:.3f}")

    return recommandations if recommandations else ["Aucune recommandation claire détectée à partir des données."]

# morceau de lancement pour le fichier .txt
import numpy as np

sleep = None
X_train = None

test_extreme = np.array([1, 0.9, 5, 0.9, 1.0, 1.0, 4, 1.0, 1.0, 1.0]).reshape(1, -1)
max_values = {'Gender': 2, 'Age': 130, 'Occupation': 5, 'Sleep Duration': 24, 
                        'Physical Activity Level': 200, 'Stress Level': 10, 'BMI Category': 4, 
                        'Blood Pressure': 200, 'Heart Rate': 200, 'Daily Steps': 50000}
        
recommandations = recommander_améliorations(
model=sleep,
input_row=test_extreme,
features=X_train.columns,
non_modifiables=['Sleep Disorder', 'Quality of Sleep', 'Age', 'Occupation', 'Gender'],
max_values=max_values,
seuil_qualite=0.8, 
verbose=True, 
max_combinaison=6,
log_path="TIPE/Saves/recommandations_sleep_quality.txt"
)

for rec in recommandations:
    print("-", rec)



