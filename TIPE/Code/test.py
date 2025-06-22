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
