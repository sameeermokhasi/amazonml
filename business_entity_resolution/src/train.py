import sys
import argparse
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

def resolve_path(rel_path_str: str) -> Path:
    candidates = [
        Path(rel_path_str),
        Path("..") / rel_path_str,
        Path(__file__).resolve().parent.parent / rel_path_str,
        Path(__file__).resolve().parent.parent.parent / rel_path_str,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return Path(rel_path_str).resolve()

def load_ground_truth(gt_path: Path):
    gt = pd.read_csv(gt_path, sep="\t")
    gt_map = {}
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        matches = row['matched_entity_ids']
        if pd.isna(matches) or not str(matches).strip():
            gt_map[str(s1_id)] = set()
        else:
            gt_map[str(s1_id)] = set([x.strip() for x in str(matches).split(",")])
    return gt_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True)
    args = parser.parse_args()

    feat_path = resolve_path(args.features)
    if not feat_path.exists():
        sys.exit(f"Features file not found: {feat_path}")

    print(f"Loading features from {feat_path}...")
    df = pd.read_parquet(feat_path)
    print(f"Loaded {len(df)} candidate pairs.")

    gt_path = resolve_path("dataset/train/train_ground_truth.tsv")
    print(f"Loading ground truth from {gt_path}...")
    gt_map = load_ground_truth(gt_path)

    # Assign labels
    labels = []
    for _, row in df.iterrows():
        s1_id = str(row['source1_entity_id'])
        c_id = str(row['candidate_entity_id'])
        if s1_id in gt_map and c_id in gt_map[s1_id]:
            labels.append(1)
        else:
            labels.append(0)
    
    df['label'] = labels
    print(f"Labels assigned. Positive samples: {sum(labels)} / {len(labels)}")

    feature_cols = [
        "name_token_sort_ratio",
        "name_token_set_ratio",
        "name_jaro_winkler",
        "address_token_sort_ratio",
        "address_jaro_winkler",
        "postal_code_exact",
        "house_number_exact",
        "same_first_name_token",
        "shared_token_count",
    ]
    for col in ["name_tfidf_cosine", "address_tfidf_cosine"]:
        if col in df.columns:
            feature_cols.append(col)

    X = df[feature_cols]
    y = df['label']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training HistGradientBoostingClassifier model...")
    model = HistGradientBoostingClassifier(random_state=42, max_iter=100)
    model.fit(X_train, y_train)
    
    model_path = resolve_path("models/lgbm_matcher.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

    val_preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_preds)
    
    # Save validation predictions for evaluation
    df_val = df.loc[X_val.index].copy()
    df_val['probability'] = val_preds
    val_preds_path = resolve_path("dataset_processed/val_predictions.parquet")
    df_val[['source1_entity_id', 'candidate_entity_id', 'label', 'probability']].to_parquet(val_preds_path, index=False)
    print(f"Saved validation predictions to {val_preds_path}")
    
    print("="*80)
    print(f"Validation AUC: {auc:.4f}")
    if auc > 0.999:
        print("[FLAGGED] Suspiciously perfect AUC (> 0.999). Potential data leak!")
    elif auc < 0.5:
        print("[FLAGGED] AUC worse than random (< 0.5). Something is wrong!")
    else:
        print("[OK] Real AUC is above 0.5.")
    
    # Feature importance proxy
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt.fit(X_train, y_train)
    print("\nFeature Importances (proxy via Decision Tree):")
    importances = dt.feature_importances_
    feat_imps = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    for feat, imp in feat_imps:
        print(f"  {feat}: {imp}")
    print("="*80)

if __name__ == "__main__":
    main()
