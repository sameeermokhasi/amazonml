import pandas as pd
from pathlib import Path

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
        s1_id = str(row['source1_entity_id'])
        matches = row['matched_entity_ids']
        if pd.isna(matches) or not str(matches).strip():
            gt_map[s1_id] = set()
        else:
            gt_map[s1_id] = set([x.strip() for x in str(matches).split(",") if x.strip()])
    return gt_map

def sweep_thresholds(val_predictions_path: str, gt_path: str, thresholds=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]):
    print(f"Loading validation predictions from {val_predictions_path}...")
    df = pd.read_parquet(resolve_path(val_predictions_path))
    
    print(f"Loading ground truth from {gt_path}...")
    gt_map = load_ground_truth(resolve_path(gt_path))
    
    val_entities = df['source1_entity_id'].unique()
    print(f"Evaluating {len(val_entities)} unique Source 1 entities in validation set...")
    
    print("-" * 60)
    print(f"{'Threshold':<15} | {'Macro F0.5 Score':<20}")
    print("-" * 60)
    
    best_f05 = -1
    best_thresh = -1
    
    for thresh in thresholds:
        # Keep candidates with prob >= threshold
        pred_df = df[df['probability'] >= thresh]
        
        # Group by source1_entity_id
        pred_map = pred_df.groupby('source1_entity_id')['candidate_entity_id'].apply(set).to_dict()
        
        f05_scores = []
        
        for s1_id in val_entities:
            true_matches = gt_map.get(s1_id, set())
            pred_matches = pred_map.get(s1_id, set())
            
            if len(true_matches) == 0:
                if len(pred_matches) == 0:
                    f05 = 1.0
                else:
                    f05 = 0.0
            else:
                tp = len(true_matches.intersection(pred_matches))
                precision = tp / len(pred_matches) if len(pred_matches) > 0 else 0.0
                recall = tp / len(true_matches)
                
                if precision == 0.0 and recall == 0.0:
                    f05 = 0.0
                else:
                    f05 = (1.25 * precision * recall) / ((0.25 * precision) + recall)
            
            f05_scores.append(f05)
            
        macro_f05 = sum(f05_scores) / len(f05_scores)
        print(f"{thresh:<15.2f} | {macro_f05:<20.4f}")
        
        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_thresh = thresh
            
    print("-" * 60)
    print(f"Best Threshold: {best_thresh} (F0.5 = {best_f05:.4f})")
    
def main():
    val_preds_path = "dataset_processed/val_predictions.parquet"
    gt_path = "dataset/train/train_ground_truth.tsv"
    sweep_thresholds(val_preds_path, gt_path)

if __name__ == "__main__":
    main()
