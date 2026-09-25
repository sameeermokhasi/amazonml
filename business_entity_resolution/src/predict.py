import sys
import subprocess
import pandas as pd
import joblib
from pathlib import Path
from write_output import write_results

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

def run_cmd(cmd):
    print(f"\n[RUNNING] {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    print("="*80)
    print("    BUSINESS ENTITY RESOLUTION - END-TO-END PREDICTION")
    print("="*80)
    
    # 1. Preprocess
    print("Skipping preprocessing (already completed).")
    # run_cmd("python3 src/preprocess.py --dataset_type test")

    # 2. Blocking
    run_cmd("python3 src/blocking.py --dataset_type test --output test_candidate_pairs.parquet --batch_size 100000")

    # 3. Features
    run_cmd("python3 src/features.py --dataset_type test --candidates dataset_processed/test_candidate_pairs.parquet --output dataset_processed/test_features.parquet")

    # 4. Load trained model and predict
    print("\nLoading test features...")
    df_feat = pd.read_parquet(resolve_path("dataset_processed/test_features.parquet"))
    
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
    X = df_feat[feature_cols]
    
    print("\nLoading trained model...")
    model_path = resolve_path("models/lgbm_matcher.joblib")
    if not model_path.exists():
        sys.exit(f"Model not found at {model_path}. Run train.py first.")
        
    model = joblib.load(model_path)
    
    print("\nPredicting match probabilities...")
    preds = model.predict_proba(X)[:, 1]
    df_feat['probability'] = preds
    
    # 5. Apply chosen threshold
    THRESHOLD = 0.50
    print(f"\nApplying threshold {THRESHOLD} to determine matches...")
    df_matches = df_feat[df_feat['probability'] >= THRESHOLD].copy()
    
    # 6. Apply one-to-one resolution
    print("\nApplying one-to-one resolution (candidate can only match one S1 entity)...")
    # Sort by probability descending so the first duplicate candidate is the highest probability one
    df_matches = df_matches.sort_values('probability', ascending=False)
    # Drop duplicates on candidate_entity_id, keeping the first
    df_matches = df_matches.drop_duplicates(subset=['candidate_entity_id'], keep='first')
    
    # 7 & 8. DataFrames for final output using write_output.py
    out_dir = resolve_path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating candidate_pairs.tsv...")
    df_cands = df_feat.groupby('source1_entity_id')['candidate_entity_id'].apply(list).reset_index()
    df_cands = df_cands.rename(columns={'candidate_entity_id': 'candidate_entity_ids'})
    
    test_s1_path = resolve_path("dataset/test/test_source1.tsv")
    test_s1 = pd.read_csv(test_s1_path, sep="\t")
    all_s1 = test_s1[['entity_id']].rename(columns={'entity_id': 'source1_entity_id'})
    
    # Ensure all test source1 entities are in candidate list
    df_cands_all = pd.merge(all_s1, df_cands, on='source1_entity_id', how='left')
    write_results(
        df_cands_all, 
        path=out_dir / "candidate_pairs.tsv", 
        id_col_name="candidate_entity_ids", 
        test_source1_path=test_s1_path
    )
    
    print("\nGenerating matching_results.tsv...")
    df_final = df_matches.groupby('source1_entity_id')['candidate_entity_id'].apply(list).reset_index()
    df_final = df_final.rename(columns={'candidate_entity_id': 'matched_entity_ids'})
    
    # Ensure all test source1 entities are in matching list
    df_final_all = pd.merge(all_s1, df_final, on='source1_entity_id', how='left')
    write_results(
        df_final_all, 
        path=out_dir / "matching_results.tsv", 
        id_col_name="matched_entity_ids", 
        test_source1_path=test_s1_path
    )
    
    # 9. Verify exact match
    num_s1_rows = len(test_s1)
    with open(out_dir / "matching_results.tsv", "r", encoding="utf-8") as f:
        num_out_rows = sum(1 for _ in f) - 1 # subtract header
        
    print("\n" + "="*80)
    print("                       VERIFICATION")
    print("="*80)
    print(f"Test Source1 Entities: {num_s1_rows:,}")
    print(f"Output rows in matching_results.tsv: {num_out_rows:,}")
    if num_s1_rows == num_out_rows:
        print("[OK] Exact match confirmed! Output is ready for submission.")
    else:
        print(f"[ERROR] Mismatch! Expected {num_s1_rows} but got {num_out_rows}.")
    print("="*80)

if __name__ == "__main__":
    main()
