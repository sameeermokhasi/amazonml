import pandas as pd
import random
import os

print("Loading matching results...")
df = pd.read_csv("output/matching_results.tsv", sep="\t")
total = len(df)
df['match_count'] = df['matched_entity_ids'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) and str(x).strip() else 0)

empty_count = (df['match_count'] == 0).sum()
non_empty_count = total - empty_count
avg_matches = df[df['match_count'] > 0]['match_count'].mean() if non_empty_count > 0 else 0

print("="*60)
print(f"Total Source 1 Entities: {total:,}")
print(f"Entities WITH matches:   {non_empty_count:,} ({(non_empty_count/total)*100:.1f}%)")
print(f"Entities EMPTY:          {empty_count:,} ({(empty_count/total)*100:.1f}%)")
print(f"Average matches per non-empty entity: {avg_matches:.2f}")
print("="*60)

# 10 random samples
print("\n--- 10 RANDOM SAMPLES ---")
print("Since we don't have the text data loaded in this quick script, here are the raw ID mappings:")
samples = df.sample(10)
for _, row in samples.iterrows():
    print(f"S1: {row['source1_entity_id']}  --->  Matches: {row['matched_entity_ids']}")

