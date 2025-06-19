import pandas as pd
import numpy as np
from tqdm import tqdm
from fpdgd.data.LetorDataset import LetorDataset
import ast
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance

# Use letor dataset instead of lmdb to get normalized features
blacklisted_features = list(range(18, 33))+list(range(75, 78))+list(range(93, 96))
ltr_dataset = LetorDataset("dataset/aol4foltr/letor.txt", 103, query_level_norm=True, blacklisted_features=blacklisted_features)

df = pd.read_csv('dataset/aol_dataset_top10000.csv')
df['candidate_doc_ids'] = df['candidate_doc_ids'].apply(lambda x: list(ast.literal_eval(x)))

feat_groups_by_user = []
# Group queries by user and collect document features for each user
for user_id, user_queries in tqdm(df.groupby('user_id'), desc="Collecting user features"):

    # Iterate through each query for this user
    user_feats = []
    for _, query_row in user_queries.iterrows():
        qid = str(query_row['query_id'])
        doc_index = query_row['candidate_doc_ids'].index(query_row['doc_id'])
        
        # Get normalized features using the all_features matrix instead of individual docid
        try:
            all_features = ltr_dataset.get_all_features_by_query(qid)
        except KeyError:
            print(f"Query {qid} not found in dataset")
            continue
        
        feats = all_features[doc_index]

        assert len(feats) == 103
        for feat_idx, feat in enumerate(feats):
            assert feat >= 0 and feat <= 1, f"Feature {feat_idx} out of range: {feat}"
        
        user_feats.append(feats)
    
    feat_groups_by_user.append(user_feats)


# For each feature index, collect values across all groups
feature_distributions = []
for feat_idx in range(103):
    # Get the distribution of this feature for each group
    group_distributions = []
    all_feat_values = []
    for group in feat_groups_by_user:
        # Extract feature values for this index from the group
        feat_values = [feats[feat_idx] for feats in group]
        if not feat_values:  # Skip empty groups
            continue
        all_feat_values.extend(feat_values)
        # Create histogram (distribution) for these values
        hist, _ = np.histogram(feat_values, bins=20, range=(0,1), density=True)
        # Ensure the histogram is valid (sums to 1 and has no NaN values)
        if np.isnan(hist).any() or np.sum(hist) == 0:
            continue
        group_distributions.append(hist)
    
    if not group_distributions:  # Skip if no valid distributions
        feature_distributions.append(0)
        continue
        
    mean_value = np.mean(all_feat_values)
    print(f"Feature {feat_idx+1}: mean = {mean_value:.4f}")
    
    # Compute average divergence between all pairs of distributions
    n_groups = len(group_distributions)
    total = 0
    num_pairs = 0
    for i in range(n_groups):
        for j in range(i+1, n_groups):
            try:
                emd = wasserstein_distance(range(20), range(20), group_distributions[i], group_distributions[j])
                total += emd
                num_pairs += 1
            except ValueError:
                continue
    
    avg = total / num_pairs if num_pairs > 0 else 0
    feature_distributions.append(avg)

# Create dataframe and save to CSV
results_df = pd.DataFrame({
    'feature_index': range(103),
    'divergence': feature_distributions
})
results_df.to_csv('results/feature_divergences.csv', index=False)
