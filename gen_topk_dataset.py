import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int)
args = parser.parse_args()

df = pd.read_csv('dataset/aol_dataset.csv', low_memory=False)
user_counts = df['user_id'].value_counts()
top_users = user_counts.nlargest(args.k).index
df = df[df['user_id'].isin(top_users)]
df.to_csv(f'dataset/aol_dataset_top{args.k}.csv', index=False)
print(f"dataset/aol_dataset_top{args.k}.csv saved")
