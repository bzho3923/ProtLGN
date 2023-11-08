import argparse
import os
import json
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mutant_dataset', type=str, default='data/mutant_dataset')
    parser.add_argument('--mutant_site', type=int, default=1, help='1: site1, 2: site2')
    parser.add_argument('--out_file', type=str, default=None)
    args = parser.parse_args()
    
    proteins = sorted(os.listdir(args.mutant_dataset))
    score_info = {}
    for p in tqdm(proteins):
        df = pd.read_table(os.path.join(args.mutant_dataset, p))
        if args.mutant_site:
            df = df[df['mutat_pt_num'] == args.mutant_site]
        sp_score = spearmanr(df['true_score'], df['pred_score'])[0]
        score_info[p[:-4]] = sp_score
    
    if args.out_file is None:
        args.out_file = f"spearmanr_{args.mutant_site}.json"
    with open(args.out_file, "w") as f:
        json.dump(score_info, f)