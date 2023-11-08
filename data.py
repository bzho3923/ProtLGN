import os, sys
# set path


current_dir = os.getcwd()
sys.path.append(current_dir)
import argparse
from src.Dataset.protein_dataset import Protein
from src.Dataset.mutant_dataset import Mutant
from src.Dataset.localization_dataset import Localization
from src.Dataset.dataset_utils import NormalizeProtein



# Download protein_dataset or Loading your dataset.
def protein_dataset(args,split):
    dataset = Protein(
        args.protein_dataset,
        split=split,
        divide_num=1,
        divide_idx=0,
        c_alpha_max_neighbors=args.c_alpha_max_neighbors,
        set_length=None,
        random_sampling=True,
        p=args.p,
        use_sasa=args.use_sasa,
        use_bfactor=args.use_bfactor,
        use_dihedral=args.use_dihedral,
        use_coordinate=args.use_coordinate,
        use_denoise=args.use_denoise,
        noise_type=args.noise_type
        )
    return dataset


# Download multi_mutantDataset or Loading your dataset.
def mutant_dataset(args):
    """
    #Or you can take it to your weight:
    # forexample:
    dataset_arg['normal_file']='/home/yuguang/xinyexiong/protein/63w_k10/Train/mean_attr.pt'
    """
    mm_dataset = Mutant(
        args.mutant_dataset,
        args.mutant_dataset.split('/')[-1]+f"k{args.c_alpha_max_neighbors}",
        args.mutant_dataset+"/DATASET",
        c_alpha_max_neighbors=args.c_alpha_max_neighbors,
        pre_transform=NormalizeProtein(filename=f'data/cath40_k{args.c_alpha_max_neighbors}/mean_attr.pt'),
    )
    return mm_dataset

def loc_dataset(args):
    loc_dataset = Localization(
        args.loc_dataset,
        args.loc_dataset.split('/')[-1]+f"k{args.c_alpha_max_neighbors}",
        args.loc_dataset+"/DATASET",
        args.split,
        c_alpha_max_neighbors=args.c_alpha_max_neighbors,
        id2label_file=args.loc_dataset+"/id2label.txt"
    )
    return loc_dataset
        

def get_dataset(args):
    # load protein dataset like CATHs40
    train_dataset = protein_dataset(args, "train")
    val_dataset = protein_dataset(args, "val")
    test_dataset = protein_dataset(args, "test")
    
    # load multiple mutation dataset 
    mm_dataset = mutant_dataset(args)
    
    # print info
    print(f"Number of train graphs: {len(train_dataset)}")
    print(f"Number of validation graphs: {len(val_dataset)}")
    print(f"Number of test graphs: {len(test_dataset)}")
    print(f"Number of mutation proteins: {len(mm_dataset)}")
    print("-"*50)
    return train_dataset, val_dataset, test_dataset, mm_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protein_dataset', type=str, default='data/cath40_k10')
    parser.add_argument('--mutant_dataset', type=str, default=None)
    parser.add_argument('--loc_dataset', type=str, default=None)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--c_alpha_max_neighbors', type=int, default=10)
    parser.add_argument('--p', type=float, default=0.1)
    parser.add_argument('--use_sasa', action='store_true', default=False)
    parser.add_argument('--use_bfactor', action='store_true', default=False)
    parser.add_argument('--use_dihedral', action='store_true', default=False)
    parser.add_argument('--use_coordinate', action='store_true', default=False)
    parser.add_argument('--use_denoise', action='store_true', default=False)
    parser.add_argument('--noise_type', type=str, default='wild')
    parser.add_argument('--build_cath', action='store_true', default=False)
    parser.add_argument('--build_mutant', action='store_true', default=False)
    parser.add_argument('--build_loc', action='store_true', default=False)
    args = parser.parse_args()
    
    if args.build_cath:
        protein_dataset(args, "train")
    elif args.build_mutant:
        mutant_dataset(args)
    elif args.build_loc:
        loc_dataset(args)
    