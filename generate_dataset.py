import os
import sys
import argparse
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle
# Import the compatible dataset class
try:
    from src.datasets.synthetic_homography import SyntheticHomographyDataset
except ImportError:
    print("Error: Could not import SyntheticHomographyDataset.")
    print("Please make sure you run this script from the root of your project.")
    sys.exit(1)


def save_sample(batch, index, output_dir):
    """
    Saves a single pre-generated data dictionary to a .pt file and
    returns cleaned metadata.
    """
    try:
        sample_path = output_dir / "samples" / f"{index:08d}.pt"
        metadata_pickle_path = output_dir / "samples" / f"{index:08d}_metadata.pkl"
        data_item, metadata = batch
        # Squeeze the tensors to remove the batch dimension added by the DataLoader
        data_to_save = {key: val.squeeze(0) if isinstance(val, torch.Tensor) else val for key, val in data_item.items()}

        torch.save(data_to_save, str(sample_path))
        with open(metadata_pickle_path, 'wb') as f:
            pickle.dump(metadata, f)
        # --- SOLUTION ---
        # The DataLoader converts metadata to Tensors. We convert it back to
        # native Python types before creating the CSV.
        cleaned_metadata = {}
        for key, val in metadata.items():
            if isinstance(val, torch.Tensor):
                # .squeeze(0) removes the batch dim; then convert to Python type
                squeezed_val = val.squeeze(0)
                if squeezed_val.numel() == 1:
                    cleaned_metadata[key] = squeezed_val.item() # For single values
                else:
                    cleaned_metadata[key] = squeezed_val.tolist() # For lists/arrays
            elif isinstance(val, list) and len(val) == 1:
                # Dataloader can wrap single items (like paths) in a list
                cleaned_metadata[key] = val[0]
            else:
                cleaned_metadata[key] = val

        # Return the cleaned metadata
        cleaned_metadata['paird_id_index'] = index
        cleaned_metadata['sample_path'] = sample_path.relative_to(output_dir).as_posix()
        return cleaned_metadata

    except Exception as e:
        print(f"Error saving sample {index}: {e}")
        return None

def main(args):
    print(f"Starting dataset generation...")
    # ... (print statements)
    
    output_path = Path(args.output_dir)
    
    # --- Process Training and Validation Sets ---
    for mode in ["train", "val"]:
        if mode == "train":
            print("\n--- Generating Training Set ---")
            num_samples_to_gen = args.num_train
            source_list = args.source_train_list
            dataset_seed = 42 + 10000
            base_dir = output_path / "train"
        else: # mode == "val"
            print("\n--- Generating Validation Set ---")
            num_samples_to_gen = args.num_val
            source_list = args.source_val_list
            dataset_seed = 42
            base_dir = output_path / "val"
        
        if num_samples_to_gen == 0:
            continue
            
        (base_dir / "samples").mkdir(parents=True, exist_ok=True)

        # 1. Initialize the dataset instance
        generator_dataset = SyntheticHomographyDataset(
            root_dir=args.source_dir,
            list_path=source_list,
            img_resize=(args.img_w, args.img_h),
            num_samples=num_samples_to_gen,
            seed=dataset_seed,
            load_to_ram=True
        )

        # 2. Create a DataLoader to manage the parallel workers
        loader = DataLoader(
            generator_dataset,
            batch_size=1, # We process one sample at a time
            shuffle=False, # We iterate sequentially from 0 to N-1
            num_workers=args.num_workers,
            pin_memory=False, # Not needed for saving to disk
            persistent_workers=True if args.num_workers > 0 else False,
        )
        
        metadata_list = []
        # The 'enumerate' provides the global index for saving files
        for i, data_item in enumerate(tqdm(loader, desc=f"Generating {mode} set")):
            metadata = save_sample(data_item, i, base_dir)
            if metadata:
                metadata_list.append(metadata)

        # 3. Save the metadata to a CSV file
        df = pd.DataFrame(metadata_list)
        df.to_csv(base_dir / "metadata.csv", index=False)
        print(f"Saved {mode} metadata to {base_dir / 'metadata.csv'}")

    print("\nDataset generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-generate the Synthetic Homography Dataset.")
    parser.add_argument("--source_dir", type=str, required=True, help="Path to large source TIFFs.")
    parser.add_argument("--source_train_list", type=str, required=True, help="Text file listing source images.")
    parser.add_argument("--source_val_list", type=str, required=True, help="Text file listing source images.")
    parser.add_argument("--output_dir", type=str, default="data/pregenerated_homography", help="Directory to save the generated dataset.")
    
    parser.add_argument("--num_train", type=int, default=8192, help="Number of training pairs.")
    parser.add_argument("--num_val", type=int, default=1600, help="Number of validation pairs.")
    
    parser.add_argument("--img_w", type=int, default=640, help="Width of generated images.")
    parser.add_argument("--img_h", type=int, default=480, help="Height of generated images.")

    parser.add_argument("--num_workers", type=int, default=8, help="Number of CPU workers for parallel generation.")

    args = parser.parse_args()
    main(args)