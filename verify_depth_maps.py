import numpy as np
from pathlib import Path

def verify_depth_maps(depth_dir="depth_maps/numpy"):
    depth_dir = Path(depth_dir)
    if not depth_dir.exists():
        print(f"Directory {depth_dir} not found.")
        return
    files = list(depth_dir.glob("*.npy"))
    print(f"Total depth maps: {len(files)}")
 
    sequences = set(f.name.split('_')[0] for f in files)
    for seq in sorted(sequences):
        seq_files = list(depth_dir.glob(f"{seq}_*.npy"))
        print(f"  Sequence {seq}: {len(seq_files)} files")

    sample = np.load(files[0])
    valid = np.sum(sample > 0)
    total = sample.size
    print(f"Sample coverage: {valid}/{total} ({100*valid/total:.1f}%)")

if __name__ == "__main__":
    verify_depth_maps()
