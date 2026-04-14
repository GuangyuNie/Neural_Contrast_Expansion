import torch
import numpy as np
import json
from pathlib import Path
from copy import deepcopy
from sce_pipeline import *  # adjust import path as needed
import re
from helper.npcf_calculation import *
from pde import *
# from bessel_network_wave import *
# ------------------ user configuration ------------------
# directory containing binary microstructures saved as .npz with key 'micro' or 'microstructure'
micro_dir = Path("material_gt/saved_microstructures/")
output_dir = Path("pde_data")
output_dir.mkdir(exist_ok=True, parents=True)

# SCE material parameters (can be overridden per run if desired)
eps_matrix = 1.0  # background/host permittivity (eps_q)
eps_inclusion = 2.0  # inclusion permittivity (eps_p)

# expansion order: 2 or 3 or 4
n_order = 3

# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# helper to load .npz microstructure file; expects binary array under key 'micro' or 'microstructure'
def load_binary_micro(path: Path):
    """
    Load a microstructure saved with save_microstructure.
    Returns (microstructure: np.ndarray, metadata: dict)
    """
    with np.load(path, allow_pickle=True) as data:
        micro = data['micro']
        meta = json.loads(data['metadata'].item() if hasattr(data['metadata'], 'item') else data['metadata'])
    return micro, meta

# Iterate over microstructures, skip those with idx > 10
pattern = re.compile(r'_idx(\d+)\.npz$')
for micro_path in sorted(micro_dir.glob('*.npz')):
    m = pattern.search(micro_path.name)
    if m:
        idx_val = int(m.group(1))
        if idx_val > 10:
            continue  # 跳过 idx 大于 10 的文件

    micro_tensor, meta = load_binary_micro(micro_path)

    gt = effective_conductivity_pde(micro_tensor, eps_matrix, eps_inclusion).compute()

    # build eps_tensor (keep full 2x2 complex)
    eps_tensor = gt

    # save binary npz
    tag = micro_path.stem
    out_npz = output_dir / f"{tag}_eps_tensor.npz"
    np.savez(
        out_npz,
        eps_tensor=eps_tensor,
    )

    # write JSON summary
    json_list = []
    # since SCE returns single tensor per setting (no frequency sweep), output principal components
    entry = {
        "eps_tensor": {
            "xx": {"real": float(np.real(eps_tensor[0, 0])), "imag": float(np.imag(eps_tensor[0, 0]))},
            "yy": {"real": float(np.real(eps_tensor[1, 1])), "imag": float(np.imag(eps_tensor[1, 1]))},
            "xy": {"real": float(np.real(eps_tensor[0, 1])), "imag": float(np.imag(eps_tensor[0, 1]))},
            "yx": {"real": float(np.real(eps_tensor[1, 0])), "imag": float(np.imag(eps_tensor[1, 0]))},
        },
        "n_order": n_order,
    }
    with open(output_dir / f"{tag}_eps_tensor.json", 'w') as f:
        json.dump(entry, f, indent=2)

    print(f"[SCE RESULT] micro={micro_path.name} saved to {out_npz}")
