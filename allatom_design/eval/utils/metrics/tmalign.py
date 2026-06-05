import os
import subprocess

import numpy as np


def _compute_tmalign_score(pdb_1: str, pdb_2: str) -> tuple[float, float]:
    """
    Compute TM-score between two PDBs. This uses TM-align, so
    we don't need to check if the residue indices are aligned.

    Returns:
        - tmalign_score_1: TM-align score normalized by length of pdb_1
        - tmalign_score_2: TM-align score normalized by length of pdb_2
    """
    try:
        cmd = [f"{os.environ['SOFTWARE_PATH']}/tmalign/TMalign", pdb_1, pdb_2]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout

        tmalign_score_1 = None
        tmalign_score_2 = None

        # Parse TM-align output
        for line in output.splitlines():
            if line.startswith("TM-score="):
                parts = line.strip().split()
                score = float(parts[1])
                if "Chain_1" in line:
                    tmalign_score_1 = score
                elif "Chain_2" in line:
                    tmalign_score_2 = score

        tmalign_score_1 = tmalign_score_1 if tmalign_score_1 is not None else np.nan
        tmalign_score_2 = tmalign_score_2 if tmalign_score_2 is not None else np.nan
    except Exception as e:
        print(f"Error computing TM-align score: {e}")
        tmalign_score_1 = np.nan
        tmalign_score_2 = np.nan

    return tmalign_score_1, tmalign_score_2
