import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dir', required=True)
parser.add_argument('--states', default=2, type=int)
parser.add_argument('--samples', default=5, type=int)
parser.add_argument('--out', default='./')
args = parser.parse_args()

import pretty
from pymol import cmd, util

import os, glob

for state in range(args.states):
    for sample in range(args.samples):
        cmd.load(f"{args.dir}/state{state}_sample{sample}.cif", f"state{state}_sample{sample}")
        cmd.color("grey90", f"state{state}_sample{sample}")
        cmd.set_util.cnc(f"state{state}_sample{sample}")
        if state > 0 or sample > 0:
            cmd.align(f"state{state}_sample{sample}", f"state0_sample0")
        cmd.disable(f"state{state}_sample{sample}")
for state in range(args.states):
    for sample in range(args.samples):
        cmd.enable(f"state{state}_sample{sample}")
    cmd.png(f"{args.out}/state{state}.png", 640, 640, ray=1)
    for sample in range(args.samples):
        cmd.disable(f"state{state}_sample{sample}")
    