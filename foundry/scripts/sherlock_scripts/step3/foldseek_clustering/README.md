# denovoval step3 Foldseek clustering

Clusters denovoval step2 generated samples with Foldseek.

Default input:

```text
/scratch/users/zhkim216/datasets/evaluation_datasets/curation/ver2/outputs/denovoval/step2/generated_samples/full
```

Default full output:

```text
/scratch/users/zhkim216/datasets/evaluation_datasets/curation/ver2/outputs/denovoval/step3/foldseek_clustering
```

The script discovers condition directories dynamically. Empty conditions are
recorded in `manifest.json` and `validation_report.json`, not hard-coded, so the
same command can be rerun after currently empty groups are filled.

Smoke:

```bash
bash scripts/sherlock_scripts/step3/foldseek_clustering/run_smoke.sh
```

Full Sherlock run:

```bash
sbatch scripts/sherlock_scripts/step3/foldseek_clustering/run_denovoval_step3_foldseek_clustering.sbatch --overwrite
```

The Foldseek command uses default `easy-cluster` settings except `-c 0.8` and
the requested thread count.
