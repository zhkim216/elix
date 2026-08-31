from collections import defaultdict
from typing import Dict

import lightning as L
import torch
from lightning.pytorch.utilities import grad_norm
from omegaconf import DictConfig
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LinearLR
from torchtyping import TensorType

import logging

from allatom_design.model.ema.phema import PowerFunctionEMA
from allatom_design.model.lr_schedule import InverseSqrtLR, NoamLR
from allatom_design.model.seq_denoiser.sd_loss import SDLoss
from allatom_design.model.seq_denoiser.sd_model import SeqDenoiser
from allatom_design.utils.checkpoint_utils import (
    elix_mpnn_config,
    migrate_elix_feature_projection_state_dict,
)

logger = logging.getLogger(__name__)


_MULTI_HEAD_LATEST_STATS = {
    "head_exp_log_scale",
    "field_head_exp_log_scale",
}


def _merge_multi_head_stats(
    accumulated: dict[str, torch.Tensor],
    update: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    for name, value in update.items():
        value = value.detach().float()
        if name in _MULTI_HEAD_LATEST_STATS or name not in accumulated:
            accumulated[name] = value.clone()
        else:
            accumulated[name] = accumulated[name] + value
    return accumulated


def _reduce_multi_head_stats(
    local_stats: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not local_stats:
        return {}
    names = sorted(local_stats)
    shapes = {name: local_stats[name].shape for name in names}
    sizes = {name: local_stats[name].numel() for name in names}
    packed = torch.cat([local_stats[name].reshape(-1) for name in names])

    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(packed)
        world_size = torch.distributed.get_world_size()

    reduced = {}
    offset = 0
    for name in names:
        value = packed[offset:offset + sizes[name]].view(shapes[name])
        if name in _MULTI_HEAD_LATEST_STATS:
            value = value / world_size
        reduced[name] = value
        offset += sizes[name]
    return reduced


def _multi_head_metrics(
    stats: dict[str, torch.Tensor],
    phase: str,
) -> dict[str, torch.Tensor]:
    metrics = {}
    exp_log_scale = stats["head_exp_log_scale"]
    field_exp_log_scale = stats.get("field_head_exp_log_scale")
    h_count = stats["head_h_count"]
    J_count = stats["head_J_count"]
    for head_idx in range(exp_log_scale.numel()):
        head_prefix = f"{phase}/head_{head_idx}"
        metrics[f"{head_prefix}/exp_log_scale"] = exp_log_scale[head_idx]
        if field_exp_log_scale is not None:
            metrics[f"{head_prefix}/field_exp_log_scale"] = (
                field_exp_log_scale[head_idx]
            )
        if h_count > 0:
            metrics[f"{head_prefix}/gated_h_rms"] = torch.sqrt(
                stats["head_gated_h_sq_sum"][head_idx] / h_count
            )
        if J_count > 0:
            metrics[f"{head_prefix}/gated_J_rms"] = torch.sqrt(
                stats["head_gated_J_sq_sum"][head_idx] / J_count
            )

    for gate_name in ("node_gate", "edge_gate"):
        count_name = f"{gate_name}_count"
        if count_name not in stats or stats[count_name] <= 0:
            continue
        count = stats[count_name]
        gate_prefix = f"{phase}/{gate_name}"
        for head_idx in range(exp_log_scale.numel()):
            metrics[
                f"{gate_prefix}/head_{head_idx}/mean_weight"
            ] = stats[f"{gate_name}_weight_sum"][head_idx] / count
            metrics[
                f"{gate_prefix}/head_{head_idx}/argmax_fraction"
            ] = stats[f"{gate_name}_argmax_count"][head_idx] / count
        metrics[f"{gate_prefix}/normalized_entropy"] = (
            stats[f"{gate_name}_entropy_sum"] / count
        )
        metrics[f"{gate_prefix}/tie_fraction"] = (
            stats[f"{gate_name}_tie_count"] / count
        )
    return metrics


class LitSeqDenoiser(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        cfg = elix_mpnn_config(cfg)
        self.cfg = cfg
        self.model = SeqDenoiser(cfg.model)

        if cfg.train.compile.compile_model:
            print(f"Using torch.compile to optimize model performance...")
            self.model = torch.compile(self.model,
                                        backend=cfg.train.compile.compile_backend,
                                        mode=cfg.train.compile.mode,
                                        fullgraph=cfg.train.compile.fullgraph,
                                        dynamic=cfg.train.compile.dynamic)

        self.use_phema = cfg.model.get("ema", {}).get("use_phema", True)
        if self.use_phema:
            # Use EDM2 post-hoc EMA
            self.ema_tracker = PowerFunctionEMA(self.model)

        # Set up loss
        self.loss = SDLoss(cfg.loss)
        self._multi_head_train_stats: dict[str, torch.Tensor] = {}
        self._multi_head_val_stats: dict[str, torch.Tensor] = {}
        self._multi_head_last_optimizer_step = 0
        self.save_hyperparameters()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["state_dict"] = migrate_elix_feature_projection_state_dict(
            self,
            checkpoint["state_dict"],
        )
        hyper_parameters = checkpoint.get("hyper_parameters", {})
        if "cfg" in hyper_parameters:
            hyper_parameters["cfg"] = elix_mpnn_config(hyper_parameters["cfg"])


    def setup(self, stage: str):
        if stage == "fit":
            # At start of training, load in pretrained modules if needed
            if self.cfg.resume.ckpt_path is None:
                self.model.setup()

    def forward(self, batch, **kwargs):
        return self.model(batch, **kwargs)

    def on_train_start(self):
        # Initialize EMA trackers at the start of training (if using phema)
        if self.use_phema:
            self.ema_tracker.reset()
        self._multi_head_last_optimizer_step = int(self.trainer.global_step)

    @staticmethod
    def _pop_non_tensor_fields(batch: dict) -> dict:
        """Remove non-tensor entries from ``batch`` and return them.

        torch.compile's dynamo guards on Python-level structures (e.g. the
        contents of ``batch["example_id"]`` string lists) and will recompile
        on every step if those values change, eventually hitting the
        recompile limit and falling back to eager. Stripping them before
        the forward avoids that.
        """
        meta_fields = {k: batch[k] for k in list(batch) if not isinstance(batch[k], torch.Tensor)}
        for k in meta_fields:
            del batch[k]
        return meta_fields

    def training_step(self, batch: dict[str, TensorType["b ..."]], batch_idx: int):
        meta_fields = self._pop_non_tensor_fields(batch)
        outputs = self(batch)
        batch.update(meta_fields)

        loss, aux, aux_sum_count = self.loss(outputs, batch, return_aux=True)

        self._accumulate_multi_head_output(
            self._multi_head_train_stats,
            outputs,
        )

        # Logging
        self._log(batch, outputs, aux, batch_idx, phase="train", aux_sum_count=aux_sum_count)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            if self.use_phema:
                # Update EMA tracker
                self.ema_tracker.update(t=self.trainer.global_step)

        optimizer_step = int(self.trainer.global_step)
        if optimizer_step <= self._multi_head_last_optimizer_step:
            return
        self._multi_head_last_optimizer_step = optimizer_step
        log_interval = int(self.cfg.logging.log_every_n_steps)
        if (
            log_interval > 0
            and optimizer_step % log_interval == 0
            and self._multi_head_train_stats
        ):
            reduced_stats = _reduce_multi_head_stats(self._multi_head_train_stats)
            self.log_dict(
                _multi_head_metrics(reduced_stats, "train_multihead_potts"),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=False,
            )
            self._multi_head_train_stats = {}

    def on_train_epoch_end(self):
        # Report non-standard AA token violations accumulated by SDLoss
        # over the epoch (single device->host sync, once per epoch).
        count = int(self._nonstd_aa_violation_tensor().item())
        if count > 0:
            logger.warning(
                f"seq_loss was computed over {count} non-standard AA tokens this epoch"
            )
        self._nonstd_aa_violation_tensor().zero_()

    def _nonstd_aa_violation_tensor(self) -> torch.Tensor:
        """Return the SDLoss violation accumulator, regardless of compile wrapping."""
        return self.loss._nonstd_aa_violation_count

    @staticmethod
    def _multi_head_stats_from_outputs(outputs: dict | None) -> dict[str, torch.Tensor] | None:
        if not outputs:
            return None
        potts_aux = outputs.get("potts_decoder_aux")
        if not potts_aux:
            return None
        return potts_aux.get("multi_head_stats")

    @classmethod
    def _accumulate_multi_head_output(
        cls,
        accumulator: dict[str, torch.Tensor],
        outputs: dict | None,
    ) -> None:
        stats = cls._multi_head_stats_from_outputs(outputs)
        if stats is not None:
            _merge_multi_head_stats(accumulator, stats)

    def _accumulate_multi_head_validation_output(
        self,
        outputs: dict | None,
    ) -> None:
        self._accumulate_multi_head_output(self._multi_head_val_stats, outputs)

    def on_validation_epoch_start(self) -> None:
        self._multi_head_val_stats = {}

    def on_validation_epoch_end(self) -> None:
        if not self._multi_head_val_stats:
            return

        reduced_stats = _reduce_multi_head_stats(self._multi_head_val_stats)
        metrics = _multi_head_metrics(
            reduced_stats,
            "val_multihead_potts",
        )
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )

    @staticmethod
    def _validation_metric_phase(metric_name: str) -> str:
        if metric_name.startswith(("seq_", "ligand_pocket_seq_")):
            return "val_seq"
        if metric_name.startswith("sidechain_"):
            return "val_sidechain"
        if metric_name.startswith(("potts_", "ligand_pocket_potts_")):
            return "val_potts"
        return "val"

    def _validation_forward(self, batch: dict, **kwargs):
        """Run validation eagerly when the training model is compiled.

        Keep the static compiled training model, but use its shared-parameter
        eager module for variable-shape validation.
        """
        validation_model = getattr(self.model, "_orig_mod", self.model)
        return validation_model(batch, **kwargs)


    def validation_step(self, batch: dict[str, TensorType["b ..."]], batch_idx: int, dataloader_idx: int = 0):
        # Lightning automatically disables grads + sets model to eval mode
        phase_suffix = ""
        validation_scn_context_ratio = self.cfg.eval.get("scn_context_ratio", 0.0)

        # Strip non-tensor fields once for the whole step; self._log needs
        # them back (e.g. batch["example_id"] for batch size), so restore
        # before any logging call.
        meta_fields = self._pop_non_tensor_fields(batch)
        outputs = self._validation_forward(
            batch,
            scn_context_ratio=validation_scn_context_ratio,
        )
        self._accumulate_multi_head_validation_output(outputs)
        _, aux, aux_sum_count = self.loss(outputs, batch, return_aux=True)

        # restore non-tensor fields
        batch.update(meta_fields)
        self._log(batch, outputs, aux, batch_idx, phase="val", phase_suffix=phase_suffix, aux_sum_count=aux_sum_count)

        # eval seq design over discrete sequence noise

        aux_t = defaultdict(list)
        aux_sc_list = []

        for eval_t in self.cfg.eval.eval_timesteps:
            B = batch["token_pad_mask"].shape[0]
            t_seq = torch.full((B, ), fill_value=eval_t).to(self.device)

            meta_fields = self._pop_non_tensor_fields(batch)
            outputs = self._validation_forward(
                batch,
                t=t_seq,
                scn_context_ratio=validation_scn_context_ratio,
            )
            _, aux, aux_sum_count = self.loss(outputs, batch, eval_total = False, return_aux=True)
            batch.update(meta_fields)

            key_suffix = f"_t{eval_t}"
            aux = {
                k: v
                for k, v in aux.items()
                if ("seq" in k) or ("potts" in k) or ("sidechain" in k)
            }
            self._log(batch, outputs, aux, batch_idx, phase="val", phase_suffix=phase_suffix, key_suffix=key_suffix, aux_sum_count=aux_sum_count)

            # aggregate across timesteps
            for k, v in aux.items():
                aux_t[k].append(v)
            aux_sc_list.append(aux_sum_count)

        # average across timesteps and log. The (sum, count) monitor metrics are
        # accumulated across timesteps so the reduced ratio is a count-weighted
        # average; the key set is identical every timestep, keeping the all-reduce
        # in _log rank-aligned.
        aux_t = {k: torch.stack(v).mean().item() for k, v in aux_t.items()}
        aux_sc_avg = {
            name: (
                torch.stack([d[name][0] for d in aux_sc_list]).sum(),
                torch.stack([d[name][1] for d in aux_sc_list]).sum(),
            )
            for name in (aux_sc_list[0] if aux_sc_list else {})
        }
        self._log(batch, None, aux_t, batch_idx, phase="val", phase_suffix=phase_suffix, key_suffix="_avg_t", aux_sum_count=aux_sc_avg)


    def _log(self,
             batch: dict[str, TensorType["b ..."]],
             outputs: dict[str, TensorType["b ..."]],
             aux: dict[str, float],
             batch_idx: int,
             phase: str,
             phase_suffix: str = "",
             key_suffix: str = "",
             aux_sum_count: dict[str, tuple] | None = None):
        """
        phase_suffix: used to differentiate between different phases of validation (e.g. different fixed sizes), should include a leading "/"
        key_suffix: adds a suffix to the key
        aux_sum_count: data-dependent monitor metrics as {name: (local_sum, local_count)}.
            Reduced here via a single fixed-shape all-reduce so the ratio is
            correct across ranks without the per-key sync_dist that would desync
            DDP when the key set differs between ranks.
        """
        bs = len(batch["example_id"])

        log_dict = {}
        for k, v in aux.items():
            metric_phase = self._validation_metric_phase(k) if phase == "val" else phase
            log_dict[f"{metric_phase}{phase_suffix}/{k}{key_suffix}"] = v

        self.log_dict(
            log_dict,
            on_step=(phase == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
            batch_size=bs,
        )

        self._log_sum_count_metrics(aux_sum_count, phase, phase_suffix, key_suffix)

    def _log_sum_count_metrics(self,
                               aux_sum_count: dict[str, tuple] | None,
                               phase: str,
                               phase_suffix: str,
                               key_suffix: str):
        """Log data-dependent monitor metrics as an unbiased cross-rank ratio.

        The all-reduce operates on a single fixed-shape tensor and runs on every
        rank regardless of which samples were present, so it stays lockstep with
        the other ranks. After reduction the ratio is identical on all ranks, so
        it is logged with sync_dist=False (no further collective).
        """
        if not aux_sum_count:
            return

        names = sorted(aux_sum_count)
        stat = torch.stack(
            [torch.stack(aux_sum_count[name]).float() for name in names]
        )  # [K, 2] -> (sum, count) per metric
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(stat)

        for i, name in enumerate(names):
            total_sum, total_count = stat[i, 0], stat[i, 1]
            if total_count <= 0:
                # Every rank sees the same reduced count, so this branch is taken
                # in lockstep; skipping self.log issues no collective.
                continue
            metric_phase = self._validation_metric_phase(name) if phase == "val" else phase
            self.log(
                f"{metric_phase}{phase_suffix}/{name}{key_suffix}",
                (total_sum / total_count).item(),
                on_step=(phase == "train"),
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=False,
                add_dataloader_idx=False,
                batch_size=int(total_count.item()),
            )


    def configure_optimizers(self):
        optim_cfg = self.cfg.optim
        if optim_cfg.optimizer == "adamw":
            optimizer = AdamW(list(self.model.parameters()) + list(self.loss.parameters()),
                            lr=optim_cfg.adamw.lr, eps=1.0e-15)
            scheduler = LinearLR(optimizer, start_factor=1e-3, end_factor=1, total_iters=optim_cfg.adamw.warmup_steps)
        elif optim_cfg.optimizer == "noam":
            optimizer = Adam(list(self.model.parameters()) + list(self.loss.parameters()),
                             lr=0, betas=(0.9, 0.98), eps=1e-9)
            scheduler = NoamLR(optimizer,
                               model_size=128,  # hardcoded to MPNN hidden dim
                               factor=optim_cfg.noam.factor,
                               warmup=optim_cfg.noam.warmup_steps)
        elif optim_cfg.optimizer == "adam_inv_sqrt":
            optimizer = Adam(list(self.model.parameters()) + list(self.loss.parameters()),
                             lr=0, betas=(0.9, 0.99), eps=1e-9)
            scheduler = InverseSqrtLR(optimizer,
                                      ref_lr=optim_cfg.adam_inv_sqrt.ref_lr,
                                      ref_steps=optim_cfg.adam_inv_sqrt.ref_steps,
                                      warmup_steps=optim_cfg.adam_inv_sqrt.warmup_steps)
        else:
            raise ValueError(f"Unknown optimizer: {optim_cfg.optimizer}")

        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                    }
                }


    def on_before_optimizer_step(self, optimizer):
        # Compute the 1-norm and 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        trainer = getattr(self, "trainer", None)
        has_logger = bool(getattr(trainer, "logger", None))
        for norm_type in [1, 2]:
            grad_norms = grad_norm(self.model, norm_type=norm_type)

            total_norm_key = f"grad_{float(norm_type)}_norm_total"
            if total_norm_key in grad_norms:
                total_norm = grad_norms[total_norm_key]
                self.log_dict({f"total_l{norm_type}_grad_norm": total_norm}, logger=has_logger)
