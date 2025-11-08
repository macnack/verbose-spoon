from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path

import torch
import numpy as np
import lightning.pytorch as pl
from matplotlib import pyplot as plt

from src.edm import EDM
from src.edm.utils.supervision import (
    compute_supervision_coarse,
    compute_supervision_fine,
)
from src.losses.edm_loss import EDMLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    compute_homography_errors,
    aggregate_metrics,
)
from src.utils.plotting import make_matching_figures
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler


class PL_EDM(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)

        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(
            config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1
        )

        # Matcher: EDM
        self.matcher = EDM(config=_config["edm"])
        self.loss = EDMLoss(_config)

        # Pretrained weights
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location="cpu")[
                "state_dict"]
            self.matcher.load_state_dict(state_dict, strict=True)
            logger.info(f"Load '{pretrained_ckpt}' as pretrained checkpoint")

        # Testing
        self.warmup = False
        self.reparameter = False
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.total_ms = 0
        self.dump_dir = dump_dir

        # outputs
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == "linear":
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + (
                    self.trainer.global_step / self.config.TRAINER.WARMUP_STEP
                ) * abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
            elif self.config.TRAINER.WARMUP_TYPE == "constant":
                pass
            else:
                raise ValueError(
                    f"Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}"
                )

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def _trainval_inference(self, batch):
        with self.profiler.profile("Compute coarse supervision"):
            with torch.autocast(enabled=False, device_type="cuda"):
                compute_supervision_coarse(batch, self.config)

        with self.profiler.profile("EDM"):
            with torch.autocast(enabled=self.config.EDM.MP, device_type="cuda"):
                self.matcher(batch)

        # with self.profiler.profile("Compute fine supervision"):
        #     with torch.autocast(enabled=False, device_type='cuda'):
        #         compute_supervision_fine(batch, self.config)

        with self.profiler.profile("Compute losses"):
            with torch.autocast(enabled=self.config.EDM.MP, device_type="cuda"):
                self.loss(batch)

    def _compute_metrics(self, batch):
        # This function now decides which evaluation to run
        dataset_name = batch['dataset_name'][0]

        if dataset_name == 'SyntheticHomography':
            # --- Homography Evaluation ---
            compute_homography_errors(batch, self.config)
            rel_pair_names = list(zip(*batch["pair_names"]))
            bs = batch["image0"].size(0)
            metrics = {
                "identifiers": [f"pair_{batch['pair_id'][b].item()}" for b in range(bs)],
                "corner_error": batch["corner_error"],
                "inliers": batch["inliers"],
                "num_matches": [batch["mconf"].shape[0]],
            }
        
        else:
            # --- Original Pose Evaluation ---
            compute_symmetrical_epipolar_errors(batch)
            compute_pose_errors(batch, self.config)
            rel_pair_names = list(zip(*batch["pair_names"]))
            bs = batch["image0"].size(0)
            metrics = {
                "identifiers": ["#".join(rel_pair_names[b]) for b in range(bs)],
                "epi_errs": [
                    (batch["epi_errs"].reshape(-1, 1))[batch["m_bids"] == b]
                    .reshape(-1)
                    .cpu()
                    .numpy()
                    for b in range(bs)
                ],
                "R_errs": batch["R_errs"],
                "t_errs": batch["t_errs"],
                "inliers": batch["inliers"],
                "num_matches": [batch["mconf"].shape[0]],
            }

        ret_dict = {"metrics": metrics}
        return ret_dict, rel_pair_names

    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        
        # logging
        if (
            self.trainer.global_rank == 0
            and self.global_step % self.trainer.log_every_n_steps == 0
        ):
            # scalars
            for k, v in batch["loss_scalars"].items():
                self.logger.experiment.add_scalar(
                    f"train/{k}", v, self.global_step)

            # figures
            if self.config.TRAINER.ENABLE_PLOTTING:
                compute_symmetrical_epipolar_errors(
                    batch
                )  # compute epi_errs for each match
                figures = make_matching_figures(
                    batch, self.config, self.config.TRAINER.PLOT_MODE
                )
                for k, v in figures.items():
                    self.logger.experiment.add_figure(
                        f"train_match/{k}", v, self.global_step
                    )

        out = {"loss": batch["loss"]}
        self.log("loss", batch["loss"], prog_bar=True, rank_zero_only=True)

        # avoid significant memory growth
        # self.train_step_outputs.append(out)
        return out

    def on_after_backward(self) -> None:
        for n, p in self.named_parameters():
            if p.grad is None:
                print(n)
        return super().on_after_backward()

    def on_train_epoch_end(self):
        pass  # avoid significant memory growth during training
        # outputs = self.train_step_outputs
        # avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # if self.trainer.global_rank == 0:
        #     self.logger.experiment.add_scalar(
        #         'train/avg_loss_on_epoch', avg_loss,
        #         global_step=self.current_epoch)
        # self.train_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch)

        ret_dict, _ = self._compute_metrics(batch)

        val_plot_interval = max(
            self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {self.config.TRAINER.PLOT_MODE: []}
        if batch_idx % val_plot_interval == 0:
            figures = make_matching_figures(
                batch, self.config, mode=self.config.TRAINER.PLOT_MODE
            )

        out = {
            **ret_dict,
            "loss_scalars": batch["loss_scalars"],
            "figures": figures,
        }
        self.validation_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        
        # Guard against empty outputs (e.g., if validation is skipped)
        if not outputs:
            return

        # Handle multiple validation sets (dataloaders)
        multi_outputs = [outputs] if not isinstance(outputs[0], list) else outputs
        
        # This will hold the value of the primary metric from each validation set
        # for checkpointing purposes.
        primary_metric_values = []

        # Determine the primary metric to monitor based on the dataset type.
        # We check the first output of the first validation set to determine this.
        first_output = multi_outputs[0][0]
        # A simple heuristic: check if 'H_errs' (or your corner_error key) exists.
        is_homography_eval = 'H_errs' in first_output['metrics'] or 'corner_error' in first_output['metrics']

        if is_homography_eval:
            # For homography, we'll monitor Mean Corner Error (MCE), where lower is better.
            primary_metric_name = 'MCE'
        else:
            # For pose datasets, we monitor the standard pose AUC@10, where higher is better.
            primary_metric_name = 'auc@10'

        for valset_idx, outputs_per_set in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            # if self.trainer.is_sanity_check:
            #     cur_epoch = -1

            # 1. Gather loss scalars from all GPUs
            _loss_scalars = [o["loss_scalars"] for o in outputs_per_set]
            loss_scalars = {
                k: flattenList(all_gather([_ls.get(k, torch.tensor(0.0)) for _ls in _loss_scalars]))
                for k in _loss_scalars[0]
            }

            # 2. Gather and aggregate all metrics from all GPUs
            _metrics = [o["metrics"] for o in outputs_per_set]
            # Create a full list of all keys from all dictionaries
            all_keys = set(k for d in _metrics for k in d.keys())
            metrics = {
                k: flattenList(all_gather(flattenList([_me.get(k, []) for _me in _metrics])))
                for k in all_keys
            }

            # `aggregate_metrics` will now calculate all relevant metrics (pose AUC, H AUC, MCE, etc.)
            # and return them in a single dictionary.
            val_metrics_4tb = aggregate_metrics(
                metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config
            )

            # --- Store the value of the primary metric for this validation set ---
            if primary_metric_name in val_metrics_4tb:
                primary_metric_values.append(val_metrics_4tb[primary_metric_name])

            # 3. Gather figures for visualization (only on rank 0)
            _figures = [o["figures"] for o in outputs_per_set]
            # Ensure figures dict is not empty before processing
            if _figures and _figures[0]:
                figure_keys = _figures[0].keys()
                figures = {
                    k: flattenList(gather(flattenList([_me.get(k, []) for _me in _figures])))
                    for k in figure_keys
                }
            else:
                figures = {}


            # --- Logging to TensorBoard (only on rank 0) ---
            if self.trainer.global_rank == 0:
                # Log average loss scalars
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.logger.experiment.add_scalar(
                        f"val_{valset_idx}/avg_{k}", mean_v, global_step=cur_epoch
                    )

                # Log all computed metrics
                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(
                        f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch
                    )

                # Log figures
                for k, v_list in figures.items():
                    for plot_idx, fig in enumerate(v_list):
                        self.logger.experiment.add_figure(
                            f"val_match_{valset_idx}/{k}/pair-{plot_idx}",
                            fig,
                            cur_epoch,
                            close=True,
                        )
            
            # Close all matplotlib figures to prevent memory leaks
            plt.close("all")

        # --- Log the primary metric for ModelCheckpoint on all ranks ---
        if primary_metric_values:
            # Average the primary metric across all validation sets if there are multiple
            mean_primary_metric = np.mean(primary_metric_values)
            self.log(
                primary_metric_name,          # e.g., 'MCE' or 'auc@10'
                torch.tensor(mean_primary_metric),
                sync_dist=True,               # Ensure all GPUs have the same value
            )
        
        # Clear the outputs list for the next validation epoch
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        if self.config.EDM.HALF:
            self.matcher = self.matcher.eval().half()

        # Following EfficientLoFTR
        if not self.warmup:
            if self.config.EDM.HALF:
                for i in range(50):
                    self.matcher(batch)
            else:
                with torch.autocast(enabled=self.config.EDM.MP, device_type="cuda"):
                    for i in range(50):
                        self.matcher(batch)
            self.warmup = True

        torch.cuda.synchronize()
        if self.config.EDM.HALF:
            self.start_event.record()
            self.matcher(batch)
            self.end_event.record()
            torch.cuda.synchronize()
            self.total_ms += self.start_event.elapsed_time(self.end_event)
        else:
            with torch.autocast(enabled=self.config.EDM.MP, device_type="cuda"):
                self.start_event.record()
                self.matcher(batch)
                self.end_event.record()
                torch.cuda.synchronize()
                self.total_ms += self.start_event.elapsed_time(self.end_event)

        ret_dict, rel_pair_names = self._compute_metrics(batch)

        if self.dump_dir is not None:
            with self.profiler.profile("dump_results"):
                # dump results for further analysis
                pair_names = list(zip(*batch["pair_names"]))
                bs = batch["image0"].shape[0]
                dumps = []
                
                # Check which dataset is being used to decide what to dump
                dataset_name = batch['dataset_name'][0]

                for b_id in range(bs):
                    item = {}
                    mask = batch["m_bids"] == b_id
                    
                    if dataset_name == 'SyntheticHomography':
                        # --- DUMP LOGIC FOR HOMOGRAPHY ---
                        keys_to_save = {"mkpts0_f", "mkpts1_f", "mconf"}
                        item["pair_id"] = batch["pair_id"][b_id].item()
                        item["identifier"] = f"pair_{item['pair_id']}"
                        
                        for key in keys_to_save:
                            item[key] = batch[key][mask].cpu().numpy()
                        
                        # Save homography-specific metrics and ground truth
                        item["corner_error"] = batch["corner_error"][b_id]
                        item["inliers"] = batch["inliers"][b_id]
                        item["homography_gt"] = batch["homography"][b_id].cpu().numpy()
                        item["homography_est"] = batch["H_est"][b_id].cpu().numpy()
                        item["sym_transfer_err"] = batch["sym_transfer_err"][b_id].cpu().numpy()
                    
                    else:
                        # --- ORIGINAL DUMP LOGIC FOR POSE ---
                        keys_to_save = {"mkpts0_f", "mkpts1_f", "mconf", "epi_errs"}
                        item["pair_names"] = pair_names[b_id]
                        item["identifier"] = "#".join(rel_pair_names[b_id])

                        for key in keys_to_save:
                            item[key] = batch[key][mask].cpu().numpy()
                        
                        for key in ["R_errs", "t_errs", "inliers"]:
                            item[key] = batch[key][b_id]
                    
                    dumps.append(item)
                ret_dict["dumps"] = dumps

        self.test_step_outputs.append(ret_dict)
        return ret_dict

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        # metrics: dict of list, numpy
        _metrics = [o["metrics"] for o in outputs]

        metrics = {
            k: flattenList(gather(flattenList([_me[k] for _me in _metrics])))
            for k in _metrics[0]
        }

        # dump
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o["dumps"]
                                 for o in outputs])  # [{...}, #bs*#batch]
            dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
            logger.info(
                f"Prediction and evaluation results will be saved to: {self.dump_dir}"
            )

        # [{key: [{...}, *#bs]}, *#batch]
        if self.trainer.global_rank == 0:
            val_metrics_4tb = aggregate_metrics(
                metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config
            )

            logger.info("\n" + pprint.pformat(val_metrics_4tb))
            print(
                "Averaged Matching time over 1500 pairs: {:.2f} ms".format(
                    self.total_ms / 1500
                )
            )
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / "EDM_pred_eval", dumps)

        self.test_step_outputs.clear()
