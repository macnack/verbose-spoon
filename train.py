import math
import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger

import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.loggers import TensorBoardLogger, NeptuneLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
from src.config.default import get_cfg_defaults
from src.utils.misc import get_rank_zero_only_logger, setup_gpus, lower_config
from src.utils.profiler import build_profiler
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_edm import PL_EDM
import torch
loguru_logger = get_rank_zero_only_logger(loguru_logger)


def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("data_cfg_path", type=str, help="data config path")
    parser.add_argument("main_cfg_path", type=str, help="main config path")
    parser.add_argument("--exp_name", type=str, default="default_exp_name")
    parser.add_argument("--gpus", default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--strategy", type=str, default="ddp")
    parser.add_argument("--batch_size", type=int,
                        default=4, help="batch_size per gpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--pin_memory",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        default=True,
        help="whether loading data to pinned memory or not",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="pretrained checkpoint path, helpful for using a pre-trained coarse-only EDM",
    )
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    parser.add_argument("--num_sanity_val_steps", type=int, default=10)
    parser.add_argument("--benchmark", default=True)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--split_data_idx", type=int, default=0)
    parser.add_argument(
        "--parallel_load_data",
        default=False,
        action="store_true",
        help="load datasets in with multiple processes.",
    )
    parser.add_argument(
        "--disable_ckpt",
        action="store_true",
        help="disable checkpoint saving (useful for debugging).",
    )
    parser.add_argument(
        "--profiler_name",
        type=str,
        default=None,
        help="options: [inference, pytorch], or leave it unset",
    )

    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()
    rank_zero_only(pprint.pprint)(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility
    # TODO: Use different seeds for each dataloader workers
    # This is needed for data augmentation

    # scale lr and warmup-step automatically
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    config.TRAINER.WARMUP_STEP = math.floor(
        config.TRAINER.WARMUP_STEP / _scaling)

    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_EDM(config, pretrained_ckpt=args.ckpt_path, profiler=profiler)
    loguru_logger.info(f"EDM LightningModule initialized!")

    # lightning data
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"EDM DataModule initialized!")

    # TensorBoard Logger
    logger = TensorBoardLogger(
        save_dir="logs/tb_logs", name=args.exp_name, default_hp_metric=False
    )
    ckpt_dir = Path(logger.log_dir) / "checkpoints"
    neptune_logger = NeptuneLogger(
        name=args.exp_name,
        log_model_checkpoints=True,  # Upload model checkpoints to Neptune
        tags=[args.exp_name, "EDM-training"],
        capture_stdout=True,
        capture_stderr=True,
    )
    neptune_logger.log_hyperparams(vars(args))
    try:
        neptune_logger.run["config/main_cfg"].upload(args.main_cfg_path)
        neptune_logger.run["config/data_cfg"].upload(args.data_cfg_path)
    except Exception as e:
        loguru_logger.warning(f"Neptune: Failed to upload source code or configs. Error: {e}")
    # Callbacks
    # TODO: update ModelCheckpoint to monitor multiple metrics
    if config.DATASET.TRAINVAL_DATA_SOURCE in ['SyntheticHomography', 'synthetichomographypregen']:
        # For our homography dataset, we monitor the Mean Corner Error (MCE).
        # Lower is better, so the mode is 'min'.
        monitor_metric = 'MCE'
        monitor_mode = 'min'
        filename_format = "{epoch}-{MCE:.2f}"
        print(f"\n--- ModelCheckpoint configured to monitor '{monitor_metric}' (mode: {monitor_mode}) ---\n")
    else:
        # For the original datasets (ScanNet, MegaDepth), monitor auc@10.
        # Higher is better, so the mode is 'max'.
        monitor_metric = 'auc@10'
        monitor_mode = 'max'
        filename_format = "{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}"
        print(f"\n--- ModelCheckpoint configured to monitor '{monitor_metric}' (mode: {monitor_mode}) ---\n")
    
    ckpt_callback = ModelCheckpoint(
        monitor=monitor_metric,
        verbose=True,
        save_top_k=10,
        mode=monitor_mode,
        save_last=True,
        save_weights_only=True,
        dirpath=str(ckpt_dir),
        filename=filename_format,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)

    # Lightning Trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.gpus,
        strategy=args.strategy,
        num_nodes=args.num_nodes,
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_every_n_steps,
        # limit_val_batches=1,
        num_sanity_val_steps=args.num_sanity_val_steps,
        benchmark=True,
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        logger=[logger, neptune_logger],
        sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
        use_distributed_sampler=False,
        profiler=profiler,
    )
    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")
    trainer.fit(model, datamodule=data_module, ckpt_path=None)


if __name__ == "__main__":
    main()
