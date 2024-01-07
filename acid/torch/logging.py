from lightning.pytorch.loggers import CSVLogger, WandbLogger

from acid import conf


def get_logger(name, logdir, use_wandb=False):
    if use_wandb:
        return WandbLogger(project=name, save_dir=conf.NOTEBOOKS_TMP_DIR)

    # add dummy method watch (used by some loggers, not available for CSVLogger)
    logger = CSVLogger(logdir, name=name)
    logger.watch = lambda _: None
    return logger
