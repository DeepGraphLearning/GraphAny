import os
from datetime import datetime
from uuid import uuid4

import numpy as np
import rootutils
import torch
import wandb
from omegaconf import OmegaConf

from .config import save_config
from .logging import ExpLogger, logger

root = rootutils.find_root(__file__)
from pytorch_lightning.utilities import rank_zero_only


def set_seed(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_unique_id(cfg):
    """Generate a Unique ID (UID) for (1) File system (2) Communication between submodules
    By default, we use time and UUID4 as UID. UIDs could be overwritten by wandb or UID specification.
    """
    #
    if cfg.get("uid") is not None and cfg.wandb.id is not None:
        assert cfg.get("uid") == cfg.wandb.id, "Confliction: Wandb and uid mismatch!"
    cur_time = datetime.now().strftime("%b%-d-%-H:%M-")
    given_uid = cfg.wandb.id or cfg.get("uid")
    uid = given_uid if given_uid else cur_time + str(uuid4()).split("-")[0]
    return uid


def init_experiment(cfg):
    # Prevent ConfigKeyError when accessing non-existing keys
    OmegaConf.set_struct(cfg, False)
    # Add global attribute to reproduce hydra configs at ease.
    wandb_init(cfg)
    local_rank = cfg.get("local_rank", 0)
    set_seed(local_rank + cfg.seed)
    cfg.uid = generate_unique_id(cfg)
    for directory in cfg.dirs.values():
        os.makedirs(directory, exist_ok=True)
    cfg_out_file = cfg.dirs.output + "hydra_cfg.yaml"
    save_config(cfg, cfg_out_file, as_global=True)
    exp_logger = ExpLogger(cfg)
    exp_logger.save_file_to_wandb(cfg_out_file, base_path=cfg.dirs.output, policy="now")
    exp_logger.info(f"Local_rank={local_rank}, working_dir={cfg.dirs.temp}")
    return cfg, exp_logger


@rank_zero_only
def wandb_init(cfg) -> None:
    os.environ["WANDB_WATCH"] = "false"
    try:
        wandb_tags = cfg.wandb.tags.split(".")
        mode = "online" if cfg.use_wandb else "disabled"

        # ! Create wandb session
        if cfg.wandb.id is None:
            # First time running, create new wandb
            os.makedirs(cfg.dirs.wandb_cache, exist_ok=True)
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                dir=cfg.dirs.wandb_cache,
                reinit=True,
                config=OmegaConf.to_object(cfg),
                name=cfg.wandb.name,
                tags=wandb_tags,
                mode=mode,
            )
        else:  # Resume from previous run
            logger.critical(f"Resume from previous wandb run {cfg.wandb.id}")
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                reinit=True,
                resume="must",
                id=cfg.wandb.id,
                mode=mode,
            )
            cfg.wandb.is_master_process = False  # Running as a sub_process
        if mode == "online":
            cfg.wandb.id, cfg.wandb.name, cfg.wandb.url = (
                wandb.run.id,
                wandb.run.name,
                wandb.run.url,
            )
        return
    except Exception as e:
        print(f"An error occurred during wandb initialization: {e}\n'WANDB NOT INITIALIZED.'")

    # If wandb not already initialized, set all wandb settings to None.
    os.environ["WANDB_DISABLED"] = "true"
    cfg.wandb_on = False
    cfg.wandb.id, cfg.wandb.name, cfg.wandb.url = None, None, None
    return
