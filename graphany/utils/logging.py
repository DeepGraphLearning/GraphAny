import logging
import time
from collections import defaultdict
from contextlib import ContextDecorator
from datetime import datetime
from functools import wraps

import hydra
import wandb
from codetiming import Timer
from humanfriendly import format_timespan
from rich.console import Console
from rich.logging import RichHandler
from rich.pretty import pretty_repr

logger = rich_logger = logging.getLogger()
rich_handler = RichHandler(
    rich_tracebacks=False,
    tracebacks_suppress=[hydra],
    console=Console(width=165),
    enable_link_path=False,
)
logging.basicConfig(
    level="WARNING",
    format="%(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[rich_handler],
)
from lightning.pytorch.loggers import WandbLogger


class ExpLogger(WandbLogger):
    """Customized Logger with
    1. Wandb integration
    2. Rich logger
    3. Logging levels like info, debug, warning, error
    """

    def __init__(self, cfg):
        self.wandb_on = cfg.wandb.id is not None
        super().__init__(
            save_dir=cfg.dirs.temp,
            name=cfg.wandb.name,
            offline=cfg.wandb.id is None,
            project=cfg.wandb.project,
        )
        self.local_rank = cfg.get("local_rank", 0)
        rich_handler._log_render.prefix = cfg.logging.prefix
        rich_handler._log_render.prefix_color = cfg.logging.get("prefix_color", "red")
        self.logger = rich_logger  # Rich logger
        self.console = rich_handler.console  # Rich logger console
        self.logger.setLevel(getattr(logging, cfg.logging.level.upper()))
        logger.info("Logger initialized.")
        self.log_metric_to_stdout = (
            not self.wandb_on and self.local_rank <= 0
        ) or cfg.logging.log_wandb_metric_to_stdout

        pass_func = lambda *args, **kwargs: None

        # ! Migrated Rich Functions
        self.rule = self.console.rule if self.local_rank <= 0 else pass_func
        self.print = self.console.print if self.local_rank <= 0 else pass_func

        # ! Migrated Logger functions
        self.info = self.logger.info if self.local_rank <= 0 else pass_func
        self.critical = self.logger.critical if self.local_rank <= 0 else pass_func
        self.warning = self.logger.warning if self.local_rank <= 0 else pass_func
        self.debug = self.logger.debug if self.local_rank <= 0 else pass_func
        self.info = self.logger.info if self.local_rank <= 0 else pass_func
        self.error = self.logger.error if self.local_rank <= 0 else pass_func
        self.exception = self.logger.exception if self.local_rank <= 0 else pass_func

        # ! Experiment Metrics
        self.results = defaultdict(list)

    # ! Log functions
    def log(self, *args, level="", **kwargs):
        if self.local_rank <= 0:
            self.logger.log(getattr(logging, level.upper()), *args, **kwargs)

    def log_metrics(self, metrics, step, level="info"):
        super().log_metrics(metrics, step=step)
        if self.log_metric_to_stdout:
            self.log(pretty_repr(metrics), level=level)

    def wandb_summary_update(self, result, finish_wandb=False):
        # ! Finish wandb
        if wandb.run is not None and self.local_rank <= 0:
            wandb.summary.update(result)
        if finish_wandb:
            wandb_finish()

    def save_file_to_wandb(self, file, base_path, policy="now", **kwargs):
        if wandb.run is not None and self.local_rank <= 0:
            wandb.save(file, base_path=base_path, policy=policy, **kwargs)


def wandb_finish(result=None):
    if wandb.run is not None:
        wandb.summary.update(result or {})
        wandb.finish()


def get_cur_time(timezone=None, t_format="%m-%d %H:%M:%S"):
    return datetime.fromtimestamp(int(time.time()), timezone).strftime(t_format)


class timer(ContextDecorator):
    def __init__(self, name=None, log_func=logger.info):
        self.name = name
        self.log_func = log_func
        self.timer = Timer(name=name, logger=None)  # Disable internal logging

    def __enter__(self):
        self.timer.start()
        self.log_func(f"Started {self.name} at {get_cur_time()}")
        return self

    def __exit__(self, *exc):
        elapsed_time = self.timer.stop()
        formatted_time = format_timespan(elapsed_time)
        self.log_func(
            f"Finished {self.name} at {get_cur_time()}, running time = {formatted_time}."
        )
        return False

    def __call__(self, func):
        self.name = self.name or func.__name__

        @wraps(func)
        def decorator(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorator
