import pytorch_lightning as pl
import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
from graphany.utils import logger, timer
from graphany.utils.experiment import init_experiment
from graphany.data import GraphDataset, CombinedDataset
from graphany.model import GraphAny

import torch
import hydra
from omegaconf import DictConfig
import wandb
import numpy as np
import torchmetrics
from rich.pretty import pretty_repr

mean = lambda input: np.round(np.mean(input).item(), 2)


class InductiveLabelPred(pl.LightningModule):
    def __init__(self, cfg, combined_dataset, checkpoint=None):
        super().__init__()
        self.cfg = cfg
        if checkpoint:
            # Initialize from previous checkpoint using previous graphany config
            ckpt = torch.load(checkpoint, map_location="cpu")
            logger.critical(f"Loaded checkpoint at {checkpoint}")
            self.gnn_model = GraphAny(**ckpt["graph_any_config"])
            self.load_state_dict(ckpt["state_dict"])
        else:
            self.gnn_model = GraphAny(**cfg.graph_any)
        self.combined_dataset = combined_dataset
        self.attn_dict, self.loss_dict, self.res_dict = {}, {}, {}
        # Initialize accuracy metrics for validation and testing
        self.metrics = {}
        held_out_datasets = list(
            set(self.cfg._all_datasets) - set(self.cfg._trans_datasets)
        )  # 27 datasets in total
        self.heldout_metrics = [
            f"{setting}/{d.lower()[:4]}_{split}_acc"
            for split in ["val", "test"]
            for d in held_out_datasets
            for setting in ["trans", "ind"]
        ]
        for split in ("val", "test"):
            self.metrics[split] = {
                k: torchmetrics.Accuracy(task="multiclass", num_classes=v.num_class)
                for k, v in combined_dataset.eval_ds_dict.items()
            }

        self.criterion = torch.nn.CrossEntropyLoss()

    def on_train_end(self):
        checkpoint_path = f"{self.cfg.dirs.output}{self.cfg.dataset}_val_acc={self.res_dict['val_acc']}.pt"
        self.save_checkpoint(checkpoint_path)

    def save_checkpoint(self, file_path):
        checkpoint = {
            "state_dict": self.state_dict(),
            "optimizer_state_dict": [
                opt.state_dict() for opt in self.trainer.optimizers
            ],
            "graph_any_config": self.cfg.graph_any,
        }
        torch.save(checkpoint, file_path)
        logger.critical(f"Checkpoint saved to {file_path}")

    def get_metric_name(self, ds_name, split):
        if ds_name in self.cfg.train_datasets:
            return f"trans/{ds_name.lower()[:4]}_{split}_acc"
        else:
            return f"ind/{ds_name.lower()[:4]}_{split}_acc"

    def configure_optimizers(self):
        num_devices = self.cfg.gpus if self.cfg.gpus > 0 else 1

        # start with all the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.info(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        logger.info(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        if self.cfg.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        else:  # AdamW
            optimizer = torch.optim.AdamW(
                optim_groups,
                lr=self.cfg.lr * num_devices,
                weight_decay=self.cfg.weight_decay,
            )
        return optimizer

    def on_fit_start(self):
        super().on_fit_start()
        # move all datasets to the correct GPU device
        print(f"moving train and eval datasets to {self.device}")
        self.combined_dataset.to(self.device)
        self.move_metrics_to_device()

    def move_metrics_to_device(self):
        for metrics_dict in self.metrics.values():
            for metric in metrics_dict.values():
                metric.to(self.device)
        # Example for a direct metric attribute
        if hasattr(self, "accuracy"):
            self.accuracy.to(self.device)

    def predict(self, ds, nodes, input, is_training=False):
        # Use preprocessed distance during evaluation
        dist = ds.dist if not is_training else None
        dist = dist.to(nodes.device)[nodes] if dist is not None else dist
        preds, attn = self.gnn_model(
            {c: chn_pred[nodes] for c, chn_pred in input.items()}, dist=dist
        )
        self.attn_dict.update(
            {
                f"Attention/{ds.name}-{c}": v
                for c, v in zip(self.cfg.feat_channels, attn)
            }
        )
        return preds

    def training_step(self, batch, batch_idx):
        """
        batch is in form {ds_name1 : batch_indices1, ds_name2 : batch_indices2, ...}
        """
        loss = {}
        for ds_name, batch_nodes in batch.items():
            ds = self.combined_dataset.train_ds_dict[ds_name]
            train_target_idx = batch_nodes
            # Batch nodes are not visible to avoid trivial solution and overfitting
            visible_nodes = list(
                set(ds.train_indices.tolist()) - set(batch_nodes.tolist())
            )
            ref_nodes = torch.tensor(visible_nodes, dtype=torch.long).to(self.device)
            ds_too_small = len(visible_nodes) < len(batch_nodes)
            if ds_too_small:
                # Visible nodes are too few, add first half of the batch to visible nodes
                ref_nodes = torch.cat((ref_nodes, batch_nodes[: len(batch_nodes) // 2]))

            input = ds.compute_channel_logits(
                ds.features, ref_nodes, sample=True, device=self.device
            )

            preds = self.predict(ds, train_target_idx, input, is_training=True)
            loss[f"loss/{ds_name}_loss"] = self.criterion(
                preds, ds.label[train_target_idx]
            )

        detached_loss = {k: v.detach().cpu() for k, v in loss.items()}
        avg_loss = mean(list(detached_loss.values()))
        self.loss_dict.update({"loss/avg_loss": avg_loss, **detached_loss})
        return sum(loss.values())

    def evaluation_step(self, split, batch, batch_idx):
        self.move_metrics_to_device()
        for ds_name, eval_idx in batch.items():
            if eval_idx is None:  # Skip if dataset is already evaluated (empty batch)
                continue
            ds = self.combined_dataset.eval_ds_dict[ds_name]
            ds.to(self.device)
            eval_idx.to(self.device)
            # Use unmasked feature for evaluation
            processed_feat = ds.unmasked_pred
            preds = self.predict(
                ds, eval_idx, processed_feat, is_training=False
            ).argmax(-1)
            self.metrics[split][ds_name].update(preds, ds.label[eval_idx])

    def validation_step(self, batch, batch_idx):
        self.evaluation_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self.evaluation_step("test", batch, batch_idx)

    def compute_and_log_metrics(self, split):
        # Compute metrics from collected outputs
        res = {}
        for ds_name, metric in self.metrics[split].items():
            metric_name = self.get_metric_name(ds_name, split)
            accuracy = metric.compute().cpu().numpy()
            res[metric_name] = np.round(accuracy * 100, 2)
            metric.reset()  # Reset metrics for the next epoch

        combined_res = {f"{split}_acc": np.round(sum(res.values()) / len(res), 2)}
        combined_res[f"trans_{split}_acc"] = mean(
            [v for k, v in res.items() if k.startswith("trans")]
        )
        combined_res[f"ind_{split}_acc"] = mean(
            [v for k, v in res.items() if k.startswith("ind")]
        )

        combined_res[f"heldout_{split}_acc"] = mean(
            [v for k, v in res.items() if k in self.heldout_metrics]
        )
        self.log_dict(res, prog_bar=False, logger=True, add_dataloader_idx=False)
        self.log_dict(
            combined_res, prog_bar=True, logger=True, add_dataloader_idx=False
        )
        self.res_dict.update({**res, **combined_res})

    def on_train_epoch_end(self):
        self.log_dict(self.loss_dict, on_epoch=True, prog_bar=True, logger=True)
        if len(self.attn_dict):
            self.log_dict(self.attn_dict, on_epoch=True, prog_bar=False, logger=True)

    def on_validation_epoch_end(self):
        self.compute_and_log_metrics("val")

    def on_test_epoch_end(self):
        self.compute_and_log_metrics("test")


@timer()
@hydra.main(config_path=f"{root}/configs", config_name="main", version_base=None)
def main(cfg: DictConfig):
    cfg, logger = init_experiment(cfg)
    # Define the default step metric for all metrics
    wandb.define_metric("*", step_metric="epoch")
    if torch.cuda.is_available() and cfg.preprocess_device == "gpu":
        preprocess_device = torch.device("cuda")
    else:
        preprocess_device = torch.device("cpu")

    def construct_ds_dict(datasets):
        datasets = [datasets] if isinstance(datasets, str) else datasets
        ds_dict = {
            dataset: GraphDataset(
                cfg,
                dataset,
                cfg.dirs.data_cache,
                cfg.train_batch_size,
                cfg.val_test_batch_size,
                cfg.n_hops,
                preprocess_device,
            )
            for dataset in datasets
        }
        return ds_dict

    train_ds_dict = construct_ds_dict(cfg.train_datasets)
    eval_ds_dict = construct_ds_dict(cfg.eval_datasets)

    combined_dataset = CombinedDataset(train_ds_dict, eval_ds_dict, cfg)

    model = InductiveLabelPred(cfg, combined_dataset, cfg.get("prev_ckpt"))
    # Set up the checkpoint callback to save only at the end of training
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=cfg.dirs.output,  # specify where to save
        filename="final_checkpoint.pt",  # set a filename
        save_top_k=0,  # do not save based on metric, just save last
        save_last=True,  # ensures only the last checkpoint is kept
        save_on_train_epoch_end=True,  # save at the end of training epoch
    )
    trainer = pl.Trainer(
        max_epochs=cfg.total_steps,
        callbacks=[checkpoint_callback],
        limit_train_batches=cfg.limit_train_batches,
        check_val_every_n_epoch=cfg.eval_freq,
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() and cfg.gpus > 0 else "cpu",
        default_root_dir=cfg.dirs.lightning_root,
    )
    dataloaders = {
        "train": combined_dataset.train_dataloader(),
        "val": combined_dataset.val_dataloader(),
        "test": combined_dataset.test_dataloader(),
    }
    if cfg.total_steps > 0:
        trainer.fit(
            model,
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["val"],
        )
    trainer.validate(model, dataloaders=dataloaders["val"])
    trainer.test(model, dataloaders=dataloaders["test"])
    final_results = model.res_dict
    logger.critical(pretty_repr(final_results))
    logger.wandb_summary_update(final_results, finish_wandb=True)


if __name__ == "__main__":
    main()
