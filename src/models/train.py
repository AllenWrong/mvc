import wandb
import torch as th
import os
import sys
import time
from torch.utils.data import DataLoader

# ---- path preprocessing ----
# current_path = "....\mvc\src\models\"
parent_dir = os.path.dirname(__file__)
for i in range(2):
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from src import config
from src import helpers
from src.data.load import load_dataset
from src.models import callback
from src.models.build_model import build_model
from src.models import evaluate

# define train log file
sys.stdout = open("../../log.txt", "w")


def train(cfg, net, loader, eval_data, callbacks=tuple()):
    """
    Train the model for one run.
    Args:
        cfg: [src.config.defaults.Experiment] experiment config
        net: model
        loader: [th.utils.data.DataLoader] data loader
        eval_data: [th.utils.data.DataLoader]  data loader for evaluation data
        callbacks: [List] training callbacks

    Returns: None
    """

    # number of batches
    n_batches = len(loader)
    for e in range(1, cfg.n_epochs + 1):
        print(f"-------- epoch: {e} --------")
        start_time = time.time()
        iter_losses = []
        for i, data in enumerate(loader):
            # batch is a list which contains one tensor X
            *batch, _ = data
            try:
                batch_losses = net.train_step(batch, epoch=(e-1), it=i, n_batches=n_batches)
            except Exception as e:
                print(f"Training stopped due to exception: {e}")
                return

            iter_losses.append(helpers.npy(batch_losses))
        logs = evaluate.get_logs(cfg, net, eval_data=eval_data, iter_losses=iter_losses, epoch=e, include_params=True)
        try:
            for cb in callbacks:
                cb.epoch_end(e, logs=logs, net=net)
        except callback.StopTraining as err:
            print(err)
            break
        end_time = time.time()
        used_time = ("%.2fs") % (end_time - start_time)
        print(used_time)


def main():
    """
    Run an experiment.
    """
    experiment_name, cfg = config.get_experiment_config()
    dataset = load_dataset(**cfg.dataset_config.dict())
    loader = DataLoader(dataset, batch_size=int(cfg.batch_size), shuffle=True, num_workers=0,
                                      drop_last=True, pin_memory=False)
    eval_data = evaluate.get_eval_data(dataset, cfg.n_eval_samples, cfg.batch_size)
    experiment_identifier = wandb.util.generate_id()

    run_logs = []
    for run in range(cfg.n_runs):
        net = build_model(cfg.model_config)
        print(net)
        callbacks = (
            callback.Printer(print_confusion_matrix=(cfg.model_config.cm_config.n_clusters <= 100)),
            callback.ModelSaver(cfg=cfg, experiment_name=experiment_name, identifier=experiment_identifier,
                                run=run, epoch_interval=1, best_loss_term=cfg.best_loss_term,
                                checkpoint_interval=cfg.checkpoint_interval),
            callback.EarlyStopping(patience=cfg.patience, best_loss_term=cfg.best_loss_term, epoch_interval=1)
        )
        train(cfg, net, loader, eval_data=eval_data, callbacks=callbacks)
        run_logs.append(evaluate.eval_run(cfg=cfg, cfg_name=experiment_name,
                                          experiment_identifier=experiment_identifier, run=run, net=net,
                                          eval_data=eval_data, callbacks=callbacks, load_best=True))


if __name__ == '__main__':
    main()
