"""
Code in this file is partially or fully based on, and in some cases adapted from, the implementation provided in the following repository:https://github.com/eeulig/ldct-invariances
"""

# Metrics and loss-tracking helpers used during training and validation.
# These classes collect values, average them per epoch, log to wandb, and save plots.
import csv
import os

import matplotlib.pyplot as plt
import wandb


# Helper classes for tracking losses and validation metrics during training.

class Losses(object):
    """Track and report losses for training and validation of VAE"""
    def __init__(self, dataloader, losses="px"):
        self.dataloader = dataloader
        self.names = losses if isinstance(losses, list) else [losses]
        self.losses = {phase: {loss: [0.0] for loss in self.names} for phase in dataloader.keys()}

    def push(self, loss, phase, name="px"):
        """Add the latest loss value into the current epoch counter."""
        if isinstance(loss, dict):
            for n, l in loss.items():
                self.losses[phase][n][-1] += l.item()
        else:
            self.losses[phase][name][-1] += loss.item()

    def summarize(self, phase):
        """averaging loss totals over the dataset."""
        for name in self.names:
            self.losses[phase][name][-1] /= len(self.dataloader[phase])

    def reset(self):
        """storage for the next epoch"""
        for phase in self.losses:
            for name in self.names:
                self.losses[phase][name].append(0.0)

    def log(self, savepath, epoch):
        """Save current loss values """
        # Log to wandb only when a run is active. Local CSV logging must not
        # depend on wandb, since training also supports fully local runs.
        if wandb.run is not None:
            wandb.log(
                {
                    "Loss {} ({})".format(name, phase): loss[-1]
                    for phase, losses in self.losses.items()
                    for name, loss in losses.items()
                },
                step=epoch + 1,
            )

        # Log to file
        with open(os.path.join(savepath, "Losses.csv"), "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            if epoch == 0:
                writer.writerow(
                    ["Epoch"] + ["{} {}".format(name, phase) for name in self.names for phase in self.losses]
                )
            writer.writerow([epoch] + [loss[name][-1] for name in self.names for loss in self.losses.values()])

    def plot(self, savepath, y_log=False):
        """Draw loss curves"""
        for name in self.names:
            plt.figure(figsize=(5, 3))
            for phase in self.losses.keys():
                y = self.losses[phase][name]
                x = list(range(len(y)))
                plt.plot(x, y, label=phase)
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                if y_log:
                    plt.yscale("log")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(savepath, "Losses_{}.pdf".format(name)))
            plt.close("all")


class Metrics(object):
    def __init__(self, dataloader, metrics):
        self.dataloader = dataloader
        self.names = metrics if isinstance(metrics, list) else [metrics]
        self.metrics = {metric: [0.0] for metric in self.names}

    def push(self, metric, name="px"):
        """ new validation metric measurement"""
        if isinstance(metric, dict):
            for n, m in metric.items():
                self.metrics[n][-1] += m
        else:
            self.metrics[name][-1] += metric

    def summarize(self):
        """average values."""
        for name in self.names:
            self.metrics[name][-1] /= len(self.dataloader["val"])

    def reset(self):
        """Reset the tracker for the next epoch"""
        for name in self.names:
            self.metrics[name].append(0.0)

    def log(self, savepath, epoch):
        """Log the current metric values"""
        # Log to wandb
        wandb.log({"Metric {}".format(name): metric[-1] for name, metric in self.metrics.items()}, step=epoch + 1)

        # Log to file
        with open(os.path.join(savepath, "Metrics.csv"), "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            if epoch == 0:
                writer.writerow(["Epoch"] + ["{}".format(name) for name in self.names])
            writer.writerow([epoch] + [self.metrics[name][-1] for name in self.names])

    def plot(self, savepath):
        """Plot metric values"""
        for name in self.names:
            plt.figure(figsize=(5, 3))
            y = self.metrics[name]
            x = list(range(len(y)))
            plt.plot(x, y, label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Metric")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(savepath, "Metrics_{}.pdf".format(name)))
            plt.close("all")
