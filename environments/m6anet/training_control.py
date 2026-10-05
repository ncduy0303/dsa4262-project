"""Python 3.8-compatible validation-loss stopping policy, independent of PyTorch."""

import math


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0005, min_epochs=10):
        if patience < 1 or min_delta < 0 or min_epochs < 1:
            raise ValueError("Invalid early-stopping settings")
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.best_loss = None
        self.best_epoch = None
        self.reference_loss = None
        self.bad_epochs = 0

    def step(self, epoch, loss):
        if not math.isfinite(loss):
            raise ValueError("Non-finite validation loss")
        improved = self.best_loss is None or loss < self.best_loss
        if improved:
            self.best_loss, self.best_epoch = loss, epoch
        if self.reference_loss is None or loss < self.reference_loss - self.min_delta:
            self.reference_loss = loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, epoch >= self.min_epochs and self.bad_epochs >= self.patience
