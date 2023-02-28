from torch.nn import CrossEntropyLoss

from lib.Experiment import IExperiment
from lib.train.losses import CutMix, CrossEntropy


def patch(exp: IExperiment):
    exp.train_loss = CutMix(loss=CrossEntropyLoss(), alpha=1.0)
    exp.val_loss = CrossEntropy()

