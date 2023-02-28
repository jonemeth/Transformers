from exps.swin import IExperiment
from lib.train.losses import CrossEntropy


def patch(exp: IExperiment):
    exp.train_loss = CrossEntropy()
    exp.val_loss = CrossEntropy()
