from abc import ABC

from lib.Experiment import IExperiment
from lib.models.SwinTransformer import SwinTransformer


def patch(exp: IExperiment) -> IExperiment:

    exp.model = SwinTransformer(num_classes=10, in_shape=(3, 32, 32), patch_size=2, embedding_dims=256, depths=[4, 2, 2], window_sizes=[4, 4, 4], head_dims=32)

