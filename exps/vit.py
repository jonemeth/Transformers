from lib.Experiment import IExperiment
from lib.models.VisionTransformer import VisionTransformer


def patch(exp: IExperiment):
    exp.model = VisionTransformer(num_classes=10, in_shape=(3, 32, 32), patch_size=2, embedding_dims=256, depth=8,
                                  num_heads=8, conv_patches=False, relative_encoding=False, avg_pool=False)
