"""
Geo2Vec: Shape- and Distance-Aware Neural Representation of Geospatial Entities

This is to unifiedly represent mobility behaviors via geospatial entities.
e.g., Move = MultiString; Stay = Polygon or Point

Codes refer to https://github.com/chuchen2017/GeoNeuralRepresentation/tree/master
"""


import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Geo2Vec(nn.Module):
    def __init__(
            self,
            n_poly,
    ) -> None:
        super().__init__()

    def forward(
            self
    ):
        pass


class SDFLoss(nn.Module):
    def __init__(
            self,
    ) -> None:
        super().__init__()

    def forward(
            self,
    ) -> None:
        pass


class PositionalEncoder(nn.Module):
    def __init__(
            self, 
    ):
        super().__init__()

    def forward(
            self
    ) -> None:
        pass
