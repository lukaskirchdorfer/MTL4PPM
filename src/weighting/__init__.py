from src.weighting.abstract_weighting import AbsWeighting
from src.weighting.EW import EW
from src.weighting.GradNorm import GradNorm
# from src.weighting.MGDA import MGDA
from src.weighting.UW import UW
from src.weighting.DWA import DWA
from src.weighting.GLS import GLS
from src.weighting.GradDrop import GradDrop
from src.weighting.PCGrad import PCGrad
# from src.weighting.GradVac import GradVac
from src.weighting.IMTL import IMTL
from src.weighting.CAGrad import CAGrad
from src.weighting.Nash_MTL import Nash_MTL
from src.weighting.RLW import RLW
from src.weighting.UW_SO import UW_SO
from src.weighting.UW_O import UW_O
from src.weighting.Scalarization import Scalarization

__all__ = [
    "AbsWeighting",
    "EW",
    "GradNorm",
    "MGDA",
    "UW",
    "DWA",
    "GLS",
    "GradDrop",
    "PCGrad",
    "GradVac",
    "IMTL",
    "CAGrad",
    "Nash_MTL",
    "RLW",
    "UW_SO",
    "UW_O",
    "Scalarization",
]