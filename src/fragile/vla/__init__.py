"""TopoEncoder x SmolVLA experiment pipeline."""

from .config import VLAConfig
from .create_latent_dataset import main as create_latent_dataset_cli
from .extract_features import VLAFeatureDataset, extract_smolvla_features
from .train import train_vla
from .train_joint import train_joint
from .train_unsupervised import train_unsupervised
from .covariant_world_model import GeometricWorldModel
from .dashboard import create_app

__all__ = [
    "VLAConfig",
    "VLAFeatureDataset",
    "create_latent_dataset_cli",
    "extract_smolvla_features",
    "GeometricWorldModel",
    "create_app",
    "train_vla",
    "train_joint",
    "train_unsupervised",
]
