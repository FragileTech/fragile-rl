"""TopoEncoder x SmolVLA experiment pipeline."""

from .config import VLAConfig
from .covariant_world_model import GeometricWorldModel
from .create_latent_dataset import main as create_latent_dataset_cli
from .dashboard import create_app
from .extract_features import extract_smolvla_features, VLAFeatureDataset
from .train_joint import train_joint
from .train_phase_1 import train_phase_1
from .train_unsupervised import train_unsupervised


__all__ = [
    "GeometricWorldModel",
    "VLAConfig",
    "VLAFeatureDataset",
    "create_app",
    "create_latent_dataset_cli",
    "extract_smolvla_features",
    "train_joint",
    "train_phase_1",
    "train_unsupervised",
]
