import os
from typing import Type, Optional
import torch
from utils import logger
import torch.nn as nn
import torch.optim as optim
try:
    import safetensors.torch as st
    _HAVE_SAFETENSORS = True
except Exception:
    st = None
    _HAVE_SAFETENSORS = False


MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def save_torch_model(model: torch.nn.Module, name: str) -> str:
    """Save a PyTorch model with safetensors fallback.

    Args:
        model: PyTorch model to save
        name: base name for the model file (without extension)

    Returns:
        Path to the saved model file
    """
    path = os.path.join(MODEL_DIR, f"{name}.pt")
    safetensors_path = os.path.join(MODEL_DIR, f"{name}.safetensors")

    try:
        if _HAVE_SAFETENSORS:
            st.save_file(model.state_dict(), safetensors_path)
            logger.info("Saved model to %s", safetensors_path)
            return safetensors_path
        else:
            torch.save(model.state_dict(), path)
            logger.info("Saved model to %s", path)
            return path
    except Exception:
        logger.exception("Failed to save model %s", name)
        raise


def load_torch_model(model_class: Type[torch.nn.Module], name: str, device: Optional[torch.device] = None) -> torch.nn.Module:
    """Load a PyTorch model with safetensors support.

    Args:
        model_class: Class of the model to instantiate
        name: base name of the model file (without extension)
        device: torch.device to load the model onto

    Returns:
        Instantiated and loaded model
    """
    path = os.path.join(MODEL_DIR, f"{name}.pt")
    safetensors_path = os.path.join(MODEL_DIR, f"{name}.safetensors")

    model = model_class()
    try:
        if _HAVE_SAFETENSORS and os.path.exists(safetensors_path):
            model.load_state_dict(st.load_file(safetensors_path))
            logger.info("Loaded model from %s", safetensors_path)
        elif os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            logger.info("Loaded model from %s", path)
        else:
            raise FileNotFoundError(f"Model file not found: {name}")
    except Exception:
        logger.exception("Failed to load model %s", name)
        raise

    return model.to(device) if device else model


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int = 16, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, max(latent_dim * 2, 8)),
            nn.ReLU(),
            nn.Linear(max(latent_dim * 2, 8), latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, max(latent_dim * 2, 8)),
            nn.ReLU(),
            nn.Linear(max(latent_dim * 2, 8), input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def save_model_state(model: torch.nn.Module, path: str):
    try:
        torch.save(model.state_dict(), path)
        logger.info("Saved model state to %s", path)
    except Exception:
        logger.exception("Failed to save model state to %s", path)
        raise


def load_model_state_into(model: torch.nn.Module, path: str, device=None):
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        logger.info("Loaded model state from %s", path)
        return model
    except Exception:
        logger.exception("Failed to load model state from %s", path)
        raise
