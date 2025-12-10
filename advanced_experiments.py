import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
from scipy import linalg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from config import DATA_PATH as ADV_DATA_PATH, DEVICE
from baselines_cnn import BaselineExperiment
from clip_backbone import load_clip
from diffusers import StableDiffusionPipeline


class AdvancedExperiments:
    """
    Extra experiments: synthetic augmentation curve, prompt ablation (FID + CLIPScore),
    and calibration (reliability diagram).
    """

    def __init__(self, p1_experiment: BaselineExperiment):
        self.full_data = p1_experiment.full_data
        self.train_indices = p1_experiment.train_indices
        self.device = DEVICE
        self.clip_model, self.clip_processor = load_clip(self.device)
        self.sd_pipe: StableDiffusionPipeline | None = None

    def _load_sd(self):
        if self.sd_pipe is None:
            self.sd_pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
            ).to(self.device)
            self.sd_pipe.safety_checker = None
            self.sd_pipe.set_progre
