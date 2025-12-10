import os
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torchvision import models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix

from config import DATA_ROOT, DEVICE, set_seed
from data_utils import MedicalDataset, get_image_paths
from clip_backbone import load_clip
from transformers import CLIPProcessor, CLIPModel


class BaselineExperiment:
    """
    CNN baselines vs CLIP + Linear probe on the full dataset (k=500).
    """

    def __init__(self, data_root: str = DATA_ROOT):
        set_seed()
        self.data_root = data_root
        self.results = []
        self.predictions = {}
        self.device = DEVICE

        train_paths, train_labels = get_image_paths(data_root, "train")
        test_paths, test_labels = get_image_paths(data_root, "test")
        self.full_data = {
            "train": {"paths": np.array(train_paths), "labels": np.array(train_labels)},
            "test": {"paths": np.array(test_paths), "labels": np.array(test_labels)},
        }
        self.train_indices = self._get_fixed_subset_indices(
            self.full_data["train"]["labels"], k_shot=500
        )

    @staticmethod
    def _get_fixed_subset_indices(labels, k_shot: int):
        pos = np.where(labels == 1)[0]
        neg = np.where(labels == 0)[0]
        n = k_shot // 2
        s_pos = np.random.choice(pos, min(len(pos), n), replace=False)
        s_neg = np.random.choice(neg, min(len(neg), n), replace=False)
        return np.concatenate([s_pos, s_neg])

    def _evaluate_metrics(
        self, y_true, y_pred, y_prob, model_name: str, pretrain_status: str, params: int
    ):
        auc_score = roc_auc_score(y_true, y_prob)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        key = f"{model_name} ({pretrain_status})"
        self.predictions[key] = {"y_true": y_true, "y_prob": y_prob}
        return {
            "Model": model_name,
            "Pretrain": pretrain_status,
            "Trainable Params": f"{params/1e6:.2f}M",
            "AUC": f"{auc_score:.4f}",
            "Sensitivity": f"{sensitivity:.2%}",
            "Specificity": f"{specificity:.2%}",
        }

    def run_cnn_training(self, model_name="resnet18", pretrained=False, epochs=10):
        print(f">>> Running {model_name} (pretrained={pretrained})")
        paths = self.full_data["train"]["paths"][self.train_indices]
        labels = self.full_data["train"]["labels"][self.train_indices]
        from torchvision import transforms

        transform = MedicalDataset([], [], None).transform
        train_ds = MedicalDataset(paths, labels, transform)
        test_ds = MedicalDataset(
            self.full_data["test"]["paths"],
            self.full_data["test"]["labels"],
            transform,
        )
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

        if model_name == "resnet18":
            weights = (
                models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
            model = models.resnet18(weights=weights)
            model.fc = torch.nn.Linear(model.fc.in_features, 2)
        else:
            raise ValueError("Only resnet18 implemented here.")
        model = model.to(self.device)
        params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        for _ in range(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), torch.tensor(y).to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

        model.eval()
        all_probs, all_preds = [], []
        with torch.no_grad():
            for x, _ in test_loader:
                logits = model(x.to(self.device))
                all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

        res = self._evaluate_metrics(
            self.full_data["test"]["labels"],
            np.array(all_preds),
            np.array(all_probs),
            model_name,
            "ImageNet" if pretrained else "False",
            params_count,
        )
        self.results.append(res)

    def run_clip_linear_probe(self):
        print(">>> Running CLIP + Linear probe...")
        clip_model, processor = load_clip(self.device)

        def get_feats(paths):
            feats = []
            bs = 64
            for i in range(0, len(paths), bs):
                imgs = [Image.open(p).convert("RGB") for p in paths[i : i + bs]]
                with torch.no_grad():
                    inputs = processor(images=imgs, return_tensors="pt").to(self.device)
                    emb = clip_model.get_image_features(**inputs).cpu().numpy()
                feats.append(emb)
            return np.concatenate(feats, axis=0)

        train_x = get_feats(self.full_data["train"]["paths"][self.train_indices])
        train_y = self.full_data["train"]["labels"][self.train_indices]
        test_x = get_feats(self.full_data["test"]["paths"])
        test_y = self.full_data["test"]["labels"]

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(train_x, train_y)
        probs = clf.predict_proba(test_x)[:, 1]
        preds = clf.predict(test_x)

        params = clf.coef_.size + clf.intercept_.size
        res = self._evaluate_metrics(
            test_y, preds, probs, "CLIP + Linear Probe", "CLIP (Frozen)", params
        )
        self.results.append(res)

    def sh
