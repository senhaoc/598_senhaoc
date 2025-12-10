import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

from config import DATA_ROOT, DEVICE, set_seed
from clip_backbone import load_clip
from data_utils import get_image_paths


class MedicalExperimentLab:
    """
    Zero-shot CLIP vs. CLIP + Linear probe on the Kaggle pneumonia dataset.
    """

    def __init__(self, data_root: str = DATA_ROOT):
        set_seed()
        self.device = DEVICE
        self.data_root = data_root
        print(f"[MedicalExperimentLab] Device: {self.device}")
        self.model, self.processor = load_clip(self.device)

    # ---------------- Core utilities ---------------- #
    def extract_features(self, image_paths: List[str]) -> np.ndarray:
        feats: list[np.ndarray] = []
        batch_size = 32
        print(f"[extract_features] {len(image_paths)} images...")
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            with torch.no_grad():
                inputs = self.processor(images=images, return_tensors="pt").to(
                    self.device
                )
                emb = self.model.get_image_features(**inputs).cpu().numpy()
            feats.append(emb)
        return np.concatenate(feats, axis=0)

    def run_zero_shot(
        self,
        image_paths: List[str],
        labels: np.ndarray,
        prompt_template: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        class_names = ["normal", "pneumonia"]
        prompts = [prompt_template.format(c) for c in class_names]
        preds, probs = [], []
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            inputs = self.processor(
                text=prompts, images=image, return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits_per_image
                prob = logits.softmax(dim=1).cpu().numpy()[0]
            preds.append(prob.argmax())
            probs.append(prob[1])
        return np.array(preds), np.array(probs)

    def run_linear_probe(
        self,
        train_paths: List[str],
        train_labels: np.ndarray,
        test_features: np.ndarray,
    ):
        print("[Linear probe] Extracting train features...")
        train_features = self.extract_features(train_paths)
        clf = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
        clf.fit(train_features, train_labels)
        preds = clf.predict(test_features)
        probs = clf.predict_proba(test_features)[:, 1]
        return preds, probs, train_features

    @staticmethod
    def evaluate(y_true, y_pred, y_prob, name: str):
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true, y_pred, target_names=["Normal", "Pneumonia"], output_dict=True
        )
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        return {
            "name": name,
            "accuracy": acc,
            "f1_pneumonia": report["Pneumonia"]["f1-score"],
            "auc": roc_auc,
            "cm": cm,
            "fpr": fpr,
            "tpr": tpr,
        }

    # ---------------- Plotting ---------------- #
    @staticmethod
    def plot_comparison(results):
        plt.figure(figsize=(15, 6))
        # ROC
        plt.subplot(1, 2, 1)
        for res in results:
            plt.plot(res["fpr"], res["tpr"], lw=2, label=f"{res['name']} (AUC={res['auc']:.3f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        # Confusion matrix for last result (ours)
        best = results[-1]
        plt.subplot(1, 2, 2)
        sns.heatmap(
            best["cm"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Pneumonia"],
            yticklabels=["Normal", "Pneumonia"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("True label")
        plt.title(f"Confusion Matrix ({best['name']})")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_tsne(features: np.ndarray, labels: np.ndarray, title: str):
        print("[t-SNE] computing...")
        tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
        emb_2d = tsne.fit_transform(features)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="coolwarm", s=10)
        plt.legend(handles=scatter.legend_elements()[0], labels=["Normal", "Pneumonia"])
        plt.axis("off")
        plt.title(title)
        plt.show()


def main():
    lab = MedicalExperimentLab(DATA_ROOT)
    test_paths, test_labels = get_image_paths(DATA_ROOT, "test")
    train_paths, train_labels = get_image_paths(DATA_ROOT, "train", max_samples=200)
    print(f"[Data] Train={len(train_labels)}, Test={len(test_labels)}")

    results = []

    preds_v1, probs_v1 = lab.run_zero_shot(test_paths, test_labels, "a photo of a {}")
    results.append(lab.evaluate(test_labels, preds_v1, probs_v1, "Zero-shot (naive)"))

    preds_v2, probs_v2 = lab.run_zero_shot(
        test_paths, test_labels, "chest x-ray findings showing {}"
    )
    results.append(lab.evaluate(test_labels, preds_v2, probs_v2, "Zero-shot (medical)"))

    test_features = lab.extract_features(test_paths)
    preds_ours, probs_ours, train_feats = lab.run_linear_probe(
        train_paths, train_labels, test_features
    )
    results.append(lab.evaluate(test_labels, preds_ours, probs_ours, "Ours (linear probe)"))

    print("\nMethod                        | Acc   | F1(Pn) | AUC")
    print("-" * 60)
    for r in results:
        print(
            f"{r['name']:<28} | {r['accuracy']:.2%} | {r['f1_pneumonia']:.4f} | {r['auc']:.4f}"
        )

    lab.plot_comparison(results)
    lab.visualize_tsne(test_features, test_labels, "CLIP Image Embeddings (Test Set)")


if __name__ == "__main__":
    main()
