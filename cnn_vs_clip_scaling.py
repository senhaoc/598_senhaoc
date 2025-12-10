import os
import numpy as np
import torch
from tqdm.auto import tqdm
from PIL import Image
from torchvision import transforms, models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import DATA_ROOT, DEVICE, SEED, set_seed
from clip_backbone import load_clip
from data_utils import get_image_paths


class UltimateExperimentLab:
    """
    Compare ResNet baselines vs CLIP+Linear across different k-shot settings
    and produce the main dashboard figure (scaling law, t-SNE, confusion matrix).
    """

    def __init__(self, data_root: str = DATA_ROOT):
        set_seed()
        self.data_root = data_root
        self.device = DEVICE
        print(f"[UltimateExperimentLab] Device: {self.device}")
        self.clip_model, self.clip_processor = load_clip(self.device)
        self.cnn_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )
        self.raw_data = {"train": {}, "test": {}}
        self.clip_features = {"train": None, "test": None}

    def load_data(self):
        for split in ["train", "test"]:
            paths, labels = get_image_paths(self.data_root, split)
            self.raw_data[split] = {"paths": np.array(paths), "labels": np.array(labels)}
            print(f"[Data] {split}: {len(paths)} images")

    def extract_clip_features(self):
        for split in ["train", "test"]:
            feats = []
            paths = self.raw_data[split]["paths"]
            bs = 64
            for i in tqdm(range(0, len(paths), bs), desc=f"CLIP {split}"):
                batch_paths = paths[i : i + bs]
                images = [Image.open(p).convert("RGB") for p in batch_paths]
                with torch.no_grad():
                    inputs = self.clip_processor(images=images, return_tensors="pt").to(
                        self.device
                    )
                    emb = self.clip_model.get_image_features(**inputs).cpu().numpy()
                feats.append(emb)
            self.clip_features[split] = np.concatenate(feats, axis=0)

    @staticmethod
    def _sample_indices(labels, k_shot: int):
        pos = np.where(labels == 1)[0]
        neg = np.where(labels == 0)[0]
        n = k_shot // 2
        s_pos = np.random.choice(pos, min(len(pos), n), replace=False)
        s_neg = np.random.choice(neg, min(len(neg), n), replace=False)
        return np.concatenate([s_pos, s_neg])

    # ---------------- ResNet baseline ---------------- #
    def train_resnet(self, k_shot: int, pretrained: bool, epochs: int = 10) -> float:
        full_paths = self.raw_data["train"]["paths"]
        full_labels = self.raw_data["train"]["labels"]
        idx = self._sample_indices(full_labels, k_shot)
        train_paths = full_paths[idx]
        train_labels = full_labels[idx]

        class SimpleDS(torch.utils.data.Dataset):
            def __init__(self, p, y, t):
                self.p, self.y, self.t = p, y, t

            def __len__(self):
                return len(self.p)

            def __getitem__(self, i):
                return self.t(Image.open(self.p[i]).convert("RGB")), self.y[i]

        train_ds = SimpleDS(train_paths, train_labels, self.cnn_transform)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model = model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        for _ in range(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), torch.tensor(y).to(self.device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

        # Evaluate AUC on full test set
        test_paths = self.raw_data["test"]["paths"]
        test_labels = self.raw_data["test"]["labels"]
        test_ds = SimpleDS(test_paths, test_labels, self.cnn_transform)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64)
        probs = []
        with torch.no_grad():
            model.eval()
            for x, _ in test_loader:
                logits = model(x.to(self.device))
                probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        return roc_auc_score(test_labels, probs)

    # ---------------- CLIP + linear ---------------- #
    def train_ours(self, k_shot: int):
        X_all = self.clip_features["train"]
        y_all = self.raw_data["train"]["labels"]
        idx = self._sample_indices(y_all, k_shot)
        clf = LogisticRegression(max_iter=1000, solver="liblinear", C=1.0)
        clf.fit(X_all[idx], y_all[idx])

        X_test = self.clip_features["test"]
        y_test = self.raw_data["test"]["labels"]
        probs = clf.predict_proba(X_test)[:, 1]
        preds = clf.predict(X_test)
        auc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        return auc, f1, probs, preds

    # ---------------- Main benchmark ---------------- #
    def run_benchmark(self):
        shots = [20, 50, 100, 200, 500]
        history = {"ResNet (Scratch)": [], "ResNet (Pretrained)": [], "Ours": []}

        for k in tqdm(shots, desc="k-shot benchmark"):
            res_k = {"r_scr": [], "r_pre": [], "ours": []}
            for _ in range(3):
                res_k["r_scr"].append(
                    self.train_resnet(k, pretrained=False, epochs=5)
                )
                res_k["r_pre"].append(
                    self.train_resnet(k, pretrained=True, epochs=5)
                )
                res_k["ours"].append(self.train_ours(k)[0])
            history["ResNet (Scratch)"].append(np.mean(res_k["r_scr"]))
            history["ResNet (Pretrained)"].append(np.mean(res_k["r_pre"]))
            history["Ours"].append(np.mean(res_k["ours"]))

        auc_ours, f1_ours, probs_ours, preds_ours = self.train_ours(k_shot=500)

        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, init="pca", random_state=SEED)
        X_tsne = tsne.fit_transform(self.clip_features["test"])

        return history, shots, X_tsne, probs_ours, preds_ours

    def plot_dashboard(self, history, shots, X_tsne, probs, preds):
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 3)

        # scaling law
        ax1 = fig.add_subplot(gs[0, 0:2])
        for name, res in history.items():
            marker = "o" if "Ours" in name else "x"
            lw = 3 if "Ours" in name else 2
            ax1.plot(shots, res, marker=marker, linewidth=lw, label=name)
        ax1.set_title("Data Efficiency Analysis (Scaling Law)")
        ax1.set_xlabel("Training Samples (N-shot)")
        ax1.set_ylabel("AUC")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # t-SNE
        ax2 = fig.add_subplot(gs[0, 2])
        y_test = self.raw_data["test"]["labels"]
        scatter = ax2.scatter(
            X_tsne[:, 0], X_tsne[:, 1], c=y_test, cmap="coolwarm", s=10
        )
        ax2.legend(handles=scatter.legend_elements()[0], labels=["Normal", "Pneumonia"])
        ax2.set_title("CLIP Feature Space (Test Set)")
        ax2.axis("off")

        # Confusion matrix
        ax3 = fig.add_subplot(gs[1, 0])
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3)
        ax3.set_title("Confusion Matrix (Ours)")
        ax3.set_xlabel("Predicted")
        ax3.set_ylabel("GT")

        plt.tight_layout()
        plt.show()


def main():
    lab = UltimateExperimentLab(DATA_ROOT)
    lab.load_data()
    lab.extract_clip_features()
    history, shots, X_tsne, probs, preds = lab.run_benchmark()
    lab.plot_dashboard(history, shots, X_tsne, probs, preds)


if __name__ == "__main__":
    main()
