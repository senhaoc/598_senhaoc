import numpy as np
import torch
from tqdm.auto import tqdm
from PIL import Image
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from cnn_vs_clip_scaling import UltimateExperimentLab
from config import DEVICE


class DiscussionLab:
    """
    Extra analyses: robustness to noise, prompt sensitivity and CLIP-based heatmaps.
    """

    def __init__(self, base_lab: UltimateExperimentLab):
        self.lab = base_lab
        self.device = DEVICE
        self.clip_model = base_lab.clip_model
        self.clip_processor = base_lab.clip_processor
        self.cnn_transform = base_lab.cnn_transform

    # ---------------- robustness ---------------- #
    def run_robustness_test(self, noise_levels=None):
        if noise_levels is None:
            noise_levels = [0.0, 0.1, 0.3, 0.5, 0.8]

        X_train = self.lab.clip_features["train"]
        y_train = self.lab.raw_data["train"]["labels"]
        clf = LogisticRegression(max_iter=1000, solver="liblinear", C=1.0)
        clf.fit(X_train, y_train)

        test_paths = self.lab.raw_data["test"]["paths"]
        test_y = self.lab.raw_data["test"]["labels"]

        aucs = []
        for sigma in tqdm(noise_levels, desc="Robustness"):
            noisy_features = []
            bs = 64
            for i in range(0, len(test_paths), bs):
                batch_paths = test_paths[i : i + bs]
                images = []
                for p in batch_paths:
                    img_tensor = self.cnn_transform(
                        Image.open(p).convert("RGB")
                    )  # [C,H,W]
                    noise = torch.randn_like(img_tensor) * sigma
                    noisy_img = torch.clamp(img_tensor + noise, 0, 1)
                    ndarr = noisy_img.permute(1, 2, 0).numpy() * 255
                    images.append(Image.fromarray(ndarr.astype("uint8")))
                with torch.no_grad():
                    inputs = self.clip_processor(
                        images=images, return_tensors="pt"
                    ).to(self.device)
                    emb = self.clip_model.get_image_features(**inputs).cpu().numpy()
                noisy_features.append(emb)
            X_test_noisy = np.concatenate(noisy_features, axis=0)
            probs = clf.predict_proba(X_test_noisy)[:, 1]
            aucs.append(roc_auc_score(test_y, probs))
        return noise_levels, aucs

    # ---------------- prompt analysis ---------------- #
    def run_prompt_analysis(self):
        candidates = {
            "Naive": ["normal", "pneumonia"],
            "Layman": ["healthy lungs", "sick lungs"],
            "Medical (General)": ["chest x-ray normal", "chest x-ray pneumonia"],
            "Medical (Specific)": ["clear lungs", "lung opacity and consolidation"],
            "Irrelevant": ["cat", "dog"],
        }
        results = {}
        X_test = torch.tensor(self.lab.clip_features["test"]).to(self.device)
        X_test /= X_test.norm(dim=-1, keepdim=True)
        y_test = self.lab.raw_data["test"]["labels"]

        for name, texts in candidates.items():
            with torch.no_grad():
                inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True).to(
                    self.device
                )
                text_emb = self.clip_model.get_text_features(**inputs)
                text_emb /= text_emb.norm(dim=-1, keepdim=True)
                sim = (100 * X_test @ text_emb.T).softmax(dim=-1)
                probs = sim[:, 1].cpu().numpy()
            results[name] = roc_auc_score(y_test, probs)
        return results

    # ---------------- heatmap ---------------- #
    def visualize_heatmap(self, img_path: str, target_text: str = "pneumonia lung opacity"):
        img = Image.open(img_path).convert("RGB")
        inputs = self.clip_processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision_out = self.clip_model.vision_model(
                pixel_values=inputs["pixel_values"], output_hidden_states=True
            )
            last_hidden = vision_out.last_hidden_state  # [1, 50, 768]
            patch_embeddings = last_hidden[:, 1:, :]  # drop CLS

            proj_weights = self.clip_model.visual_projection.weight  # [512,768]
            patch_embeddings = patch_embeddings @ proj_weights.t()
            patch_embeddings = patch_embeddings / patch_embeddings.norm(dim=-1, keepdim=True)

            text_in = self.clip_processor(text=[target_text], return_tensors="pt").to(
                self.device
            )
            text_emb = self.clip_model.get_text_features(**text_in)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

            similarity = torch.bmm(patch_embeddings, text_emb.unsqueeze(-1)).squeeze()

        side = int(similarity.shape[0] ** 0.5)  # 7x7
        heatmap = similarity.view(side, side).cpu().numpy()
        heatmap = zoom(heatmap, 224 / side, order=1)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        return img, heatmap

    def plot_discussion(self, noise_data, prompt_data):
        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(1, 3)

        # robustness
        ax1 = fig.add_subplot(gs[0, 0])
        noise_x, noise_y = noise_data
        ax1.plot(noise_x, noise_y, "o-", linewidth=2)
        ax1.set_title("Robustness Stress Test")
        ax1.set_xlabel("Gaussian noise level ($\\sigma$)")
        ax1.set_ylabel("AUC")
        ax1.grid(True, alpha=0.3)

        # prompt
        ax2 = fig.add_subplot(gs[0, 1])
        names = list(prompt_data.keys())
        values = list(prompt_data.values())
        colors = ["gray" if v < 0.6 else "#1f77b4" for v in values]
        sns.barplot(x=values, y=names, palette=colors, ax=ax2, orient="h")
        ax2.set_title("Zero-shot Prompt Sensitivity")
        ax2.set_xlabel("AUC")
        ax2.set_xlim(0.4, 0.9)

        # heatmap example
        ax3 = fig.add_subplot(gs[0, 2])
        pneu_paths = self.lab.raw_data["test"]["paths"][
            self.lab.raw_data["test"]["labels"] == 1
        ]
        img, heatmap = self.visualize_heatmap(pneu_paths[0])
        ax3.imshow(img)
        ax3.imshow(heatmap, cmap="jet", alpha=0.5)
        ax3.set_title("Where does AI look?\n(Text–patch similarity)")
        ax3.axis("off")

        plt.tight_layout()
        plt.show()


def main():
    lab = UltimateExperimentLab(DATA_ROOT)
    lab.load_data()
    lab.extract_clip_features()
    dlab = DiscussionLab(lab)

    noise_res = dlab.run_robustness_test()
    prompt_res = dlab.run_prompt_analysis()
    dlab.plot_discussion(noise_res, prompt_res)


if __name__ == "__main__":
    main()
