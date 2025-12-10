import os
import time
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt

from config import DATA_ROOT, DEVICE
from clip_backbone import load_clip
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline


class IntegratedMedicalAI:
    """
    End-to-end pipeline: CLIP perception + logistic regression,
    BLIP reporting, Stable Diffusion for augmentation.
    """

    def __init__(self, data_root: str = DATA_ROOT):
        self.data_root = data_root
        self.device = DEVICE
        print(f"[IntegratedMedicalAI] device={self.device}")

        # perception
        self.clip_model, self.clip_processor = load_clip(self.device)
        # reporting
        self.blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
        # generation
        self.sd_pipe: StableDiffusionPipeline | None = None

        self.classifier = LogisticRegression(max_iter=1000, solver="liblinear", C=1.0)
        self.is_trained = False

        self.train_features = None
        self.train_labels = None
        self.test_features = None
        self.test_labels = None
        self.test_paths = None

    # ---------------- training ---------------- #
    def _load_split(self, split: str, sample_size: int | None = None):
        paths, labels = [], []
        label_map = {"NORMAL": 0, "PNEUMONIA": 1}
        for name, lbl in label_map.items():
            d = os.path.join(self.data_root, split, name)
            if not os.path.exists(d):
                continue
            fs = [
                os.path.join(d, f)
                for f in os.listdir(d)
                if f.lower().endswith(("jpeg", "jpg", "png"))
            ]
            if split == "train" and sample_size is not None:
                fs = fs[:sample_size]
            paths.extend(fs)
            labels.extend([lbl] * len(fs))
        return np.array(paths), np.array(labels)

    def _extract_clip_batch(self, paths):
        feats = []
        bs = 64
        for i in tqdm(range(0, len(paths), bs), desc="Feature Extraction"):
            batch = [Image.open(p).convert("RGB") for p in paths[i : i + bs]]
            with torch.no_grad():
                inputs = self.clip_processor(images=batch, return_tensors="pt").to(
                    self.device
                )
                emb = self.clip_model.get_image_features(**inputs).cpu().numpy()
            feats.append(emb)
        return np.concatenate(feats, axis=0)

    def fit_system(self, sample_size: int = 500):
        print(f"[fit_system] few-shot={sample_size}")
        train_p, train_y = self._load_split("train", sample_size)
        test_p, test_y = self._load_split("test")
        self.test_paths = test_p

        self.train_features = self._extract_clip_batch(train_p)
        self.train_labels = train_y
        self.test_features = self._extract_clip_batch(test_p)
        self.test_labels = test_y

        self.classifier.fit(self.train_features, self.train_labels)
        self.is_trained = True
        acc = self.classifier.score(self.train_features, self.train_labels)
        print(f"[fit_system] train accuracy={acc:.2%}")

    # ---------------- inference ---------------- #
    def analyze_patient_case(self, image_path: str):
        if not self.is_trained:
            raise RuntimeError("System not trained.")

        start = time.time()
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            inputs = self.clip_processor(images=image, return_tensors="pt").to(
                self.device
            )
            feat = self.clip_model.get_image_features(**inputs).cpu().numpy()
        prob = self.classifier.predict_proba(feat)[0]
        pred_idx = int(prob.argmax())
        diagnosis = "PNEUMONIA" if pred_idx == 1 else "NORMAL"
        confidence = float(prob[pred_idx])

        if diagnosis == "PNEUMONIA":
            prompt = "chest x-ray showing signs of"
        else:
            prompt = "chest x-ray of"

        inputs_blip = self.blip_processor(image, prompt, return_tensors="pt").to(
            self.device
        )
        out_blip = self.blip_model.generate(**inputs_blip, max_new_tokens=40)
        report = self.blip_processor.decode(out_blip[0], skip_special_tokens=True)

        latency = time.time() - start
        return {"diagnosis": diagnosis, "confidence": confidence, "report": report, "latency": latency}

    def evaluate_system(self):
        preds = self.classifier.predict(self.test_features)
        probs = self.classifier.predict_proba(self.test_features)[:, 1]
        acc = accuracy_score(self.test_labels, preds)
        recall = recall_score(self.test_labels, preds)
        auc = roc_auc_score(self.test_labels, probs)
        tn, fp, fn, tp = confusion_matrix(self.test_labels, preds).ravel()
        specificity = tn / (tn + fp)

        sample_idx = np.random.choice(len(self.test_paths), size=50, replace=False)
        bleu_scores = []
        for idx in tqdm(sample_idx, desc="Captioning Eval"):
            path = self.test_paths[idx]
            true_label = self.test_labels[idx]
            res = self.analyze_patient_case(path)
            hyp = res["report"].split()
            ref_str = (
                "chest x-ray showing pneumonia opacity"
                if true_label == 1
                else "chest x-ray of normal healthy lungs"
            )
            ref = ref_str.split()
            score = sentence_bleu(
                [ref],
                hyp,
                weights=(0.5, 0.5),
                smoothing_function=SmoothingFunction().method1,
            )
            bleu_scores.append(score)
        avg_bleu = float(np.mean(bleu_scores))

        latencies = [
            self.analyze_patient_case(self.test_paths[i])["latency"]
            for i in range(min(10, len(self.test_paths)))
        ]
        avg_latency = float(np.mean(latencies))

        print("\n========== System Summary ==========")
        print(f"AUC={auc:.4f}, Sensitivity={recall:.2%}, Specificity={specificity:.2%}")
        print(f"BLEU={avg_bleu:.4f}, Latency={avg_latency*1000:.1f} ms, "
              f"Throughput={1/avg_latency:.1f} cases/s")
        print("====================================")
        return {
            "auc": auc,
            "recall": recall,
            "specificity": specificity,
            "bleu": avg_bleu,
            "latency": avg_latency,
        }

    # ---------------- generation for augmentation demo ---------------- #
    def generate_augmentation_sample(self):
        if self.sd_pipe is None:
            self.sd_pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
            ).to(self.device)
            self.sd_pipe.safety_checker = None
            self.sd_pipe.requires_safety_checker = False
            self.sd_pipe.set_progress_bar_config(disable=True)

        prompt = (
            "frontal chest x-ray, PA view, radiograph, lungs, ribs, pneumonia opacity, high quality"
        )
        img = self.sd_pipe(
            prompt, negative_prompt="skeleton, 3d, color", num_inference_steps=25
        ).images[0]
        return img


def main():
    system = IntegratedMedicalAI(DATA_ROOT)
    system.fit_system(sample_size=500)
    system.evaluate_system()

    # one visual example
    pneu_indices = np.where(system.test_labels == 1)[0]
    if len(pneu_indices) > 0:
        idx = int(np.random.choice(pneu_indices))
        path = system.test_paths[idx]
        res = system.analyze_patient_case(path)
        img = Image.open(path).convert("RGB")
        plt.figure(figsize=(6, 4))
        plt.imshow(img, cmap="gray")
        plt.title(
            f"AI Diagnosis: {res['diagnosis']} ({res['confidence']:.2%})\nReport: {res['report']}"
        )
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
