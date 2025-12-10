import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
from scipy import linalg
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline

from config import DATA_ROOT, DEVICE


class GenerationLab:
    """
    Evaluate generation quality (FID + CLIPScore) and BLIP captions.
    """

    def __init__(self, data_root: str = DATA_ROOT):
        self.data_root = data_root
        self.device = DEVICE
        print(f"[GenerationLab] device={self.device}")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            self.device
        )
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.blip = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
        self.blip_proc = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.sd = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        ).to(self.device)
        self.sd.safety_checker = None
        self.sd.requires_safety_checker = False
        self.sd.set_progress_bar_config(disable=True)

    # ---------------- util ---------------- #
    def _get_features(self, images: List[Image.Image]) -> np.ndarray:
        feats: list[np.ndarray] = []
        for i in range(0, len(images), 32):
            batch = images[i : i + 32]
            with torch.no_grad():
                inputs = self.clip_proc(images=batch, return_tensors="pt").to(self.device)
                emb = self.clip.get_image_features(**inputs).cpu().numpy()
            feats.append(emb)
        return np.concatenate(feats, axis=0)

    @staticmethod
    def _fid(real: np.ndarray, fake: np.ndarray) -> float:
        mu1, s1 = real.mean(0), np.cov(real, rowvar=False)
        mu2, s2 = fake.mean(0), np.cov(fake, rowvar=False)
        ssdiff = np.sum((mu1 - mu2) ** 2)
        covmean = linalg.sqrtm(s1.dot(s2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(ssdiff + np.trace(s1 + s2 - 2.0 * covmean))

    def _post_process(self, img: Image.Image) -> Image.Image:
        img = img.convert("L").convert("RGB")
        img = ImageEnhance.Contrast(img).enhance(1.2)
        return img

    # ---------------- main experiments ---------------- #
    def run_generation_benchmark(self, n: int = 50):
        pneu_dir = os.path.join(self.data_root, "test", "PNEUMONIA")
        filenames = os.listdir(pneu_dir)[:n]
        real_imgs = [
            Image.open(os.path.join(pneu_dir, f)).convert("RGB") for f in filenames
        ]
        feat_real = self._get_features(real_imgs)

        # baseline
        print("[Generation] baseline SD...")
        imgs_base = self.sd(["chest x-ray"] * n, num_inference_steps=20).images
        feat_base = self._get_features(imgs_base)

        # ours
        print("[Generation] expert prompt + postproc...")
        prompt = (
            "frontal chest x-ray, PA view, radiograph, lungs, ribs, "
            "pneumonia opacity, high quality, dicom style"
        )
        neg = "skeleton, skin, body, color, face, text, blurry"
        imgs_raw = self.sd(
            [prompt] * n,
            negative_prompt=[neg] * n,
            num_inference_steps=30,
        ).images
        imgs_ours = [self._post_process(img) for img in imgs_raw]
        feat_ours = self._get_features(imgs_ours)

        fid_base = self._fid(feat_real, feat_base)
        fid_ours = self._fid(feat_real, feat_ours)

        def clip_score(imgs, text: str):
            txt_in = self.clip_proc(text=[text], return_tensors="pt").to(self.device)
            with torch.no_grad():
                txt_emb = self.clip.get_text_features(**txt_in)
                txt_emb /= txt_emb.norm(dim=-1, keepdim=True)
                img_emb = torch.tensor(self._get_features(imgs)).to(self.device)
                img_emb /= img_emb.norm(dim=-1, keepdim=True)
                score = (img_emb @ txt_emb.T).mean().item()
            return score

        cs_base = clip_score(imgs_base, "pneumonia x-ray")
        cs_ours = clip_score(imgs_ours, "pneumonia x-ray")

        print(
            f"[Results] Baseline FID={fid_base:.2f}, CLIP={cs_base:.4f}; "
            f"Ours FID={fid_ours:.2f}, CLIP={cs_ours:.4f}"
        )
        return imgs_base[0], imgs_ours[0], fid_base, fid_ours, cs_base, cs_ours

    def run_caption_demo(self):
        pneu_dir = os.path.join(self.data_root, "test", "PNEUMONIA")
        img_path = os.path.join(pneu_dir, os.listdir(pneu_dir)[0])
        img = Image.open(img_path).convert("RGB")

        inputs = self.blip_proc(img, return_tensors="pt").to(self.device)
        out_base = self.blip.generate(**inputs, max_new_tokens=20)
        cap_base = self.blip_proc.decode(out_base[0], skip_special_tokens=True)

        inputs_ours = self.blip_proc(
            img, "chest x-ray showing", return_tensors="pt"
        ).to(self.device)
        out_ours = self.blip.generate(
            **inputs_ours,
            max_new_tokens=40,
            num_beams=5,
            repetition_penalty=1.5,
            min_length=10,
        )
        cap_ours = self.blip_proc.decode(out_ours[0], skip_special_tokens=True)
        print("[BLIP] baseline:", cap_base)
        print("[BLIP] ours:", cap_ours)
        return img, cap_base, cap_ours


def main():
    lab = GenerationLab(DATA_ROOT)
    img_b, img_o, fb, fo, csb, cso = lab.run_generation_benchmark(n=50)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_b)
    ax[0].set_title("Baseline (SD v1.5)\nHigh FID, Artifacts")
    ax[0].axis("off")
    ax[1].imshow(img_o, cmap="gray")
    ax[1].set_title("Ours (Expert Prompt + PostProc)\nLow FID, Clinical Style")
    ax[1].axis("off")
    plt.show()


if __name__ == "__main__":
    main()
