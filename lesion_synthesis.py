import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
from diffusers import AutoPipelineForImage2Image

from config import DATA_ROOT, DEVICE


class TopTierVisualizer:
    """
    Image-to-image diffusion for adding pneumonia-like opacities to a normal X-ray.
    """

    def __init__(self, data_root: str = DATA_ROOT):
        self.data_root = data_root
        self.device = DEVICE
        print(f"[TopTierVisualizer] device={self.device}")
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(self.device)
        self.pipe.safety_checker = None
        self.pipe.requires_safety_checker = False

    def load_real_sample(self):
        normal_dir = os.path.join(self.data_root, "test", "NORMAL")
        fname = os.listdir(normal_dir)[0]
        img_path = os.path.join(normal_dir, fname)
        return Image.open(img_path).convert("RGB").resize((512, 512))

    def generate_pathology(self, base_image: Image.Image):
        prompt = (
            "chest x-ray, pneumonia opacity in right lung, consolidation, "
            "hazy white patches, medical imaging, high contrast"
        )
        neg_prompt = "color, skin, muscle, text, low quality, blurry, deformation"
        result = self.pipe(
            prompt,
            image=base_image,
            negative_prompt=neg_prompt,
            strength=0.45,
            guidance_scale=8.5,
            num_inference_steps=50,
        )
        return result.images[0]

    @staticmethod
    def create_diff_heatmap(real: Image.Image, fake: Image.Image):
        r = np.array(real.convert("L"))
        f = np.array(fake.convert("L"))
        diff = cv2.absdiff(r, f)
        diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return heatmap

    def plot_figure(self, real, fake, heatmap):
        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        axes[0].imshow(real, cmap="gray")
        axes[0].set_title("(a) Input: healthy\n(real X-ray)")
        axes[0].axis("off")
        axes[1].imshow(fake, cmap="gray")
        axes[1].set_title("(b) Output: synthetic\npneumonia")
        axes[1].axis("off")
        axes[2].imshow(fake, cmap="gray")
        axes[2].imshow(heatmap, alpha=0.5)
        axes[2].set_title("(c) Where disease\nwas added")
        axes[2].axis("off")

        crop_x, crop_y, w, h = 50, 50, 200, 200
        real_crop = np.array(real)[crop_y : crop_y + h, crop_x : crop_x + w]
        fake_crop = np.array(fake)[crop_y : crop_y + h, crop_x : crop_x + w]
        combined = np.concatenate([real_crop, fake_crop], axis=1)
        axes[3].imshow(combined)
        axes[3].set_title("(d) Zoom: healthy vs opacity")
        axes[3].axis("off")
        plt.tight_layout()
        plt.show()


def main():
    viz = TopTierVisualizer(DATA_ROOT)
    real_img = viz.load_real_sample()
    fake_img = viz.generate_pathology(real_img)
    heatmap = viz.create_diff_heatmap(real_img, fake_img)
    viz.plot_figure(real_img, fake_img, heatmap)


if __name__ == "__main__":
    main()
