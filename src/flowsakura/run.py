import os
import torch
import numpy as np
from PIL import Image
from diffusers import AutoencoderKLQwenImage

ROOT_DIR = "/home/linlifeng/ONEPIECE/bin/tiles/chengdu/"
MODEL_ID = "Qwen/Qwen-Image"
DEVICE = "cuda:2"
DTYPE = torch.bfloat16

vae = AutoencoderKLQwenImage.from_pretrained(
    MODEL_ID,
    subfolder="vae",
    torch_dtype=DTYPE
).to(DEVICE).eval()

vae.requires_grad_((False))


def load_image(
        path,
        width: int = 512,
        height: int = 512
):
    img = Image.open(path).convert("RGB")
    img = img if img.size == (width, height) else img.resize((width, height))

    x = np.array(img)
    x = torch.from_numpy(x).permute(2, 0, 1).float()

    x = x / 255.0
    x = x * 2.0 - 1.0

    return x


if __name__ == "__main__":
    vector_maps = ["10-1", "10-2", "100-1", "100-2", "1000-1", "1001-1", "1001-2", "1002-2"]

    imgs = [
        load_image(
            path=os.path.join(
                ROOT_DIR,
                f"CD-20140803-{m}.png"
            )
        )
        for m in vector_maps
    ]
    x = torch.stack(imgs, dim=0)
    x = x.unsqueeze(2)

    x = x.to(device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        z_raw = vae.encode(x).latent_dist.mode()

    mean = torch.tensor(
        vae.config.latents_mean,
        device=DEVICE,
        dtype=DTYPE
    ).view(1, 16, 1, 1, 1)
    std = torch.tensor(
        vae.config.latents_std,
        device=DEVICE,
        dtype=DTYPE
    ).view(1, 16, 1, 1, 1)

    z = (z_raw - mean) / std
    z = z * std + mean

    with torch.no_grad():
        reconstructed = vae.decode(
            z=z
        ).sample

    reconstructed = reconstructed[:, :, 0]
    reconstructed = (
        reconstructed.float()
        .clamp(-1, 1)
        .add(1)
        .div(2)
    )

    for i, rimg in enumerate(reconstructed):
        rimg = (
            rimg.permute(1, 2, 0)
            .cpu()
            .numpy()
            * 255
        ).round().astype(np.uint8)

        Image.fromarray(rimg).save(
            os.path.join(
                ROOT_DIR,
                f"CD-20140803-{vector_maps[i]}-reconstructed.png"
            )
        )
