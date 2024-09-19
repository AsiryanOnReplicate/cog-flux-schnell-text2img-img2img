# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
from PIL import Image
from typing import List
from diffusers import (
    FluxPipeline,
    FluxImg2ImgPipeline
)
from torchvision import transforms
from weights import WeightsDownloadCache
from transformers import CLIPImageProcessor

MAX_IMAGE_SIZE = 1440
MODEL_CACHE = "FLUX.1-schnell"
FEATURE_EXTRACTOR = "/src/feature-extractor"
MODEL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-schnell/files.tar"


def download_weights(url, dest, file=False):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    if not file:
        subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    else:
        subprocess.check_call(["pget", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()

        self.weights_cache = WeightsDownloadCache()
        self.last_loaded_lora = None
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)
        
        print("Loading Flux txt2img Pipeline")
        if not os.path.exists("FLUX.1-schnell"):
            download_weights(MODEL_URL, ".")
        self.txt2img_pipe = FluxPipeline.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.bfloat16
        ).to("cuda")

        print("Loading Flux img2img pipeline")
        self.img2img_pipe = FluxImg2ImgPipeline(
            transformer=self.txt2img_pipe.transformer,
            scheduler=self.txt2img_pipe.scheduler,
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2,
        ).to("cuda")
        
        print("setup took: ", time.time() - start)

    def get_image(self, image: str):
        image = Image.open(image).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 2.0 * x - 1.0),
            ]
        )
        img: torch.Tensor = transform(image)
        return img[None, ...]
    
    @staticmethod
    def make_multiple_of_16(n):
        return ((n + 15) // 16) * 16
    
    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Prompt for generated image",
            default="a tiny astronaut hatching from an egg on the moon"
        ),
        image: Path = Input(
            description="Input image for image to image mode. The aspect ratio of your output will match this image",
            default=None,
        ),
        width: int = Input(
            description="Width of output image",
            default=1024
        ),
        height: int = Input(
            description="Height of output image",
            default=1024
        ),
        prompt_strength: float = Input(
            description="Prompt strength (or denoising strength) when using image to image. 1.0 corresponds to full destruction of information in image.",
            ge=0,le=1,default=0.8,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps",
            ge=1,le=12,default=4,
        ),
        seed: int = Input(description="Random seed. Set for reproducible generation", default=None),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="png",
        ),
        output_quality: int = Input(
            description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
            default=100,
            ge=0,
            le=100,
        )
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        max_sequence_length=256
        guidance_scale=0.0

        flux_kwargs = {"width": width, "height": height}
        print(f"Prompt: {prompt}")
        device = self.txt2img_pipe.device

        if image:
            pipe = self.img2img_pipe
            print("img2img mode")
            init_image = self.get_image(image)
            width = init_image.shape[-1]
            height = init_image.shape[-2]
            print(f"Input image size: {width}x{height}")

            # scaling factor
            scale = min(MAX_IMAGE_SIZE / width, MAX_IMAGE_SIZE / height, 1)
            if scale < 1:
                width = int(width * scale)
                height = int(height * scale)
                print(f"Scaling image down to {width}x{height}")

            # Round to nearest multiple of 16
            width = self.make_multiple_of_16(width)
            height = self.make_multiple_of_16(height)
            
            print(f"Input image size set to: {width}x{height}")
            init_image = init_image.to(device)
            init_image = torch.nn.functional.interpolate(init_image, (height, width))
            init_image = init_image.to(torch.bfloat16)
            flux_kwargs["image"] = init_image
            flux_kwargs["strength"] = prompt_strength
        else:
            print("txt2img mode")
            pipe = self.txt2img_pipe

        generator = torch.Generator("cuda").manual_seed(seed)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
            "max_sequence_length": max_sequence_length,
            "output_type": "pil"
        }

        output = pipe(**common_args, **flux_kwargs)

        output_paths = []
        for i, image in enumerate(output.images):
            output_path = f"/tmp/out-{i}.{output_format}"
            if output_format != 'png':
                image.save(output_path, quality=output_quality, optimize=True)
            else:
                image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
