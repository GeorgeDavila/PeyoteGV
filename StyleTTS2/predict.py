## ============================= Pre for StyleTTS2 =============================
#import time
#import random
#from collections import OrderedDict
#
#import yaml
#import nltk
#import torch
#import librosa
#import numpy as np
#import torchaudio
#import phonemizer
#from torch import nn
#import torch.nn.functional as F
#from munch import Munch
#from pydub import AudioSegment
#import IPython.display as ipd
#from nltk.tokenize import word_tokenize
#from cog import BasePredictor, Input, Path
#
#from models import *
#from utils import *
#from text_utils import TextCleaner
#from Utils.PLBERT.util import load_plbert
#from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
#
#textclenaer = TextCleaner()
#nltk.download("punkt")
#
#def load_model(config_path, ckpt_path):
#    config = yaml.safe_load(open(config_path))
#
#    # Load pretrained ASR model, F0 and BERT models
#    plbert = load_plbert(config.get('PLBERT_dir', False))
#    pitch_extractor = load_F0_models(config.get("F0_path", False))
#    text_aligner = load_ASR_models(config.get("ASR_path", False), config.get("ASR_config", False))
#
#    model_params = recursive_munch(config["model_params"])
#    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
#    _ = [model[key].eval() for key in model]
#    _ = [model[key].to("cuda") for key in model]
#
#    params_whole = torch.load(ckpt_path, map_location="cpu")
#    params = params_whole["net"]
#
#    for key in model:
#        if key in params:
#            try:
#                model[key].load_state_dict(params[key])
#            except:
#                state_dict = params[key]
#                new_state_dict = OrderedDict()
#                for k, v in state_dict.items():
#                    name = k[7:]
#                    new_state_dict[name] = v
#                model[key].load_state_dict(new_state_dict, strict=False)
#
#    _ = [model[key].eval() for key in model]
#    return model, model_params

# ============================= pixart predict base: =============================
from styleTTS2Funcs import *
from cog import BasePredictor, Input, Path
import os
import torch
from typing import List
from diffusers import (
    PixArtAlphaPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
)

class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)

MODEL_NAME = "PixArt-alpha/PixArt-XL-2-1024-MS"
MODEL_CACHE = "model-cache"

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

style_list = [
    {
        "name": "None",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
    },
]

def apply_style(style, prompt, negative_prompt):
    if style == "None":
        return prompt, negative_prompt
    else:
        for style_dict in style_list:
            if style_dict["name"] == style:
                return style_dict["prompt"].format(prompt=prompt), style_dict["negative_prompt"]

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        pipe = PixArtAlphaPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            use_safetensors=True,
            cache_dir=MODEL_CACHE
        )
        # speed-up T5
        pipe.text_encoder.to_bettertransformer()
        self.pipe = pipe.to("cuda")

        # =================== StyleTTS2 ===================
        self.device = "cuda"
        self.global_phonemizer = phonemizer.backend.EspeakBackend(
            language='en-us', preserve_punctuation=True, with_stress=True, words_mismatch="ignore"
        )
        self.model, _ = load_model(
            config_path="Models/LJSpeech/config.yml", ckpt_path="Models/LJSpeech/epoch_2nd_00100.pth"
        )
        self.model_ref, self.model_ref_config = load_model(
            config_path="Models/LibriTTS/config.yml", ckpt_path="Models/LibriTTS/epochs_2nd_00020.pth"
        )

        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False
        )
        self.sampler_ref = DiffusionSampler(
            self.model_ref.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
            clamp=False
        )

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Input prompt", default="A small cactus with a happy face in the Sahara desert"),
        negative_prompt: str = Input(description="Negative prompt", default=None),
        style: str = Input(
            description="Image style",
            choices=["None", "Cinematic", "Photographic", "Anime", "Manga", "Digital Art", "Pixel Art", "Fantasy Art", "Neonpunk", "3D Model"],
            default="None",
        ),
        width: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="DPMSolverMultistep",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=100, default=14
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=4.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        self.pipe.scheduler = SCHEDULERS[scheduler].from_config(self.pipe.scheduler.config)

        prompt, negative_prompt = apply_style(style, prompt, negative_prompt)
        print("Prompt:", prompt, " Negative Prompt:", negative_prompt)
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            num_images_per_prompt=num_outputs,
            output_type="pil",
        )


        audio_out_path = "/tmp/out-audio.mp3"

        # higher embedding_scale = adherence to text aka higher emotion
        inferencePlusExport(audio_out_path, prompt, diffusion_stepsParam=5, embedding_scaleParam=1)

        output_paths = []
        output_paths.append(audio_out_path)

        for i, img in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            img.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths