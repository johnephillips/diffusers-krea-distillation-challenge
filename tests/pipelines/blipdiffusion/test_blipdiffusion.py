# coding=utf-8
# Copyright 2023 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import random
import unittest

import numpy as np
import torch

from diffusers import BlipDiffusionPipeline, PNDMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers.models.blip_2.configuration_blip_2 import Blip2Config, Blip2QFormerConfig, Blip2VisionConfig
from src.diffusers.pipelines.blip_diffusion.modeling_blip2 import Blip2QFormerModel
from src.diffusers.pipelines.blip_diffusion.blip_image_processing import BlipImageProcessor
from diffusers.utils import floats_tensor, load_numpy, slow, torch_device
from diffusers.utils.testing_utils import enable_full_determinism, require_torch_gpu
from transformers import CLIPTokenizer
from ..test_pipelines_common import PipelineTesterMixin, assert_mean_pixel_difference
from transformers.models.clip.configuration_clip import CLIPTextConfig
from src.diffusers.pipelines.blip_diffusion.modeling_ctx_clip import ContextCLIPTextModel
from PIL import Image

enable_full_determinism()


class BlipDiffusionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = BlipDiffusionPipeline
    params = [
        "prompt",
        "reference_image",
        "source_subject_category",
        "target_subject_category",
    ]
    batch_params = [
        "prompt",
        "reference_image",
        "source_subject_category",
        "target_subject_category",
    ]
    required_optional_params = [
        "generator",
        "height",
        "width",
        "latents",
        "guidance_scale",
        "num_inference_steps",
        "neg_prompt",
        "guidance_scale",
        "prompt_strength",
        "prompt_reps",
    ]

    def get_dummy_components(self):
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            hidden_size=32,
            intermediate_size=32,
            projection_dim=32,
            num_hidden_layers=1,
            num_attention_heads=1,
            max_position_embeddings=77,
        )
        text_encoder = ContextCLIPTextModel(text_encoder_config)

        vae = AutoencoderKL(
            in_channels=4,
            out_channels=4,
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(64,),
            layers_per_block=1,
            act_fn="silu",
            latent_channels=4,
            norm_num_groups=32,
            sample_size=32,
        )

        blip_vision_config = {
            "hidden_size": 32,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "image_size": 224,
            "patch_size": 14,
            "hidden_act": "quick_gelu",
        }

        blip_qformer_config = {
            "vocab_size": 30522,
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "intermediate_size": 32,
            "max_position_embeddings": 512,
            "cross_attention_frequency": 1,
            "encoder_hidden_size": 32,
        }
        qformer_config = Blip2Config(
            vision_config=blip_vision_config, qformer_config=blip_qformer_config, num_query_tokens=16
        )
        qformer = Blip2QFormerModel(qformer_config)

        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        tokenizer = CLIPTokenizer.from_pretrained("ayushtues/blipdiffusion", subfolder="tokenizer")

        scheduler = PNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            set_alpha_to_one=False,
            skip_prk_steps=True,
        )

        vae.eval()
        qformer.eval()
        text_encoder.eval()

        image_processor = BlipImageProcessor()

        components = {
            "text_encoder": text_encoder,
            "vae": vae,
            "qformer": qformer,
            "unet": unet,
            "tokenizer": tokenizer,
            "scheduler": scheduler,
            "image_processor": image_processor,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        np.random.seed(seed)
        reference_image = np.random.rand(224, 224, 3) * 255
        reference_image = Image.fromarray(reference_image.astype("uint8")).convert("RGBA")

        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "swimming underwater",
            "generator": generator,
            "reference_image": reference_image,
            "source_subject_category": "dog",
            "target_subject_category": "dog",
            "height": 512,
            "width": 512,
            "guidance_scale": 7.5,
            "num_inference_steps": 2,
            "output_type": "np",
        }
        return inputs

    def test_blipdiffusion(self):
        device = "cpu"
        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)

        pipe.set_progress_bar_config(disable=None)

        image = pipe(**self.get_dummy_inputs(device))[0]
        image_slice = image[0, -3:, -3:, 0]

        assert image.shape == (1, 64, 64, 4)

        expected_slice = np.array([0.5250, 0.4738, 0.3700, 0.4632, 0.5203, 0.5275, 0.5552, 0.5766, 0.5742])

        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        ), f" expected_slice {image_slice.flatten()}, but got {image_slice.flatten()}"
