# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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
import unittest

import numpy as np
import torch
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel
from transformers.models.clip.configuration_clip import CLIPTextConfig

from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
from diffusers.utils.testing_utils import enable_full_determinism
from src.diffusers.plus_models.ella import ELLA, ELLAProxyUNet
from src.diffusers.plus_pipelines.ella.pipeline_ella import EllaFixedDiffusionPipeline

from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class EllaDiffusionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = EllaFixedDiffusionPipeline
    params = [
        "prompt",
        "negative_prompt",
    ]
    batch_params = [
        "prompt",
        "negative_prompt",
    ]
    required_optional_params = [
        "generator",
        "height",
        "width",
        "latents",
        "guidance_scale",
        "num_inference_steps",
        "guidance_scale",
    ]

    def get_dummy_components(self):
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            vocab_size=1000,
            hidden_size=16,
            intermediate_size=16,
            projection_dim=16,
            num_hidden_layers=1,
            num_attention_heads=1,
            max_position_embeddings=77,
        )
        text_encoder = CLIPTextModel(text_encoder_config)

        vae = AutoencoderKL(
            in_channels=4,
            out_channels=4,
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(32,),
            layers_per_block=1,
            act_fn="silu",
            latent_channels=4,
            norm_num_groups=16,
            sample_size=16,
        )
        ella = ELLA.from_pretrained('shauray/ELLA_SD15')

        unet = UNet2DConditionModel(
            block_out_channels=(16, 32),
            norm_num_groups=16,
            layers_per_block=1,
            sample_size=16,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=16,
        )
        proxy_unet = ELLAProxyUNet(ella, unet)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        scheduler = PNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            set_alpha_to_one=False,
            skip_prk_steps=True,
        )

        vae.eval()

        components = {
            "text_encoder": text_encoder,
            "vae": vae,
            "unet": proxy_unet,
            "tokenizer": tokenizer,
            "scheduler": scheduler,
            "ELLA": ELLA,
            "safety_checker":None,
            "feature_extractor":None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        np.random.seed(seed)

        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "swimming underwater",
            "negative_prompt": '',
            "generator": generator,
            "height": 32,
            "width": 32,
            "guidance_scale": 7.5,
            "num_inference_steps": 2,
            "output_type": "np",
        }
        return inputs

    def test_elladiffusion(self):
        device = "cpu"
        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)

        pipe.set_progress_bar_config(disable=None)

        image = pipe(**self.get_dummy_inputs(device))[0]
        image_slice = image[0, -3:, -3:, 0]

        assert image.shape == (1, 16, 16, 4)

        expected_slice = np.array([0.7096, 0.5900, 0.6703, 0.4032, 0.7766, 0.3629, 0.5447, 0.4149, 0.8172])

        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        ), f" expected_slice {image_slice.flatten()}, but got {image_slice.flatten()}"
