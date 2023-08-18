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
import tempfile
import time
import traceback
import unittest

import numpy as np
from PIL import Image
import torch
from huggingface_hub import hf_hub_download
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    FabricPipeline,
    UNet2DConditionModel,
    logging,
)
from diffusers.models.attention_processor import AttnProcessor, LoRAXFormersAttnProcessor
from diffusers.utils import load_numpy, nightly, slow, torch_device
from diffusers.utils.testing_utils import (
    CaptureLogger,
    enable_full_determinism,
    require_torch_2,
    require_torch_gpu,
    run_test_in_subprocess,
)

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineKarrasSchedulerTesterMixin, PipelineLatentTesterMixin, PipelineTesterMixin


enable_full_determinism()


class FabricPipelineFastTests(
    PipelineTesterMixin, unittest.TestCase
):
    pipeline_class = FabricPipeline
    params = TEXT_TO_IMAGE_PARAMS - {'negative_prompt_embeds', 'width', 'prompt_embeds', 'cross_attention_kwargs', 'height','callback', 'callback_steps'}
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS 
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS 
    required_optional_params = PipelineTesterMixin.required_optional_params - {
        "latents",
        "num_images_per_prompt",
        "callback",
        "callback_steps",
    }

    def get_dummy_components(self):
        torch.manual_seed(0)
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
        torch.manual_seed(0)
        scheduler = EulerAncestralDiscreteScheduler()
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        generator = torch.manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "negative_prompt": "lowres, dark, cropped",
            "generator": generator,
            "num_images": 1,
            "num_inference_steps":2,
            "output_type": "np",
        }
        return inputs

    def test_fabric(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        pipe = FabricPipeline(**components)
        pipe = pipe.to(device)

        pipe.set_progress_bar_config(disable=True)

        inputs = self.get_dummy_inputs(device)
        output = pipe(**inputs)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        print(image_slice.flatten())
        assert image.shape == (1, 128, 128, 3)
        expected_slice = np.array([0.46241423, 0.45808375, 0.4768011, 0.48806447, 0.46090087, 0.5161956, 0.52250206, 0.50051796, 0.4663524])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2


    def test_fabric_w_fb(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        pipe = FabricPipeline(**components)
        pipe = pipe.to(device)

        pipe.set_progress_bar_config(disable=True)

        inputs = self.get_dummy_inputs(device)
        inputs["liked"] = [Image.fromarray(np.ones((512,512)))]
        output = pipe(**inputs)
        image = output.images
        image_slice = output.images[0, -3:, -3:, -1]

        assert image.shape == (1, 128, 128, 3)
        print(image_slice)
        expected_slice = np.array([0.77254902, 0.77647059, 0.78431373, 0.8, 0.78823529, 0.79607843, 0.78823529, 0.78823529, 0.78039216])

        assert np.abs(image_slice.flatten()/128 - expected_slice).max() < 1e-2


@require_torch_gpu
@slow
class FABRICPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_fabric(self):
        generator = torch.manual_seed(0)

        pipe = FabricPipeline.from_pretrained("dreamlike-art/dreamlike-photoreal-2.0")
        pipe.to("cuda")

        prompt = "a photograph of an astronaut riding a horse"

        images = pipe(prompt, random_seed=generator).images

        for word, image in zip(prompt, images):
            expected_image = load_numpy(
                f"https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/dit/fabric_wo_feedback.npy"
            )
            assert np.abs((expected_image - np.array(image)).max()) < 1e-2

    def test_fabric_feedback(self):
        generator = torch.manual_seed(0)

        pipe = FabricPipeline.from_pretrained("dreamlike-art/dreamlike-photoreal-2.0")
        pipe.to("cuda")

        prompt = "a photograph of an astronaut riding a horse"
        images = pipe(prompt, random_seed=generator).images

        liked = [images]
        images = pipe(prompt, random_seed=generator, liked=liked).images

        for word, image in zip(prompt, images):
            expected_image = load_numpy(
                f"https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/dit/fabric_w_feedback.npy"
            )
            assert np.abs((expected_image - np.array(image)).max()) < 1e-2
