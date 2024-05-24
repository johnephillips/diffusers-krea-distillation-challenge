# Copyright 2024 Marigold authors, PRS ETH Zurich. All rights reserved.
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
# --------------------------------------------------------------------------
# More information and citation instructions are available on the
# Marigold project website: https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------
import gc
import random
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    AutoencoderTiny,
    LCMScheduler,
    MarigoldDepthPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    print_tensor_test,
    require_torch_gpu,
    slow,
)

from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class MarigoldDepthPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = MarigoldDepthPipeline
    params = frozenset(["image"])
    batch_params = frozenset(["image"])
    image_params = frozenset(["image"])
    image_latents_params = frozenset(["latents"])
    callback_cfg_params = frozenset([])
    test_xformers_attention = False
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "output_type",
        ]
    )

    def get_dummy_components(self, time_cond_proj_dim=None):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            time_cond_proj_dim=time_cond_proj_dim,
            sample_size=32,
            in_channels=8,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        torch.manual_seed(0)
        scheduler = LCMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            prediction_type="v_prediction",
            set_alpha_to_one=False,
            steps_offset=1,
            beta_schedule="scaled_linear",
            clip_sample=False,
            thresholding=False,
        )
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
            "prediction_type": "depth",
            "scale_invariant": True,
            "shift_invariant": True,
        }
        return components

    def get_dummy_tiny_autoencoder(self):
        return AutoencoderTiny(in_channels=3, out_channels=3, latent_channels=4)

    def get_dummy_inputs(self, device, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        image = image / 2 + 0.5
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "image": image,
            "num_inference_steps": 1,
            "processing_resolution": 0,
            "generator": generator,
            "output_type": "np",
        }
        return inputs

    def _test_marigold_depth(
        self,
        generator_seed: int = 0,
        expected_slice: np.ndarray = None,
        atol: float = 1e-4,
        **pipe_kwargs,
    ):
        device = "cpu"
        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        pipe_inputs = self.get_dummy_inputs(device, seed=generator_seed)
        pipe_inputs.update(**pipe_kwargs)

        prediction = pipe(**pipe_inputs).prediction

        print_tensor_test(prediction, limit_to_slices=True)
        prediction_slice = prediction[0, -3:, -3:, -1].flatten()

        if pipe_inputs.get("match_input_resolution", True):
            self.assertEqual(prediction.shape, (1, 32, 32, 1), "Unexpected output resolution")
        else:
            self.assertTrue(prediction.shape[0] == 1 and prediction.shape[3] == 1, "Unexpected output dimensions")
            self.assertEqual(
                max(prediction.shape[1:3]),
                pipe_inputs.get("processing_resolution", 768),
                "Unexpected output resolution",
            )

        self.assertTrue(np.allclose(prediction_slice, expected_slice, atol=atol))

    def test_marigold_depth_dummy_defaults(self):
        self._test_marigold_depth(
            expected_slice=np.array([0.4529, 0.5184, 0.4985, 0.4355, 0.4273, 0.4153, 0.5229, 0.4818, 0.4627]),
        )

    def test_marigold_depth_dummy_G0_S1_P32_E1_B1_M1(self):
        self._test_marigold_depth(
            generator_seed=0,
            expected_slice=np.array([0.4529, 0.5184, 0.4985, 0.4355, 0.4273, 0.4153, 0.5229, 0.4818, 0.4627]),
            num_inference_steps=1,
            processing_resolution=32,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G0_S1_P16_E1_B1_M1(self):
        self._test_marigold_depth(
            generator_seed=0,
            expected_slice=np.array([0.4511, 0.4531, 0.4542, 0.5024, 0.4987, 0.4969, 0.5281, 0.5215, 0.5182]),
            num_inference_steps=1,
            processing_resolution=16,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G2024_S1_P32_E1_B1_M1(self):
        self._test_marigold_depth(
            generator_seed=2024,
            expected_slice=np.array([0.4671, 0.4739, 0.5130, 0.4308, 0.4411, 0.4720, 0.5064, 0.4796, 0.4795]),
            num_inference_steps=1,
            processing_resolution=32,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G0_S2_P32_E1_B1_M1(self):
        self._test_marigold_depth(
            generator_seed=0,
            expected_slice=np.array([0.4165, 0.4485, 0.4647, 0.4003, 0.4577, 0.5074, 0.5106, 0.5077, 0.5042]),
            num_inference_steps=2,
            processing_resolution=32,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G0_S1_P64_E1_B1_M1(self):
        self._test_marigold_depth(
            generator_seed=0,
            expected_slice=np.array([0.4817, 0.5425, 0.5146, 0.5367, 0.5034, 0.4743, 0.4395, 0.4734, 0.4399]),
            num_inference_steps=1,
            processing_resolution=64,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G0_S1_P32_E3_B1_M1(self):
        self._test_marigold_depth(
            generator_seed=0,
            expected_slice=np.array([0.3198, 0.3486, 0.2731, 0.2900, 0.2694, 0.2391, 0.4086, 0.3505, 0.3194]),
            num_inference_steps=1,
            processing_resolution=32,
            ensemble_size=3,
            ensembling_kwargs={"reduction": "mean"},
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G0_S1_P32_E4_B2_M1(self):
        self._test_marigold_depth(
            generator_seed=0,
            expected_slice=np.array([0.3179, 0.4160, 0.2991, 0.2904, 0.3228, 0.2878, 0.4691, 0.4148, 0.3683]),
            num_inference_steps=1,
            processing_resolution=32,
            ensemble_size=4,
            ensembling_kwargs={"reduction": "mean"},
            batch_size=2,
            match_input_resolution=True,
        )

    def test_marigold_depth_dummy_G0_S1_P16_E1_B1_M0(self):
        self._test_marigold_depth(
            generator_seed=0,
            expected_slice=np.array([0.5515, 0.4588, 0.4197, 0.4741, 0.4229, 0.4328, 0.5333, 0.5314, 0.5182]),
            num_inference_steps=1,
            processing_resolution=16,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=False,
        )

    def test_marigold_depth_dummy_no_num_inference_steps(self):
        with self.assertRaises(ValueError) as e:
            self._test_marigold_depth(
                num_inference_steps=None,
                expected_slice=np.array([0.0]),
            )
            self.assertIn("num_inference_steps", str(e))

    def test_marigold_depth_dummy_no_processing_resolution(self):
        with self.assertRaises(ValueError) as e:
            self._test_marigold_depth(
                processing_resolution=None,
                expected_slice=np.array([0.0]),
            )
            self.assertIn("processing_resolution", str(e))


@slow
@require_torch_gpu
class MarigoldDepthPipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def _test_marigold_depth(
        self,
        is_fp16: bool = True,
        device: str = "cuda",
        generator_seed: int = 0,
        expected_slice: np.ndarray = None,
        model_id: str = "prs-eth/marigold-lcm-v1-0",
        image_url: str = "https://marigoldmonodepth.github.io/images/einstein.jpg",
        atol: float = 1e-4,
        **pipe_kwargs,
    ):
        from_pretrained_kwargs = {}
        if is_fp16:
            from_pretrained_kwargs["variant"] = "fp16"
            from_pretrained_kwargs["torch_dtype"] = torch.float16

        pipe = MarigoldDepthPipeline.from_pretrained(model_id, **from_pretrained_kwargs)
        if device == "cuda":
            pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=device).manual_seed(generator_seed)

        image = load_image(image_url)
        width, height = image.size

        prediction = pipe(image, generator=generator, **pipe_kwargs).prediction

        print_tensor_test(prediction, limit_to_slices=True)
        prediction_slice = prediction[0, -3:, -3:, -1].flatten()

        if pipe_kwargs.get("match_input_resolution", True):
            self.assertEqual(prediction.shape, (1, height, width, 1), "Unexpected output resolution")
        else:
            self.assertTrue(prediction.shape[0] == 1 and prediction.shape[3] == 1, "Unexpected output dimensions")
            self.assertEqual(
                max(prediction.shape[1:3]),
                pipe_kwargs.get("processing_resolution", 768),
                "Unexpected output resolution",
            )

        self.assertTrue(np.allclose(prediction_slice, expected_slice, atol=atol))

    def test_marigold_depth_einstein_f32_cpu_G0_S1_P32_E1_B1_M1(self):
        self._test_marigold_depth(
            is_fp16=False,
            device="cpu",
            generator_seed=0,
            expected_slice=np.array([0.4323, 0.4323, 0.4323, 0.4323, 0.4323, 0.4323, 0.4323, 0.4323, 0.4323]),
            num_inference_steps=1,
            processing_resolution=32,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_einstein_f32_cuda_G0_S1_P768_E1_B1_M1(self):
        self._test_marigold_depth(
            is_fp16=False,
            device="cuda",
            generator_seed=0,
            expected_slice=np.array([0.0]),
            num_inference_steps=1,
            processing_resolution=768,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_einstein_f16_cuda_G0_S1_P768_E1_B1_M1(self):
        self._test_marigold_depth(
            is_fp16=True,
            device="cuda",
            generator_seed=0,
            expected_slice=np.array([0.0]),
            num_inference_steps=1,
            processing_resolution=768,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_einstein_f16_cuda_G2024_S1_P768_E1_B1_M1(self):
        self._test_marigold_depth(
            is_fp16=True,
            device="cuda",
            generator_seed=2024,
            expected_slice=np.array([0.0]),
            num_inference_steps=1,
            processing_resolution=768,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_einstein_f16_cuda_G0_S2_P768_E1_B1_M1(self):
        self._test_marigold_depth(
            is_fp16=True,
            device="cuda",
            generator_seed=0,
            expected_slice=np.array([0.0]),
            num_inference_steps=2,
            processing_resolution=768,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_einstein_f16_cuda_G0_S1_P512_E1_B1_M1(self):
        self._test_marigold_depth(
            is_fp16=True,
            device="cuda",
            generator_seed=0,
            expected_slice=np.array([0.0]),
            num_inference_steps=1,
            processing_resolution=512,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_einstein_f16_cuda_G0_S1_P768_E3_B1_M1(self):
        self._test_marigold_depth(
            is_fp16=True,
            device="cuda",
            generator_seed=0,
            expected_slice=np.array([0.0]),
            num_inference_steps=1,
            processing_resolution=768,
            ensemble_size=3,
            ensembling_kwargs={"reduction": "mean"},
            batch_size=1,
            match_input_resolution=True,
        )

    def test_marigold_depth_einstein_f16_cuda_G0_S1_P768_E4_B2_M1(self):
        self._test_marigold_depth(
            is_fp16=True,
            device="cuda",
            generator_seed=0,
            expected_slice=np.array([0.0]),
            num_inference_steps=1,
            processing_resolution=768,
            ensemble_size=4,
            ensembling_kwargs={"reduction": "mean"},
            batch_size=2,
            match_input_resolution=True,
        )

    def test_marigold_depth_einstein_f16_cuda_G0_S1_P512_E1_B1_M0(self):
        self._test_marigold_depth(
            is_fp16=True,
            device="cuda",
            generator_seed=0,
            expected_slice=np.array([0.0]),
            num_inference_steps=1,
            processing_resolution=512,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=False,
        )
