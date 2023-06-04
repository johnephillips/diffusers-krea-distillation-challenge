import gc
import unittest

import numpy as np
import pytest
import torch

from diffusers import (
    CMStochasticIterativeScheduler,
    ConsistencyModelPipeline,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    UNet2DModel,
)
from diffusers.utils import slow, torch_device
from diffusers.utils.testing_utils import require_torch_gpu

from ..pipeline_params import UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS, UNCONDITIONAL_IMAGE_GENERATION_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


class ConsistencyModelPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = ConsistencyModelPipeline
    params = UNCONDITIONAL_IMAGE_GENERATION_PARAMS
    batch_params = UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS

    @property
    def dummy_uncond_unet(self):
        unet = UNet2DModel.from_pretrained(
            "dg845/consistency-models-test",
            subfolder="test_unet",
        )
        return unet

    @property
    def dummy_cond_unet(self):
        unet = UNet2DModel.from_pretrained(
            "dg845/consistency-models-test",
            subfolder="test_unet_class_cond",
        )
        return unet

    def get_dummy_components(self, class_cond=False):
        if class_cond:
            unet = self.dummy_cond_unet
        else:
            unet = self.dummy_uncond_unet

        # Default to CM multistep sampler
        scheduler = CMStochasticIterativeScheduler(
            num_train_timesteps=40,
            sigma_min=0.002,
            sigma_max=80.0,
            timesteps=np.array([0, 22]),
        )

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "distillation": True,
        }

        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        inputs = {
            "num_inference_steps": 2,
            "clip_denoised": True,
            "sigma_min": 0.002,
            "sigma_max": 80.0,
            "sigma_data": 0.5,
            "generator": generator,
            "output_type": "numpy",
        }

        return inputs

    def test_consistency_model_pipeline_multistep(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = ConsistencyModelPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.3572, 0.6273, 0.4031, 0.3961, 0.4321, 0.5730, 0.5266, 0.4780, 0.5004])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_consistency_model_pipeline_multistep_class_cond(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components(class_cond=True)
        pipe = ConsistencyModelPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.3572, 0.6273, 0.4031, 0.3961, 0.4321, 0.5730, 0.5266, 0.4780, 0.5004])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_consistency_model_pipeline_multistep_edm(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        components["distillation"] = False
        pipe = ConsistencyModelPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.3576, 0.6270, 0.4034, 0.3964, 0.4323, 0.5728, 0.5265, 0.4781, 0.5004])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_consistency_model_pipeline_onestep(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = ConsistencyModelPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 1
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.5004, 0.5004, 0.4994, 0.5008, 0.4976, 0.5018, 0.4990, 0.4982, 0.4987])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_consistency_model_pipeline_onestep_class_cond(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components(class_cond=True)
        pipe = ConsistencyModelPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 1
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.5004, 0.5004, 0.4994, 0.5008, 0.4976, 0.5018, 0.4990, 0.4982, 0.4987])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_consistency_model_pipeline_onestep_edm(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        components["distillation"] = False
        pipe = ConsistencyModelPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 1
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.5004, 0.5004, 0.4994, 0.5008, 0.4976, 0.5018, 0.4990, 0.4982, 0.4987])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_consistency_model_pipeline_k_euler(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_uncond_unet
        scheduler = EulerDiscreteScheduler(
            num_train_timesteps=2,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="linear",
            prediction_type="sample",
            use_karras_sigmas=True,
        )
        pipe = ConsistencyModelPipeline(unet=unet, scheduler=scheduler, distillation=False)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.5157, 0.5143, 0.4804, 0.5273, 0.4146, 0.5619, 0.4651, 0.4359, 0.4540])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    @pytest.mark.xfail(reason="Heun scheduler does not implement prediction_type 'sample' yet")
    def test_consistency_model_pipeline_k_heun(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_uncond_unet
        scheduler = HeunDiscreteScheduler(
            num_train_timesteps=2,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="linear",
            prediction_type="sample",
            use_karras_sigmas=True,
        )
        pipe = ConsistencyModelPipeline(unet=unet, scheduler=scheduler, distillation=False)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.5159, 0.5145, 0.4801, 0.5277, 0.4134, 0.5628, 0.4646, 0.4350, 0.4533])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3


@slow
@require_torch_gpu
class ConsistencyModelPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, seed=0):
        generator = torch.manual_seed(seed)

        inputs = {
            "num_inference_steps": 2,
            "class_labels": 0,
            "clip_denoised": True,
            "sigma_min": 0.002,
            "sigma_max": 80.0,
            "sigma_data": 0.5,
            "generator": generator,
            "output_type": "numpy",
        }

        return inputs

    def test_consistency_model_cd_multistep(self):
        unet = UNet2DModel.from_pretrained("ayushtues/consistency_models", subfolder="diffusers_cd_imagenet64_l2")
        scheduler = CMStochasticIterativeScheduler(
            num_train_timesteps=40,
            sigma_min=0.002,
            sigma_max=80.0,
            timesteps=np.array([0, 22]),
        )
        pipe = ConsistencyModelPipeline(unet=unet, scheduler=scheduler)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        assert image.shape == (1, 64, 64, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.2645, 0.3386, 0.1928, 0.1284, 0.1215, 0.0285, 0.0800, 0.1213, 0.3331])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 3e-3

    def test_consistency_model_cd_onestep(self):
        unet = UNet2DModel.from_pretrained("ayushtues/consistency_models", subfolder="diffusers_cd_imagenet64_l2")
        scheduler = CMStochasticIterativeScheduler(
            num_train_timesteps=40,
            sigma_min=0.002,
            sigma_max=80.0,
            timesteps=np.array([0, 22]),
        )
        pipe = ConsistencyModelPipeline(unet=unet, scheduler=scheduler)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        inputs["num_inference_steps"] = 1
        image = pipe(**inputs).images
        assert image.shape == (1, 64, 64, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.2480, 0.1257, 0.0852, 0.2474, 0.3226, 0.1637, 0.3169, 0.2660, 0.3875])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 3e-3
