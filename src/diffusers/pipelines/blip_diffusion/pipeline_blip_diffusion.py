# Copyright 2023 Salesforce.com, inc.
# Copyright 2023 The HuggingFace Team. All rights reserved.#
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
from typing import List, Optional, Union

import PIL
import torch
from transformers import CLIPTokenizer

from ...models import AutoencoderKL, UNet2DConditionModel
from ..pipeline_utils import DiffusionPipeline
from ...schedulers import PNDMScheduler
from ...utils import (
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import ImagePipelineOutput
from .blip_image_processing import BlipImageProcessor
from .modeling_blip2 import Blip2QFormerModel
from .modeling_ctx_clip import ContextCLIPTextModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import BlipDiffusionPipeline
        >>> from PIL import Image
        >>> from diffusers.utils import load_image

        >>> blip_diffusion_pipe = BlipDiffusionPipeline.from_pretrained('ayushtues/blipdiffusion')
        >>> blip_diffusion_pipe.to('cuda')

        >>> cond_subject = ["dog"]
        >>> tgt_subject = ["dog"]
        >>> text_prompt_input = ["swimming underwater"]


        >>> cond_image = load_image(
        ...     "https://huggingface.co/datasets/ayushtues/blipdiffusion_images/resolve/main/dog.jpg"
        ... )
        >>> iter_seed = 88888
        >>> guidance_scale = 7.5
        >>> num_inference_steps = 50
        >>> negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"


        >>> output = blip_diffusion_pipe(
        ...     text_prompt_input,
        ...     cond_image,
        ...     cond_subject,
        ...     tgt_subject,
        ...     guidance_scale=guidance_scale,
        ...     num_inference_steps=num_inference_steps,
        ...     neg_prompt=negative_prompt,
        ...     height=512,
        ...     width=512,
        ...     )
        >>> output[0][0].save("dog.png")
        ```
"""


class BlipDiffusionPipeline(DiffusionPipeline):
    """
    Pipeline for Zero-Shot Subject Driven Generation using Blip Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        tokenizer ([`CLIPTokenizer`]):
            Tokenizer for the text encoder
        text_encoder ([`ContextCLIPTextModel`]):
            Text encoder to encode the text prompt
        vae ([`AutoencoderKL`]):
            VAE model to map the latents to the image
        unet ([`UNet2DConditionModel`]):
            Conditional U-Net architecture to denoise the image embedding.
        scheduler ([`PNDMScheduler`]):
             A scheduler to be used in combination with `unet` to generate image latents.
        qformer ([`Blip2QFormerModel`]):
            QFormer model to get multi-modal embeddings from the text and image.
        image_processor ([`BlipImageProcessor`]):
            Image Processor to preprocess and postprocess the image.
        ctx_begin_pos (int, `optional`, defaults to 2):
            Position of the context token in the text encoder.
    """

    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        text_encoder: ContextCLIPTextModel,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: PNDMScheduler,
        qformer: Blip2QFormerModel,
        image_processor: BlipImageProcessor,
        ctx_begin_pos: int = 2,
        mean: List[float] = None,
        std: List[float] = None,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            qformer=qformer,
            image_processor=image_processor,
        )
        self.register_to_config(ctx_begin_pos=ctx_begin_pos, mean=mean, std=std)

    def get_query_embeddings(self, input_image, src_subject):
        return self.qformer(image_input=input_image, text_input=src_subject, return_dict=False)

    # from the original Blip Diffusion code, speciefies the target subject and augments the prompt by repeating it
    def _build_prompt(self, prompts, tgt_subjects, prompt_strength=1.0, prompt_reps=20):
        rv = []
        for prompt, tgt_subject in zip(prompts, tgt_subjects):
            prompt = f"a {tgt_subject} {prompt.strip()}"
            # a trick to amplify the prompt
            rv.append(", ".join([prompt] * int(prompt_strength * prompt_reps)))

        return rv

    # Copied from diffusers.pipelines.consistency_models.pipeline_consistency_models.ConsistencyModelPipeline._prepare_latents
    def prepare_latents(self, batch_size, num_channels, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels, height, width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def encode_prompt(self, query_embeds, prompt):
        # embeddings for prompt, with query_embeds as context
        max_len = self.text_encoder.text_model.config.max_position_embeddings
        max_len -= self.qformer.config.num_query_tokens

        tokenized_prompt = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(self.device)

        batch_size = query_embeds.shape[0]
        ctx_begin_pos = [self.config.ctx_begin_pos] * batch_size

        text_embeddings = self.text_encoder(
            input_ids=tokenized_prompt.input_ids,
            ctx_embeddings=query_embeds,
            ctx_begin_pos=ctx_begin_pos,
        )[0]

        return text_embeddings

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: List[str],
        reference_image: PIL.Image.Image,
        source_subject_category: List[str],
        target_subject_category: List[str],
        latents: Optional[torch.FloatTensor] = None,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        neg_prompt: Optional[str] = "",
        prompt_strength: float = 1.0,
        prompt_reps: int = 20,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`List[str]`):
                The prompt or prompts to guide the image generation.
            reference_image (`PIL.Image.Image`):
                The reference image to condition the generation on.
            source_subject_category (`List[str]`):
                The source subject category.
            target_subject_category (`List[str]`):
                The target subject category.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by random sampling.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            height (`int`, *optional*, defaults to 512):
                The height of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            neg_prompt (`str`, *optional*, defaults to ""):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            prompt_strength (`float`, *optional*, defaults to 1.0):
                The strength of the prompt. Specifies the number of times the prompt is repeated along with prompt_reps
                to amplify the prompt.
            prompt_reps (`int`, *optional*, defaults to 20):
                The number of times the prompt is repeated along with prompt_strength to amplify the prompt.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"`
                (`np.array`) or `"pt"` (`torch.Tensor`).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`
        """

        reference_image = self.image_processor.preprocess(
            reference_image, image_mean=self.config.mean, image_std=self.config.std, return_tensors="pt"
        )["pixel_values"]
        reference_image = reference_image.to(self.device)

        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(source_subject_category, str):
            source_subject_category = [source_subject_category]
        if isinstance(target_subject_category, str):
            target_subject_category = [target_subject_category]

        batch_size = len(prompt)

        prompt = self._build_prompt(
            prompts=prompt,
            tgt_subjects=target_subject_category,
            prompt_strength=prompt_strength,
            prompt_reps=prompt_reps,
        )
        query_embeds = self.get_query_embeddings(reference_image, source_subject_category)
        text_embeddings = self.encode_prompt(query_embeds, prompt)
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            max_length = self.text_encoder.text_model.config.max_position_embeddings

            uncond_input = self.tokenizer(
                [neg_prompt] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(
                input_ids=uncond_input.input_ids.to(self.device),
                ctx_embeddings=None,
            )[0]
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        scale_down_factor = 2 ** (len(self.unet.block_out_channels) - 1)
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels=self.unet.in_channels,
            height=height // scale_down_factor,
            width=width // scale_down_factor,
            generator=generator,
            latents=latents,
            dtype=self.unet.dtype,
            device=self.device,
        )
        # set timesteps
        extra_set_kwargs = {}
        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            do_classifier_free_guidance = guidance_scale > 1.0

            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            noise_pred = self.unet(
                latent_model_input,
                timestep=t,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=None,
                mid_block_additional_residual=None,
            )["sample"]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
            )["prev_sample"]
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
