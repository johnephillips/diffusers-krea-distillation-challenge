# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
""" Conversion script for the Stable Diffusion checkpoints."""

import re

import torch

from diffusers import (
    AudioLDMPipeline,
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DModel,
)
from diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion import LDMBertConfig, LDMBertModel
from transformers import (
    AutoTokenizer,
    ClapTextConfig,
    ClapTextModelWithProjection,
    SpeechT5HifiGan,
    SpeechT5HifiGanConfig,
)

from ...utils import is_omegaconf_available, is_safetensors_available
from ...utils.import_utils import BACKENDS_MAPPING


def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return ".".join(path.split(".")[n_shave_prefix_segments:])
    else:
        return ".".join(path.split(".")[:n_shave_prefix_segments])


def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item.replace("in_layers.0", "norm1")
        new_item = new_item.replace("in_layers.2", "conv1")

        new_item = new_item.replace("out_layers.0", "norm2")
        new_item = new_item.replace("out_layers.3", "conv2")

        new_item = new_item.replace("emb_layers.1", "time_emb_proj")
        new_item = new_item.replace("skip_connection", "conv_shortcut")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("nin_shortcut", "conv_shortcut")
        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_attention_paths(old_list):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item
        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_vae_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("norm.weight", "group_norm.weight")
        new_item = new_item.replace("norm.bias", "group_norm.bias")

        new_item = new_item.replace("q.weight", "query.weight")
        new_item = new_item.replace("q.bias", "query.bias")

        new_item = new_item.replace("k.weight", "key.weight")
        new_item = new_item.replace("k.bias", "key.bias")

        new_item = new_item.replace("v.weight", "value.weight")
        new_item = new_item.replace("v.bias", "value.bias")

        new_item = new_item.replace("proj_out.weight", "proj_attn.weight")
        new_item = new_item.replace("proj_out.bias", "proj_attn.bias")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def assign_to_checkpoint(
    paths, checkpoint, old_checkpoint, attention_paths_to_split=None, additional_replacements=None, config=None
):
    """
    This does the final conversion step: take locally converted weights and apply a global renaming to them. It splits
    attention layers, and takes into account additional replacements that may arise.

    Assigns the weights to the new checkpoint.
    """
    assert isinstance(paths, list), "Paths should be a list of dicts containing 'old' and 'new' keys."

    # Splits the attention layers into three variables.
    if attention_paths_to_split is not None:
        for path, path_map in attention_paths_to_split.items():
            old_tensor = old_checkpoint[path]
            channels = old_tensor.shape[0] // 3

            target_shape = (-1, channels) if len(old_tensor.shape) == 3 else (-1)

            num_heads = old_tensor.shape[0] // config["num_head_channels"] // 3

            old_tensor = old_tensor.reshape((num_heads, 3 * channels // num_heads) + old_tensor.shape[1:])
            query, key, value = old_tensor.split(channels // num_heads, dim=1)

            checkpoint[path_map["query"]] = query.reshape(target_shape)
            checkpoint[path_map["key"]] = key.reshape(target_shape)
            checkpoint[path_map["value"]] = value.reshape(target_shape)

    for path in paths:
        new_path = path["new"]

        # These have already been assigned
        if attention_paths_to_split is not None and new_path in attention_paths_to_split:
            continue

        # Global renaming happens here
        new_path = new_path.replace("middle_block.0", "mid_block.resnets.0")
        new_path = new_path.replace("middle_block.1", "mid_block.attentions.0")
        new_path = new_path.replace("middle_block.2", "mid_block.resnets.1")

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement["old"], replacement["new"])

        # proj_attn.weight has to be converted from conv 1D to linear
        if "proj_attn.weight" in new_path:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0]
        else:
            checkpoint[new_path] = old_checkpoint[path["old"]]


def conv_attn_to_linear(checkpoint):
    keys = list(checkpoint.keys())
    attn_keys = ["query.weight", "key.weight", "value.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in attn_keys:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0, 0]
        elif "proj_attn.weight" in key:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0]


def create_unet_diffusers_config(original_config, image_size: int):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    unet_params = original_config.model.params.unet_config.params
    vae_params = original_config.model.params.first_stage_config.params.ddconfig

    block_out_channels = [unet_params.model_channels * mult for mult in unet_params.channel_mult]

    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnDownBlock2D" if resolution in unet_params.attention_resolutions else "DownBlock2D"
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnUpBlock2D" if resolution in unet_params.attention_resolutions else "UpBlock2D"
        up_block_types.append(block_type)
        resolution //= 2

    vae_scale_factor = 2 ** (len(vae_params.ch_mult) - 1)

    extra_film_condition_dim = (
        unet_params.extra_film_condition_dim if "extra_film_condition_dim" in unet_params else None
    )
    extra_film_use_concat = unet_params.extra_film_use_concat if "extra_film_use_concat" in unet_params else None

    config = dict(
        sample_size=image_size // vae_scale_factor,
        in_channels=unet_params.in_channels,
        out_channels=unet_params.out_channels,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        layers_per_block=unet_params.num_res_blocks,
        cross_attention_dim=True,
        extra_film_condition_dim=extra_film_condition_dim,
        extra_film_use_concat=extra_film_use_concat,
    )

    return config


def create_vae_diffusers_config(original_config, scaling_factor: float, image_size: int):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    vae_params = original_config.model.params.first_stage_config.params.ddconfig
    _ = original_config.model.params.first_stage_config.params.embed_dim

    block_out_channels = [vae_params.ch * mult for mult in vae_params.ch_mult]
    down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
    up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)

    config = dict(
        sample_size=image_size,
        in_channels=vae_params.in_channels,
        out_channels=vae_params.out_ch,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        latent_channels=vae_params.z_channels,
        layers_per_block=vae_params.num_res_blocks,
        scaling_factor=float(scaling_factor)
    )
    return config


def create_diffusers_schedular(original_config):
    schedular = DDIMScheduler(
        num_train_timesteps=original_config.model.params.timesteps,
        beta_start=original_config.model.params.linear_start,
        beta_end=original_config.model.params.linear_end,
        beta_schedule="scaled_linear",
    )
    return schedular


def create_ldm_bert_config(original_config):
    bert_params = original_config.model.parms.cond_stage_config.params
    config = LDMBertConfig(
        d_model=bert_params.n_embed,
        encoder_layers=bert_params.n_layer,
        encoder_ffn_dim=bert_params.n_embed * 4,
    )
    return config


def convert_ldm_unet_checkpoint(checkpoint, config, path=None, extract_ema=False):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """

    # extract state_dict for UNet
    unet_state_dict = {}
    keys = list(checkpoint.keys())

    unet_key = "model.diffusion_model."
    # at least a 100 parameters have to start with `model_ema` in order for the checkpoint to be EMA
    if sum(k.startswith("model_ema") for k in keys) > 100 and extract_ema:
        print(f"Checkpoint {path} has both EMA and non-EMA weights.")
        print(
            "In this conversion only the EMA weights are extracted. If you want to instead extract the non-EMA"
            " weights (useful to continue fine-tuning), please make sure to remove the `--extract_ema` flag."
        )
        for key in keys:
            if key.startswith("model.diffusion_model"):
                flat_ema_key = "model_ema." + "".join(key.split(".")[1:])
                unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(flat_ema_key)
    else:
        if sum(k.startswith("model_ema") for k in keys) > 100:
            print(
                "In this conversion only the non-EMA weights are extracted. If you want to instead extract the EMA"
                " weights (usually better for inference), please make sure to add the `--extract_ema` flag."
            )

        for key in keys:
            if key.startswith(unet_key):
                unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(key)

    new_checkpoint = {}

    new_checkpoint["time_embedding.linear_1.weight"] = unet_state_dict["time_embed.0.weight"]
    new_checkpoint["time_embedding.linear_1.bias"] = unet_state_dict["time_embed.0.bias"]
    new_checkpoint["time_embedding.linear_2.weight"] = unet_state_dict["time_embed.2.weight"]
    new_checkpoint["time_embedding.linear_2.bias"] = unet_state_dict["time_embed.2.bias"]

    new_checkpoint["film_embedding.weight"] = unet_state_dict["film_emb.weight"]
    new_checkpoint["film_embedding.bias"] = unet_state_dict["film_emb.bias"]

    new_checkpoint["conv_in.weight"] = unet_state_dict["input_blocks.0.0.weight"]
    new_checkpoint["conv_in.bias"] = unet_state_dict["input_blocks.0.0.bias"]

    new_checkpoint["conv_norm_out.weight"] = unet_state_dict["out.0.weight"]
    new_checkpoint["conv_norm_out.bias"] = unet_state_dict["out.0.bias"]
    new_checkpoint["conv_out.weight"] = unet_state_dict["out.2.weight"]
    new_checkpoint["conv_out.bias"] = unet_state_dict["out.2.bias"]

    # Retrieves the keys for the input blocks only
    num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
    input_blocks = {
        layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}" in key]
        for layer_id in range(num_input_blocks)
    }

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
    middle_blocks = {
        layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }

    # Retrieves the keys for the output blocks only
    num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "output_blocks" in layer})
    output_blocks = {
        layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}" in key]
        for layer_id in range(num_output_blocks)
    }

    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        resnets = [
            key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
        ]
        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

        if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.weight"
            )
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.bias"
            )

        paths = renew_resnet_paths(resnets)
        meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"}
        assign_to_checkpoint(
            paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
        )

        if len(attentions):
            paths = renew_attention_paths(attentions)
            meta_path = {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"}
            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]

    resnet_0_paths = renew_resnet_paths(resnet_0)
    assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)

    resnet_1_paths = renew_resnet_paths(resnet_1)
    assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)

    attentions_paths = renew_attention_paths(attentions)
    meta_path = {"old": "middle_block.1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        attentions_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
    )

    for i in range(num_output_blocks):
        block_id = i // (config["layers_per_block"] + 1)
        layer_in_block_id = i % (config["layers_per_block"] + 1)
        output_block_layers = [shave_segments(name, 2) for name in output_blocks[i]]
        output_block_list = {}

        for layer in output_block_layers:
            layer_id, layer_name = layer.split(".")[0], shave_segments(layer, 1)
            if layer_id in output_block_list:
                output_block_list[layer_id].append(layer_name)
            else:
                output_block_list[layer_id] = [layer_name]

        if len(output_block_list) > 1:
            resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.0" in key]
            attentions = [key for key in output_blocks[i] if f"output_blocks.{i}.1" in key]

            resnet_0_paths = renew_resnet_paths(resnets)
            paths = renew_resnet_paths(resnets)

            meta_path = {"old": f"output_blocks.{i}.0", "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"}
            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

            output_block_list = {k: sorted(v) for k, v in output_block_list.items()}
            if ["conv.bias", "conv.weight"] in output_block_list.values():
                index = list(output_block_list.values()).index(["conv.bias", "conv.weight"])
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.weight"
                ]
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.bias"
                ]

                # Clear attentions as they have been attributed above.
                if len(attentions) == 2:
                    attentions = []

            if len(attentions):
                paths = renew_attention_paths(attentions)
                meta_path = {
                    "old": f"output_blocks.{i}.1",
                    "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id}",
                }
                assign_to_checkpoint(
                    paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
                )
        else:
            resnet_0_paths = renew_resnet_paths(output_block_layers, n_shave_prefix_segments=1)
            for path in resnet_0_paths:
                old_path = ".".join(["output_blocks", str(i), path["old"]])
                new_path = ".".join(["up_blocks", str(block_id), "resnets", str(layer_in_block_id), path["new"]])

                new_checkpoint[new_path] = unet_state_dict[old_path]

    return new_checkpoint


def convert_ldm_vae_checkpoint(checkpoint, config):
    # extract state dict for VAE
    vae_state_dict = {}
    vae_key = "first_stage_model."
    keys = list(checkpoint.keys())
    for key in keys:
        if key.startswith(vae_key):
            vae_state_dict[key.replace(vae_key, "")] = checkpoint.get(key)

    new_checkpoint = {}

    new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
    new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
    new_checkpoint["encoder.conv_out.weight"] = vae_state_dict["encoder.conv_out.weight"]
    new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
    new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict["encoder.norm_out.weight"]
    new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict["encoder.norm_out.bias"]

    new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv_in.weight"]
    new_checkpoint["decoder.conv_in.bias"] = vae_state_dict["decoder.conv_in.bias"]
    new_checkpoint["decoder.conv_out.weight"] = vae_state_dict["decoder.conv_out.weight"]
    new_checkpoint["decoder.conv_out.bias"] = vae_state_dict["decoder.conv_out.bias"]
    new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict["decoder.norm_out.weight"]
    new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict["decoder.norm_out.bias"]

    new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
    new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
    new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
    new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "encoder.down" in layer})
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
    }

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "decoder.up" in layer})
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]

        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.weight"
            )
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.bias"
            )

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]

        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.weight"
            ]
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.bias"
            ]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)
    return new_checkpoint

def create_transformers_diffusers_config(original_config):
    """
    Creates a config for transformers SpeechT5HifiGan based on the config of the vocoder model.
    """
    config = dict(
    model_in_dim = original_config.num_mels,
    sampling_rate = original_config.sampling_rate,
    upsample_initial_channel = original_config.upsample_initial_channel,
    upsample_rates = list(original_config.upsample_rates),
    upsample_kernel_sizes = list(original_config.upsample_kernel_sizes),
    resblock_kernel_sizes = list(original_config.resblock_kernel_sizes),
    resblock_dilation_sizes = [list(resblock_dilation) for resblock_dilation in original_config.resblock_dilation_sizes],
    normalize_before=False,
    )
    return config

def convert_hifigan_checkpoint(
    checkpoint,
    config,
):
    # extract state dict for vocoder
    vocoder_state_dict = {}
    vocoder_key = "first_stage_model.vocoder."
    keys = list(checkpoint.keys())
    for key in keys:
        if key.startswith(vocoder_key):
            vocoder_state_dict[key.replace(vocoder_key, "")] = checkpoint.get(key)

    for i in range(len(config.upsample_rates)):
        vocoder_state_dict[f"upsampler.{i}.weight"] = vocoder_state_dict.pop(f"ups.{i}.weight")
        vocoder_state_dict[f"upsampler.{i}.bias"] = vocoder_state_dict.pop(f"ups.{i}.bias")

    if not config.normalize_before:
        # these are dummy variables that are unused if we don't normalize before the vocoder
        vocoder_state_dict["mean"] = torch.zeros(config.model_in_dim)
        vocoder_state_dict["scale"] = torch.ones(config.model_in_dim)

    return vocoder_state_dict

def default_vocoder_config():
    return {
        "upsample_rates": [5, 4, 2, 2, 2],
        "upsample_kernel_sizes": [16, 16, 8, 4, 4],
        "upsample_initial_channel": 1024,
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "num_mels": 64,
        "num_freq": 1025,
        "n_fft": 1024,
        "hop_size": 160,
        "win_size": 1024,
        "sampling_rate": 16000,
        "fmin": 0,
        "fmax": 8000,
        "fmax_for_loss": None,
    }

# TODO: remove this func for final PR
# Copied from https://huggingface.co/spaces/haoheliu/audioldm-text-to-audio-generation/blob/84a0384742a22bd80c44e903e241f0623e874f1d/audioldm/utils.py#L72-L73
def default_audioldm_config():
    return {
                "model": {
                    "params": {
                        "base_learning_rate": 5e-06,
                        "linear_start": 0.0015,
                        "linear_end": 0.0195,
                        "num_timesteps_cond": 1,
                        "log_every_t": 200,
                        "timesteps": 1000,
                        "first_stage_key": "fbank",
                        "cond_stage_key": "waveform",
                        "latent_t_size": 256,
                        "latent_f_size": 16,
                        "channels": 8,
                        "cond_stage_trainable": True,
                        "conditioning_key": "film",
                        "monitor": "val/loss_simple_ema",
                        "scale_by_std": True,
                        "unet_config": {
                            "target": "audioldm.latent_diffusion.openaimodel.UNetModel",
                            "params": {
                                "image_size": 64,
                                "extra_film_condition_dim": 512,
                                "extra_film_use_concat": True,
                                "in_channels": 8,
                                "out_channels": 8,
                                "model_channels": 128,
                                "attention_resolutions": [8, 4, 2],
                                "num_res_blocks": 2,
                                "channel_mult": [1, 2, 3, 5],
                                "num_head_channels": 32,
                                "use_spatial_transformer": True,
                            },
                        },
                        "first_stage_config": {
                            "base_learning_rate": 4.5e-05,
                            "target": "audioldm.variational_autoencoder.autoencoder.AutoencoderKL",
                            "params": {
                                "monitor": "val/rec_loss",
                                "image_key": "fbank",
                                "subband": 1,
                                "embed_dim": 8,
                                "time_shuffle": 1,
                                "ddconfig": {
                                    "double_z": True,
                                    "z_channels": 8,
                                    "resolution": 256,
                                    "downsample_time": False,
                                    "in_channels": 1,
                                    "out_ch": 1,
                                    "ch": 128,
                                    "ch_mult": [1, 2, 4],
                                    "num_res_blocks": 2,
                                    "attn_resolutions": [],
                                    "dropout": 0.0,
                                },
                            },
                        },
                        "cond_stage_config": {
                            "target": "audioldm.clap.encoders.CLAPAudioEmbeddingClassifierFreev2",
                            "params": {
                                "key": "waveform",
                                "sampling_rate": 16000,
                                "embed_mode": "audio",
                                "unconditional_prob": 0.1,
                            },
                        },
                    },
                },
            }

clap_keys_to_modify_mapping = {
    "text_branch": "text_model",
    "attn": "attention.self",
    "self.proj": "output.dense",
    "attention.self_mask": "attn_mask",
    "mlp.fc1": "intermediate.dense",
    "mlp.fc2": "output.dense",
    "norm1": "layernorm_before",
    "norm2": "layernorm_after",
    "bn0": "batch_norm",
}

clap_keys_to_ignore = ["text_transform"]

def convert_open_clap_checkpoint(checkpoint):
    # extract state dict for VAE
    model_state_dict = {}
    model_key = "cond_stage_model.model.text_"
    keys = list(checkpoint.keys())
    for key in keys:
        if key.startswith(model_key):
            model_state_dict[key.replace(model_key, "text_")] = checkpoint.get(key)

    new_checkpoint = {}

    sequential_layers_pattern = r".*sequential.(\d+).*"
    text_projection_pattern = r".*_projection.(\d+).*"

    for key, value in model_state_dict.items():
        # check if key should be ignored in mapping
        if key.split(".")[0] in clap_keys_to_ignore:
            continue

        # check if any key needs to be modified
        for key_to_modify, new_key in clap_keys_to_modify_mapping.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        if re.match(sequential_layers_pattern, key):
            # replace sequential layers with list
            sequential_layer = re.match(sequential_layers_pattern, key).group(1)

            key = key.replace(f"sequential.{sequential_layer}.", f"layers.{int(sequential_layer)//3}.linear.")
        elif re.match(text_projection_pattern, key):
            projecton_layer = int(re.match(text_projection_pattern, key).group(1))

            # Because in CLAP they use `nn.Sequential`...
            transformers_projection_layer = 1 if projecton_layer == 0 else 2

            key = key.replace(f"_projection.{projecton_layer}.", f"_projection.linear{transformers_projection_layer}.")

        if "audio" and "qkv" in key:
            # split qkv into query key and value
            mixed_qkv = value
            qkv_dim = mixed_qkv.size(0) // 3

            query_layer = mixed_qkv[:qkv_dim]
            key_layer = mixed_qkv[qkv_dim : qkv_dim * 2]
            value_layer = mixed_qkv[qkv_dim * 2 :]

            new_checkpoint[key.replace("qkv", "query")] = query_layer
            new_checkpoint[key.replace("qkv", "key")] = key_layer
            new_checkpoint[key.replace("qkv", "value")] = value_layer
        else:
            new_checkpoint[key] = value

    return new_checkpoint


def load_pipeline_from_original_audioldm_ckpt(
    checkpoint_path: str,
    original_config_file: str = None,
    image_size: int = 1024,
    prediction_type: str = None,
    model_type: str = None,
    extract_ema: bool = False,
    scheduler_type: str = "pndm",
    num_in_channels: int = None,
    device: str = None,
    from_safetensors: bool = False,
    original_vocoder_config_file: str = None,
) -> AudioLDMPipeline:
    """
    Load a Stable Diffusion pipeline object from a CompVis-style `.ckpt`/`.safetensors` file and (ideally) a `.yaml`
    config file.

    Although many of the arguments can be automatically inferred, some of these rely on brittle checks against the
    global step count, which will likely fail for models that have undergone further fine-tuning. Therefore, it is
    recommended that you override the default values and/or supply an `original_config_file` wherever possible.

    :param checkpoint_path: Path to `.ckpt` file. :param original_config_file: Path to `.yaml` config file
    corresponding to the original architecture. If `None`, will be
            automatically inferred by looking for a key that only exists in SD2.0 models.
    :param image_size: The image size that the model was trained on. Use 512 for Stable Diffusion v1.X and Stable
    Siffusion v2
            Base. Use 768 for Stable Diffusion v2.
    :param prediction_type: The prediction type that the model was trained on. Use `'epsilon'` for Stable Diffusion
    v1.X and Stable
            Siffusion v2 Base. Use `'v-prediction'` for Stable Diffusion v2.
    :param num_in_channels: The number of input channels. If `None` number of input channels will be automatically
    inferred. :param scheduler_type: Type of scheduler to use. Should be one of `["pndm", "lms", "heun", "euler",
    "euler-ancestral", "dpm", "ddim"]`. :param model_type: The pipeline type. `None` to automatically infer, or one of
    `["FrozenOpenClapEmbedder", "FrozenCLAPEmbedder", "PaintByExample"]`. :param extract_ema: Only relevant for
    checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights
            or not. Defaults to `False`. Pass `True` to extract the EMA weights. EMA weights usually yield higher
            quality images for inference. Non-EMA weights are usually better to continue fine-tuning.
    :param device: The device to use. Pass `None` to determine automatically. :param from_safetensors: If
    `checkpoint_path` is in `safetensors` format, load checkpoint with safetensors instead of PyTorch. :return: A
    StableDiffusionPipeline object representing the passed-in `.ckpt`/`.safetensors` file.
    """

    if not is_omegaconf_available():
        raise ValueError(BACKENDS_MAPPING["omegaconf"][1])

    from omegaconf import OmegaConf

    if from_safetensors:
        if not is_safetensors_available():
            raise ValueError(BACKENDS_MAPPING["safetensors"][1])

        from safetensors import safe_open

        checkpoint = {}
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                checkpoint[key] = f.get_tensor(key)
    else:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if original_config_file is None:
        original_config = default_audioldm_config()
        original_config = OmegaConf.create(original_config)
    else:
        original_config = OmegaConf.load(original_config_file)

    if num_in_channels is not None:
        original_config["model"]["params"]["unet_config"]["params"]["in_channels"] = num_in_channels

    if (
        "parameterization" in original_config["model"]["params"]
        and original_config["model"]["params"]["parameterization"] == "v"
    ):
        if prediction_type is None:
            prediction_type = "v_prediction"
        if image_size is None:
            image_size = 1024
    else:
        if prediction_type is None:
            prediction_type = "epsilon"
        if image_size is None:
            image_size = 1024

    num_train_timesteps = original_config.model.params.timesteps
    beta_start = original_config.model.params.linear_start
    beta_end = original_config.model.params.linear_end

    scheduler = DDIMScheduler(
        beta_end=beta_end,
        beta_schedule="scaled_linear",
        beta_start=beta_start,
        num_train_timesteps=num_train_timesteps,
        steps_offset=1,
        clip_sample=False,
        set_alpha_to_one=False,
        prediction_type=prediction_type,
    )
    # make sure scheduler works correctly with DDIM
    scheduler.register_to_config(clip_sample=False)

    if scheduler_type == "pndm":
        config = dict(scheduler.config)
        config["skip_prk_steps"] = True
        scheduler = PNDMScheduler.from_config(config)
    elif scheduler_type == "lms":
        scheduler = LMSDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "heun":
        scheduler = HeunDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "euler":
        scheduler = EulerDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "euler-ancestral":
        scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "dpm":
        scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
    elif scheduler_type == "ddim":
        scheduler = scheduler
    else:
        raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")

    # Convert the UNet2DModel
    unet_config = create_unet_diffusers_config(original_config, image_size=image_size)
    unet = UNet2DModel(**unet_config)

    converted_unet_checkpoint = convert_ldm_unet_checkpoint(
        checkpoint, unet_config, path=checkpoint_path, extract_ema=extract_ema
    )

    unet.load_state_dict(converted_unet_checkpoint)

    # Convert the VAE model.
    vae_config = create_vae_diffusers_config(original_config, scaling_factor=checkpoint["scale_factor"], image_size=image_size)
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)

    # Convert the text model.
    if model_type is None:
        model_type = original_config.model.params.cond_stage_config.target.split(".")[-1]

    if model_type == "CLAPAudioEmbeddingClassifierFreev2":
        # AudioLDM uses the same configuration and tokenizer as the original CLAP model
        config = ClapTextConfig.from_pretrained("laion/clap-htsat-unfused")
        tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

        converted_text_model = convert_open_clap_checkpoint(checkpoint)
        text_model = ClapTextModelWithProjection(config)
        text_model.load_state_dict(converted_text_model)

    # Convert the vocoder model TODO: add the vocoder config to full config
    if original_vocoder_config_file is None:
        original_vocoder_config = default_vocoder_config()
        original_vocoder_config = OmegaConf.create(original_vocoder_config)
    else:
        original_vocoder_config = OmegaConf.load(original_vocoder_config_file)

    vocoder_config = create_transformers_diffusers_config(original_vocoder_config)
    vocoder_config = SpeechT5HifiGanConfig(**vocoder_config)
    converted_vocoder_checkpoint = convert_hifigan_checkpoint(checkpoint, vocoder_config)

    vocoder = SpeechT5HifiGan(vocoder_config)
    vocoder.load_state_dict(converted_vocoder_checkpoint)

    pipe = AudioLDMPipeline(
        vae=vae,
        text_encoder=text_model,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        vocoder=vocoder,
    )

    return pipe
