import argparse

import torch
import yaml
from safetensors.torch import load_file
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from yaml.loader import FullLoader

from diffusers import StableVideoDragNUWAPipeline
from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import is_accelerate_available, logging


if is_accelerate_available():
    pass

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def create_vae_diffusers_config(original_config, image_size: int):
    r"""
    Creates a vae config for diffusers based on the config of the LDM.
    """
    vae_params = original_config["model"]["params"]["first_stage_config"]["params"]["decoder_config"]["params"]
    block_out_channels = [vae_params["ch"] * mult for mult in vae_params["ch_mult"]]
    down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)

    vae_config = {
        "sample_size": image_size,
        "in_channels": vae_params["in_channels"],
        "out_channels": vae_params["out_ch"],
        "down_block_types": tuple(down_block_types),
        "block_out_channels": tuple(block_out_channels),
        "latent_channels": vae_params["z_channels"],
        "layers_per_block": vae_params["num_res_blocks"],
    }

    return vae_config


def create_unet_diffusers_config(original_config, image_size: int, controlnet=False):
    r"""
    Creates a unet config for diffusers based on the config of the LDM.
    """
    if controlnet:
        unet_params = original_config["model"]["params"]["control_stage_config"]["params"]
    else:
        if (
            "unet_config" in original_config["model"]["params"]
            and original_config["model"]["params"]["unet_config"] is not None
        ):
            unet_params = original_config["model"]["params"]["unet_config"]["params"]
        else:
            unet_params = original_config["model"]["params"]["network_config"]["params"]

    vae_params = original_config["model"]["params"]["first_stage_config"]["params"]["decoder_config"]["params"]

    block_out_channels = [unet_params["model_channels"] * mult for mult in unet_params["channel_mult"]]

    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = (
            "CrossAttnDownBlockSpatioTemporal"
            if resolution in unet_params["attention_resolutions"]
            else "DownBlockSpatioTemporal"
        )
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = (
            "CrossAttnUpBlockSpatioTemporal"
            if resolution in unet_params["attention_resolutions"]
            else "UpBlockSpatioTemporal"
        )
        up_block_types.append(block_type)
        resolution //= 2

    if unet_params["transformer_depth"] is not None:
        transformer_layers_per_block = (
            unet_params["transformer_depth"]
            if isinstance(unet_params["transformer_depth"], int)
            else list(unet_params["transformer_depth"])
        )
    else:
        transformer_layers_per_block = 1

    vae_scale_factor = 2 ** (len(vae_params["ch_mult"]) - 1)

    head_dim = unet_params["num_heads"] if "num_heads" in unet_params else None
    use_linear_projection = (
        unet_params["use_linear_in_transformer"] if "use_linear_in_transformer" in unet_params else False
    )
    if use_linear_projection:
        # stable diffusion 2-base-512 and 2-768
        if head_dim is None:
            head_dim_mult = unet_params["model_channels"] // unet_params["num_head_channels"]
            head_dim = [head_dim_mult * c for c in list(unet_params["channel_mult"])]

    class_embed_type = None
    addition_embed_type = None
    addition_time_embed_dim = None
    projection_class_embeddings_input_dim = None
    context_dim = None

    if unet_params["context_dim"] is not None:
        context_dim = (
            unet_params["context_dim"]
            if isinstance(unet_params["context_dim"], int)
            else unet_params["context_dim"][0]
        )

    if "num_classes" in unet_params:
        if unet_params["num_classes"] == "sequential":
            addition_time_embed_dim = 256
            assert "adm_in_channels" in unet_params
            projection_class_embeddings_input_dim = unet_params["adm_in_channels"]

    config = {
        "sample_size": image_size // vae_scale_factor,
        "in_channels": unet_params["in_channels"],
        "down_block_types": tuple(down_block_types),
        "block_out_channels": tuple(block_out_channels),
        "layers_per_block": unet_params["num_res_blocks"],
        "cross_attention_dim": context_dim,
        "attention_head_dim": head_dim,
        "use_linear_projection": use_linear_projection,
        "class_embed_type": class_embed_type,
        "addition_embed_type": addition_embed_type,
        "addition_time_embed_dim": addition_time_embed_dim,
        "projection_class_embeddings_input_dim": projection_class_embeddings_input_dim,
        "transformer_layers_per_block": transformer_layers_per_block,
    }

    if "disable_self_attentions" in unet_params:
        config["only_cross_attention"] = unet_params["disable_self_attentions"]

    if "num_classes" in unet_params and isinstance(unet_params["num_classes"], int):
        config["num_class_embeds"] = unet_params["num_classes"]

    if controlnet:
        config["conditioning_channels"] = unet_params["hint_channels"]
    else:
        config["out_channels"] = unet_params["out_channels"]
        config["up_block_types"] = tuple(up_block_types)

    return config


def assign_to_checkpoint(
    paths,
    checkpoint,
    old_checkpoint,
    attention_paths_to_split=None,
    additional_replacements=None,
    config=None,
    mid_block_suffix="",
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

    if mid_block_suffix is not None:
        mid_block_suffix = f".{mid_block_suffix}"
    else:
        mid_block_suffix = ""

    for path in paths:
        new_path = path["new"]

        # These have already been assigned
        if attention_paths_to_split is not None and new_path in attention_paths_to_split:
            continue

        # Global renaming happens here
        new_path = new_path.replace("middle_block.0", f"mid_block.resnets.0{mid_block_suffix}")
        new_path = new_path.replace("middle_block.1", "mid_block.attentions.0")
        new_path = new_path.replace("middle_block.2", f"mid_block.resnets.1{mid_block_suffix}")

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement["old"], replacement["new"])

        # proj_attn.weight has to be converted from conv 1D to linear
        is_attn_weight = "proj_attn.weight" in new_path or ("attentions" in new_path and "to_" in new_path)
        shape = old_checkpoint[path["old"]].shape
        if is_attn_weight and len(shape) == 3:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0]
        elif is_attn_weight and len(shape) == 4:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0, 0]
        else:
            checkpoint[new_path] = old_checkpoint[path["old"]]


def renew_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("time_stack", "temporal_transformer_blocks")

        new_item = new_item.replace("time_pos_embed.0.bias", "time_pos_embed.linear_1.bias")
        new_item = new_item.replace("time_pos_embed.0.weight", "time_pos_embed.linear_1.weight")
        new_item = new_item.replace("time_pos_embed.2.bias", "time_pos_embed.linear_2.bias")
        new_item = new_item.replace("time_pos_embed.2.weight", "time_pos_embed.linear_2.weight")

        mapping.append({"old": old_item, "new": new_item})

    return mapping


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

        new_item = new_item.replace("time_stack.", "")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def convert_ldm_unet_checkpoint(
    checkpoint, config, path=None, extract_ema=False, controlnet=False, skip_extract_state_dict=False
):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """

    if skip_extract_state_dict:
        unet_state_dict = checkpoint
    else:
        # extract state_dict for UNet
        unet_state_dict = {}
        keys = list(checkpoint.keys())

        unet_key = "model.diffusion_model."

        # at least a 100 parameters have to start with `model_ema` in order for the checkpoint to be EMA
        if sum(k.startswith("model_ema") for k in keys) > 100 and extract_ema:
            logger.warning(f"Checkpoint {path} has both EMA and non-EMA weights.")
            logger.warning(
                "In this conversion only the EMA weights are extracted. If you want to instead extract the non-EMA"
                " weights (useful to continue fine-tuning), please make sure to remove the `--extract_ema` flag."
            )
            for key in keys:
                if key.startswith("model.diffusion_model"):
                    flat_ema_key = "model_ema." + "".join(key.split(".")[1:])
                    unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(flat_ema_key)
        else:
            if sum(k.startswith("model_ema") for k in keys) > 100:
                logger.warning(
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

    new_checkpoint["add_embedding.linear_1.weight"] = unet_state_dict["label_emb.0.0.weight"]
    new_checkpoint["add_embedding.linear_1.bias"] = unet_state_dict["label_emb.0.0.bias"]
    new_checkpoint["add_embedding.linear_2.weight"] = unet_state_dict["label_emb.0.2.weight"]
    new_checkpoint["add_embedding.linear_2.bias"] = unet_state_dict["label_emb.0.2.bias"]

    new_checkpoint["conv_in.weight"] = unet_state_dict["input_blocks.0.0.weight"]
    new_checkpoint["conv_in.bias"] = unet_state_dict["input_blocks.0.0.bias"]

    new_checkpoint["conv_norm_out.weight"] = unet_state_dict["out.0.weight"]
    new_checkpoint["conv_norm_out.bias"] = unet_state_dict["out.0.bias"]
    new_checkpoint["conv_out.weight"] = unet_state_dict["out.2.weight"]
    new_checkpoint["conv_out.bias"] = unet_state_dict["out.2.bias"]

    num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
    num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
    num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "output_blocks" in layer})

    # Retrieves the flow layers
    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        for flow_var in [
            "flow_cond_norm",
            "flow_gamma_spatial",
            "flow_gamma_temporal",
            "flow_beta_spatial",
            "flow_beta_temporal",
        ]:
            if f"input_blocks.{i}.0.{flow_var}.weight" in unet_state_dict:
                new_checkpoint[
                    f"down_blocks.{block_id}.resnets.{layer_in_block_id}.spatial_res_block.{flow_var}.weight"
                ] = unet_state_dict.pop(f"input_blocks.{i}.0.{flow_var}.weight")
                new_checkpoint[
                    f"down_blocks.{block_id}.resnets.{layer_in_block_id}.spatial_res_block.{flow_var}.bias"
                ] = unet_state_dict.pop(f"input_blocks.{i}.0.{flow_var}.bias")

    for i in range(num_output_blocks):
        block_id = i // (config["layers_per_block"] + 1)
        layer_in_block_id = i % (config["layers_per_block"] + 1)

        for flow_var in [
            "flow_cond_norm",
            "flow_gamma_spatial",
            "flow_gamma_temporal",
            "flow_beta_spatial",
            "flow_beta_temporal",
        ]:
            if f"output_blocks.{i}.0.{flow_var}.weight" in unet_state_dict:
                new_checkpoint[
                    f"up_blocks.{block_id}.resnets.{layer_in_block_id}.spatial_res_block.{flow_var}.weight"
                ] = unet_state_dict.pop(f"output_blocks.{i}.0.{flow_var}.weight")
                new_checkpoint[
                    f"up_blocks.{block_id}.resnets.{layer_in_block_id}.spatial_res_block.{flow_var}.bias"
                ] = unet_state_dict.pop(f"output_blocks.{i}.0.{flow_var}.bias")

    for key in unet_state_dict:
        if "flow_blocks" in key:
            new_key = key.replace("input", "down").replace("output", "up").replace("op.", "")
            new_checkpoint[new_key] = unet_state_dict[key]

    # Retrieves the keys for the input blocks only
    input_blocks = {
        layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}" in key]
        for layer_id in range(num_input_blocks)
    }

    # Retrieves the keys for the middle blocks only
    middle_blocks = {
        layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }

    # Retrieves the keys for the output blocks only
    output_blocks = {
        layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}" in key]
        for layer_id in range(num_output_blocks)
    }

    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        spatial_resnets = [
            key
            for key in input_blocks[i]
            if f"input_blocks.{i}.0" in key
            and (
                f"input_blocks.{i}.0.op" not in key
                and f"input_blocks.{i}.0.time_stack" not in key
                and f"input_blocks.{i}.0.time_mixer" not in key
            )
        ]
        temporal_resnets = [key for key in input_blocks[i] if f"input_blocks.{i}.0.time_stack" in key]
        # import ipdb; ipdb.set_trace()
        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

        if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.weight"
            )
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.bias"
            )

        paths = renew_resnet_paths(spatial_resnets)
        meta_path = {
            "old": f"input_blocks.{i}.0",
            "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}.spatial_res_block",
        }
        assign_to_checkpoint(
            paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
        )

        paths = renew_resnet_paths(temporal_resnets)
        meta_path = {
            "old": f"input_blocks.{i}.0",
            "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}.temporal_res_block",
        }
        assign_to_checkpoint(
            paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
        )

        # TODO resnet time_mixer.mix_factor
        if f"input_blocks.{i}.0.time_mixer.mix_factor" in unet_state_dict:
            new_checkpoint[
                f"down_blocks.{block_id}.resnets.{layer_in_block_id}.time_mixer.mix_factor"
            ] = unet_state_dict[f"input_blocks.{i}.0.time_mixer.mix_factor"]

        if len(attentions):
            paths = renew_attention_paths(attentions)
            meta_path = {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"}
            # import ipdb; ipdb.set_trace()
            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]

    resnet_0_spatial = [key for key in resnet_0 if "time_stack" not in key and "time_mixer" not in key]
    resnet_0_paths = renew_resnet_paths(resnet_0_spatial)
    # import ipdb; ipdb.set_trace()
    assign_to_checkpoint(
        resnet_0_paths, new_checkpoint, unet_state_dict, config=config, mid_block_suffix="spatial_res_block"
    )

    resnet_0_temporal = [key for key in resnet_0 if "time_stack" in key and "time_mixer" not in key]
    resnet_0_paths = renew_resnet_paths(resnet_0_temporal)
    assign_to_checkpoint(
        resnet_0_paths, new_checkpoint, unet_state_dict, config=config, mid_block_suffix="temporal_res_block"
    )

    resnet_1_spatial = [key for key in resnet_1 if "time_stack" not in key and "time_mixer" not in key]
    resnet_1_paths = renew_resnet_paths(resnet_1_spatial)
    assign_to_checkpoint(
        resnet_1_paths, new_checkpoint, unet_state_dict, config=config, mid_block_suffix="spatial_res_block"
    )

    resnet_1_temporal = [key for key in resnet_1 if "time_stack" in key and "time_mixer" not in key]
    resnet_1_paths = renew_resnet_paths(resnet_1_temporal)
    assign_to_checkpoint(
        resnet_1_paths, new_checkpoint, unet_state_dict, config=config, mid_block_suffix="temporal_res_block"
    )

    new_checkpoint["mid_block.resnets.0.time_mixer.mix_factor"] = unet_state_dict[
        "middle_block.0.time_mixer.mix_factor"
    ]
    new_checkpoint["mid_block.resnets.1.time_mixer.mix_factor"] = unet_state_dict[
        "middle_block.2.time_mixer.mix_factor"
    ]

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
            spatial_resnets = [
                key
                for key in output_blocks[i]
                if f"output_blocks.{i}.0" in key
                and (f"output_blocks.{i}.0.time_stack" not in key and "time_mixer" not in key)
            ]
            # import ipdb; ipdb.set_trace()

            temporal_resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.0.time_stack" in key]

            paths = renew_resnet_paths(spatial_resnets)
            meta_path = {
                "old": f"output_blocks.{i}.0",
                "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}.spatial_res_block",
            }
            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

            paths = renew_resnet_paths(temporal_resnets)
            meta_path = {
                "old": f"output_blocks.{i}.0",
                "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}.temporal_res_block",
            }
            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

            if f"output_blocks.{i}.0.time_mixer.mix_factor" in unet_state_dict:
                new_checkpoint[
                    f"up_blocks.{block_id}.resnets.{layer_in_block_id}.time_mixer.mix_factor"
                ] = unet_state_dict[f"output_blocks.{i}.0.time_mixer.mix_factor"]

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

            attentions = [key for key in output_blocks[i] if f"output_blocks.{i}.1" in key and "conv" not in key]
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
            spatial_layers = [
                layer for layer in output_block_layers if "time_stack" not in layer and "time_mixer" not in layer
            ]
            resnet_0_paths = renew_resnet_paths(spatial_layers, n_shave_prefix_segments=1)
            # import ipdb; ipdb.set_trace()
            for path in resnet_0_paths:
                old_path = ".".join(["output_blocks", str(i), path["old"]])
                new_path = ".".join(
                    ["up_blocks", str(block_id), "resnets", str(layer_in_block_id), "spatial_res_block", path["new"]]
                )

                new_checkpoint[new_path] = unet_state_dict[old_path]

            temporal_layers = [
                layer for layer in output_block_layers if "time_stack" in layer and "time_mixer" not in "layers"
            ]
            resnet_0_paths = renew_resnet_paths(temporal_layers, n_shave_prefix_segments=1)
            # import ipdb; ipdb.set_trace()
            for path in resnet_0_paths:
                old_path = ".".join(["output_blocks", str(i), path["old"]])
                new_path = ".".join(
                    ["up_blocks", str(block_id), "resnets", str(layer_in_block_id), "temporal_res_block", path["new"]]
                )

                new_checkpoint[new_path] = unet_state_dict[old_path]

            new_checkpoint["up_blocks.0.resnets.0.time_mixer.mix_factor"] = unet_state_dict[
                f"output_blocks.{str(i)}.0.time_mixer.mix_factor"
            ]

    return new_checkpoint


def conv_attn_to_linear(checkpoint):
    keys = list(checkpoint.keys())
    attn_keys = ["to_q.weight", "to_k.weight", "to_v.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in attn_keys:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0, 0]
        elif "proj_attn.weight" in key:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0]


def renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0, is_temporal=False):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        # Temporal resnet
        new_item = old_item.replace("in_layers.0", "norm1")
        new_item = new_item.replace("in_layers.2", "conv1")

        new_item = new_item.replace("out_layers.0", "norm2")
        new_item = new_item.replace("out_layers.3", "conv2")

        new_item = new_item.replace("skip_connection", "conv_shortcut")

        new_item = new_item.replace("time_stack.", "temporal_res_block.")

        # Spatial resnet
        new_item = new_item.replace("conv1", "spatial_res_block.conv1")
        new_item = new_item.replace("norm1", "spatial_res_block.norm1")

        new_item = new_item.replace("conv2", "spatial_res_block.conv2")
        new_item = new_item.replace("norm2", "spatial_res_block.norm2")

        new_item = new_item.replace("nin_shortcut", "spatial_res_block.conv_shortcut")

        new_item = new_item.replace("mix_factor", "spatial_res_block.time_mixer.mix_factor")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

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

        new_item = new_item.replace("q.weight", "to_q.weight")
        new_item = new_item.replace("q.bias", "to_q.bias")

        new_item = new_item.replace("k.weight", "to_k.weight")
        new_item = new_item.replace("k.bias", "to_k.bias")

        new_item = new_item.replace("v.weight", "to_v.weight")
        new_item = new_item.replace("v.bias", "to_v.bias")

        new_item = new_item.replace("proj_out.weight", "to_out.0.weight")
        new_item = new_item.replace("proj_out.bias", "to_out.0.bias")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def convert_ldm_vae_checkpoint(checkpoint, config):
    # extract state dict for VAE
    vae_state_dict = {}
    keys = list(checkpoint.keys())
    vae_key = "first_stage_model." if any(k.startswith("first_stage_model.") for k in keys) else ""
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
    new_checkpoint["decoder.time_conv_out.weight"] = vae_state_dict["decoder.conv_out.time_mix_conv.weight"]
    new_checkpoint["decoder.time_conv_out.bias"] = vae_state_dict["decoder.conv_out.time_mix_conv.bias"]

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


def read_config_file(filename):
    # The yaml file contains annotations that certain values should
    # loaded as tuples.
    with open(filename) as f:
        original_config = yaml.load(f, FullLoader)

    return original_config


def load_original_state_dict(filename: str):
    if filename.endswith("safetensors"):
        state_dict = load_file(filename)
    elif filename.endswith("ckpt") or filename.endswith("pth"):
        state_dict = torch.load(filename, mmap=True, map_location="cpu")
    else:
        raise ValueError("File type is not supported")

    if isinstance(state_dict, dict) and "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]

    return state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint to convert.", required=True)
    parser.add_argument(
        "--config_file", type=str, help="The config json file corresponding to the architecture.", required=True
    )
    parser.add_argument("--output_path", default=None, type=str, help="Path to the output model.", required=True)
    parser.add_argument("--flow_dim_scale", default=1, type=int, help="Flow dim scale for DragNUWA")
    parser.add_argument("--sample_size", type=int, default=768, help="VAE sample size")
    parser.add_argument(
        "--use_legacy_autoencoder",
        action="store_true",
        default=False,
        help="Whether or not to use the `quant_conv` layers from the original implementation (which is now legacy behaviour).",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", default=False, help="Whether to push to huggingface hub or not."
    )
    args = parser.parse_args()

    original_config = read_config_file(args.config_file)
    state_dict = load_original_state_dict(args.checkpoint_path)
    lora_state_dict = {}

    for key in list(state_dict.keys()):
        new_key = key.replace("linear.", "")
        state_dict[key.replace("linear.", "")] = state_dict.pop(key)
        if "lora" in key:
            lora_state_dict[new_key] = state_dict.pop(key)
            print(new_key)

    vae_config = create_vae_diffusers_config(original_config, args.sample_size)
    vae = AutoencoderKLTemporalDecoder(**vae_config, use_legacy=args.use_legacy_autoencoder)
    vae_state_dict = convert_ldm_vae_checkpoint(state_dict, vae_config)

    remove = []
    for key in vae_state_dict.keys():
        # i'm sorry to hurt your eyes
        if ("encoder" in key) or (
            "decoder" in key and "resnets" and (("temporal_res_block" in key) or ("time_mixer" in key))
        ):
            remove.append(key)

    for key in remove:
        vae_state_dict[key.replace("spatial_res_block.", "")] = vae_state_dict.pop(key)

    missing_keys, unexpected_keys = vae.load_state_dict(vae_state_dict)
    logger.info(f"[VAE] missing_keys: {missing_keys}")
    logger.info(f"[VAE] unexpected_keys: {unexpected_keys}")

    unet_config = create_unet_diffusers_config(original_config, args.sample_size)
    unet_config["flow_dim_scale"] = args.flow_dim_scale
    unet_state_dict = convert_ldm_unet_checkpoint(state_dict, unet_config)
    unet = UNetSpatioTemporalConditionModel.from_config(unet_config)
    missing_keys, unexpected_keys = unet.load_state_dict(unet_state_dict)
    logger.info(f"[UNet] missing_keys: {missing_keys}")
    logger.info(f"[UNet] unexpected_keys: {unexpected_keys}")
    logger.info("UNet conversion succeeded")

    original_svd_model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(original_svd_model_id, subfolder="image_encoder")
    feature_extractor = CLIPImageProcessor()
    scheduler = EulerDiscreteScheduler.from_pretrained(original_svd_model_id, subfolder="scheduler")

    pipe = StableVideoDragNUWAPipeline(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=scheduler,
        feature_extractor=feature_extractor,
    )

    pipe.save_pretrained(args.output_path)
    if args.push_to_hub:
        logger.info("Pushing float32 version to HF hub")
        pipe.push_to_hub(args.output_path)

    pipe.to(dtype=torch.float16)
    pipe.save_pretrained(args.output_path, variant="fp16")
    if args.push_to_hub:
        logger.info("Pushing float16 version to HF hub")
        pipe.push_to_hub(args.output_path, variant="fp16")