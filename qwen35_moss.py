"""
Qwen3.5 / MOSS-Image GGUF text encoder support for ComfyUI-GGUF.

WHY THIS EXISTS
---------------
ComfyUI core (comfy/text_encoders/qwen35.py + moss_image.py) supports the
MOSS-Image Qwen3.5-9B text encoder, but only through the *safetensors*
loader. ComfyUI-GGUF's CLIPLoaderGGUF rejects the file with

    ValueError: Unexpected text model architecture type in GGUF file: 'qwen35'

because "qwen35" is missing from TXT_ARCH_LIST.

WHAT THIS MODULE DOES
---------------------
It adapts a llama.cpp-converted Qwen3.5 GGUF to the exact key/shape layout
that ComfyUI core expects (comfy/text_encoders/qwen35.py):

  1. KEY REMAP
     llama.cpp GGUF names              ->  core module names
     blk.N.attn_qkv.weight            ->  ...linear_attn.in_proj_qkv.weight
     blk.N.attn_gate.weight           ->  ...linear_attn.in_proj_z.weight
     blk.N.ssm_alpha.weight           ->  ...linear_attn.in_proj_a.weight
     blk.N.ssm_beta.weight            ->  ...linear_attn.in_proj_b.weight
     blk.N.ssm_conv1d.weight          ->  ...linear_attn.conv1d.weight
     blk.N.ssm_dt.bias                ->  ...linear_attn.dt_bias
     blk.N.ssm_a                      ->  ...linear_attn.A_log
     blk.N.ssm_norm.weight            ->  ...linear_attn.norm.weight
     blk.N.ssm_out.weight             ->  ...linear_attn.out_proj.weight
     blk.N.attn_q/k/v.weight          ->  ...self_attn.{q,k,v}_proj.weight
     blk.N.attn_output.weight         ->  ...self_attn.o_proj.weight
     blk.N.attn_q_norm/k_norm.weight  ->  ...self_attn.{q,k}_norm.weight
     blk.N.attn_norm.weight           ->  ...input_layernorm.weight
     blk.N.post_attention_norm.weight ->  ...post_attention_layernorm.weight
     blk.N.ffn_{up,gate,down}.weight  ->  ...mlp.{up,gate,down}_proj.weight
     token_embd.weight                ->  model.language_model.embed_tokens.weight
     output_norm.weight               ->  model.language_model.norm.weight
     output.weight                    ->  lm_head.weight

  2. V-HEAD UNPERMUTE
     llama.cpp stores the linear-attention (GatedDeltaNet) V heads in "tiled"
     order (v_index = j * num_k_heads + k), while ComfyUI core expects the
     HuggingFace "grouped" order (v_index = k * num_v_per_k + j) - see
     core's `query.repeat_interleave(rep, dim=2)`. The permutation is
     inverted here.

  3. CONV1D RESHAPE
     GGML stores conv1d as [kernel, channels]; torch expects
     [channels, 1, kernel].

The tokenizer is NOT part of this: core ships it in
comfy/text_encoders/qwen35_tokenizer/ and loads it itself.
"""

import logging

import torch

from .dequant import dequantize_tensor
from .ops import GGMLTensor

# Toggle this off if your GGUF was produced without llama.cpp's V-head reorder.
UNPERMUTE_V_HEADS = True

# Whether to load the lm_head ("output.weight") into the model.
#
# The lm_head is ONLY used for text *generation* (logits). MOSS-Image outpaint
# uses the text encoder in *encode* mode, where the lm_head is never touched.
#
# ComfyUI-GGUF cannot dequantize a GGMLTensor through core's generic
# `cast_bias_weight` path (core does not know about GGMLTensor and casts the
# raw byte buffer instead), so a quantized lm_head breaks `logits()`.
# Loading it is therefore off by default: the module still exists (created by
# core), it is simply left uninitialised and unused.
INCLUDE_LM_HEAD = False

# leaf module name -> (core module name, drop_weight_suffix)
# Matching is done on the exact leaf name, so "attn_q" can never corrupt
# "attn_qkv" or "attn_q_norm".
LEAF_MAP = {
    # --- linear attention / GatedDeltaNet (24 of 32 layers) ---
    "attn_qkv":    ("linear_attn.in_proj_qkv", False),
    "attn_gate":   ("linear_attn.in_proj_z", False),
    "ssm_alpha":   ("linear_attn.in_proj_a", False),
    "ssm_beta":    ("linear_attn.in_proj_b", False),
    "ssm_conv1d":  ("linear_attn.conv1d", False),
    "ssm_dt":      ("linear_attn.dt_bias", True),
    "ssm_a":       ("linear_attn.A_log", True),
    "ssm_norm":    ("linear_attn.norm", False),
    "ssm_out":     ("linear_attn.out_proj", False),
    # --- full attention / GatedAttention (8 of 32 layers) ---
    "attn_q":      ("self_attn.q_proj", False),
    "attn_k":      ("self_attn.k_proj", False),
    "attn_v":      ("self_attn.v_proj", False),
    "attn_output": ("self_attn.o_proj", False),
    "attn_q_norm": ("self_attn.q_norm", False),
    "attn_k_norm": ("self_attn.k_norm", False),
    # --- block commons ---
    "attn_norm":          ("input_layernorm", False),
    "post_attention_norm": ("post_attention_layernorm", False),
    "ffn_up":    ("mlp.up_proj", False),
    "ffn_gate":  ("mlp.gate_proj", False),
    "ffn_down":  ("mlp.down_proj", False),
}

# global (non per-layer) keys
GLOBAL_MAP = {
    "token_embd.weight":  "model.language_model.embed_tokens.weight",
    "output_norm.weight": "model.language_model.norm.weight",
    "output.weight":      "lm_head.weight",
}


def _get_meta(metadata, *names):
    for n in names:
        if n in metadata:
            return metadata[n]
    return None


def _unpermute_v_heads(tensor, dim, num_k_heads, num_v_per_k, head_dim):
    """Invert llama.cpp's grouped->tiled V-head reorder.

    llama.cpp:  [num_k_heads, num_v_per_k, head_dim]  --permute-->  [num_v_per_k, num_k_heads, head_dim]
    here:       [num_v_per_k, num_k_heads, head_dim]  --permute-->  [num_k_heads, num_v_per_k, head_dim]
    """
    shape = list(tensor.shape)
    n = shape[dim]
    if n != num_v_per_k * num_k_heads * head_dim:
        # nothing to do (already grouped, or an unexpected layout)
        return tensor
    new_shape = shape[:dim] + [num_v_per_k, num_k_heads, head_dim] + shape[dim + 1:]
    t = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return t.permute(*perm).contiguous().reshape(*shape)


def _dequant_plain(t):
    """Dequantize to a PLAIN torch tensor (no GGMLTensor subclass).

    IMPORTANT: never keep the GGMLTensor subclass on a tensor whose data we
    reshape/permute. GGMLTensor carries `tensor_type` / `tensor_shape`
    metadata that does NOT follow reshape/permute/contiguous. A mismatch
    between that metadata and the real storage makes ComfyUI-GGUF's
    GGMLTensor.to() read out of bounds, which shows up as
    "Windows fatal exception: access violation".
    """
    if isinstance(t, GGMLTensor):
        out = dequantize_tensor(t, dtype=torch.float16)
        if isinstance(out, GGMLTensor):
            # strip the subclass (keeps the storage, drops the stale metadata)
            out = out.as_subclass(torch.Tensor)
        return out
    return t


def qwen35_moss_sd_map(raw_sd, metadata=None):
    """Remap llama.cpp Qwen3.5 GGUF keys to ComfyUI core qwen35 keys."""
    out = {}
    for key, val in raw_sd.items():
        if key.startswith("blk."):
            _, layer, leaf = key.split(".", 2)
            mod, _, suffix = leaf.partition(".")
            entry = LEAF_MAP.get(mod)
            if entry is None:
                out[f"model.language_model.layers.{layer}.{leaf}"] = val
                continue
            new_mod, drop_suffix = entry
            if drop_suffix:
                out[f"model.language_model.layers.{layer}.{new_mod}"] = val
            else:
                out[f"model.language_model.layers.{layer}.{new_mod}.{suffix}"] = val
        elif key == "output.weight" and not INCLUDE_LM_HEAD:
            # lm_head is only needed for text generation, which ComfyUI-GGUF
            # cannot do with a quantized lm_head anyway (see INCLUDE_LM_HEAD).
            continue
        elif key in GLOBAL_MAP:
            out[GLOBAL_MAP[key]] = val
        else:
            out[key] = val
    return out


def qwen35_moss_fix_shapes(sd):
    """conv1d: GGML [kernel, channels] -> torch [channels, 1, kernel].

    conv1d is F32 in the GGUF, so the data is already dequantized. The result
    is a plain torch tensor - no GGMLTensor metadata needed.
    """
    for key in list(sd.keys()):
        if key.endswith(".linear_attn.conv1d.weight"):
            t = sd[key]
            shape = list(t.shape)
            if len(shape) == 2:
                # shape is [channels, kernel] in torch convention (reversed ggml ne)
                channels, kernel = shape
                sd[key] = _dequant_plain(t).reshape(channels, 1, kernel)
    return sd


def qwen35_moss_unpermute(sd, metadata):
    """Undo llama.cpp's V-head reorder (tiled -> grouped) on the linear-attn tensors."""
    if not UNPERMUTE_V_HEADS:
        return sd

    num_k_heads = _get_meta(metadata, "qwen35.ssm.group_count")
    num_v_heads = _get_meta(metadata, "qwen35.ssm.time_step_rank")
    inner_size = _get_meta(metadata, "qwen35.ssm.inner_size")
    state_size = _get_meta(metadata, "qwen35.ssm.state_size")

    if not num_k_heads or not num_v_heads or num_k_heads == num_v_heads:
        # no grouping mismatch -> llama.cpp did not reorder either
        return sd

    num_v_per_k = num_v_heads // num_k_heads
    value_head_dim = (inner_size // num_v_heads) if inner_size else (state_size or 128)
    key_dim = num_k_heads * (state_size or 128)  # linear_key_head_dim
    conv_dim = key_dim * 2 + inner_size

    logging.info(
        "Qwen3.5: unpermuting V heads (k=%s, v=%s, v_per_k=%s, v_head_dim=%s)",
        num_k_heads,
        num_v_heads,
        num_v_per_k,
        value_head_dim,
    )

    for key in list(sd.keys()):
        if ".linear_attn." not in key:
            continue
        leaf = key.rsplit(".", 1)[-1]

        try:
            if leaf == "in_proj_qkv.weight":
                # rows: [q(key_dim) | k(key_dim) | v(value_dim)] - only v is permuted
                t = _dequant_plain(sd[key])
                qk = key_dim * 2
                qk_part, v_part = t[:qk], t[qk:]
                v_part = _unpermute_v_heads(v_part, 0, num_k_heads, num_v_per_k, value_head_dim)
                sd[key] = torch.cat([qk_part, v_part], dim=0)
            elif leaf == "in_proj_z.weight":
                t = _dequant_plain(sd[key])
                sd[key] = _unpermute_v_heads(t, 0, num_k_heads, num_v_per_k, value_head_dim)
            elif leaf in ("in_proj_a.weight", "in_proj_b.weight"):
                t = _dequant_plain(sd[key])
                sd[key] = _unpermute_v_heads(t, 0, num_k_heads, num_v_per_k, 1)
            elif leaf in ("dt_bias", "A_log"):
                t = _dequant_plain(sd[key])
                sd[key] = _unpermute_v_heads(t.reshape(-1), 0, num_k_heads, num_v_per_k, 1)
            elif leaf == "conv1d.weight":
                t = _dequant_plain(sd[key])
                # conv1d weight in torch convention: [channels, 1, kernel]
                ch = t.shape[0]
                qk_channels = key_dim * 2
                qk_part, v_part = t[:qk_channels], t[qk_channels:]
                v_part = _unpermute_v_heads(v_part, 0, num_k_heads, num_v_per_k, value_head_dim)
                sd[key] = torch.cat([qk_part, v_part], dim=0).reshape(ch, 1, -1)
            elif leaf == "out_proj.weight":
                # permutation runs along the *input* axis of out_proj, which for
                # k-quants is the quantized axis -> must be dequantized first.
                t = _dequant_plain(sd[key])
                sd[key] = _unpermute_v_heads(t, 1, num_k_heads, num_v_per_k, value_head_dim)
        except Exception as e:
            logging.warning("Qwen3.5: failed to unpermute %s: %s", key, e)

    return sd


def qwen35_moss_prepare(sd, metadata):
    """Full pipeline: remap keys -> fix conv1d shape -> unpermute V heads."""
    sd = qwen35_moss_sd_map(sd)
    sd = qwen35_moss_fix_shapes(sd)
    sd = qwen35_moss_unpermute(sd, metadata)
    return sd
