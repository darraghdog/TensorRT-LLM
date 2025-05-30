# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import json
import os
import traceback
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional

import safetensors
import torch
from transformers.models.auto import AutoModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

import tensorrt_llm.models.modeling_utils
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.qwen.convert import convert_hf_qwen


BASE_MODEL_TLLM_WEIGHT_PREFIX = "base_model."
DRAFTER_TLLM_WEIGHT_PREFIX = "drafter."


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None, required=True)
    parser.add_argument("--drafter_model_dir",
                        type=str,
                        default=None,
                        required=True)

    parser.add_argument("--tp_size",
                        type=int,
                        default=1,
                        help="N-way tensor parallelism size")
    parser.add_argument("--dtype",
                        type=str,
                        default="float16",
                        choices=["float32", "bfloat16", "float16"])

    parser.add_argument("--storage-type",
                        "-t",
                        type=str,
                        default="fp32",
                        choices=["fp32", "fp16"])
    parser.add_argument("--load_model_on_cpu", action="store_true")
    parser.add_argument(
        "--use_parallel_embedding",
        action="store_true",
        default=False,
        help="By default embedding parallelism is disabled.",
    )
    parser.add_argument(
        "--embedding_sharding_dim",
        type=int,
        default=0,
        choices=[0, 1],
        help=
        "By default the embedding lookup table is sharded along vocab dimension (=0). "
        "To shard it along hidden dimension, set embedding_sharding_dim=1"
        "Note: embedding sharing is only enabled when embedding_sharding_dim = 0",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tllm_checkpoint",
        help="The path to save the TensorRT-LLM checkpoint",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="The number of workers for converting checkpoint in parallel",
    )
    parser.add_argument(
        "--dense_context_fmha",
        default=False,
        action="store_true",
        help=
        "Enable dense fmha in context phase, otherwise sliding window attention."
        "If dense_context_fmha=False, the sliding window size is the max attention window size.",
    )

    parser.add_argument(
        "--redrafter_draft_len_per_beam",
        type=int,
        default=5,
        help=
        "Number of times that the Recurrent Drafter runs the beam search to generate draft"
        "candidates. Note that this draft_len does not include the first true/guaranteed token.",
    )
    parser.add_argument(
        "--redrafter_num_beams",
        type=int,
        default=5,
        help="Number of beam search candidates to keep during the Recurrent"
        "Drafter beam search iterations.",
    )
    parser.add_argument(
        "--redrafter_no_greedy_search",
        action="store_false",
        default=True,
        dest="redrafter_greedy_search",
        help=
        "Whether Redrafter will use the token with the highest probability from lm_head"
        "output or randomly sampled from the probability distribution.",
    )

    return parser.parse_args()


def hf_qwen2_config(
    hf_config: Qwen2Config,
    dtype: str = "float32",
    logits_dtype: str = "float32",
    mapping: Mapping = Mapping(1),
) -> tensorrt_llm.models.modeling_utils.PretrainedConfig:
    return tensorrt_llm.models.modeling_utils.PretrainedConfig(
        architecture="Qwen2ForCausalLM",
        dtype=dtype,
        logits_dtype=logits_dtype,
        vocab_size=hf_config.vocab_size,
        max_position_embeddings=hf_config.max_position_embeddings,
        hidden_size=hf_config.hidden_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        hidden_act=hf_config.hidden_act,
        intermediate_size=hf_config.intermediate_size,
        norm_epsilon=hf_config.rms_norm_eps,
        position_embedding_type="rope_gpt_neox",
        mapping=mapping,
        quantization=tensorrt_llm.models.modeling_utils.QuantConfig(),
        rotary_base=getattr(hf_config, "rope_theta", 10000.0),
        rotary_scaling=getattr(hf_config, "rope_scaling", None),
    )


def hf_redrafter_config(
    tllm_base_model_config: tensorrt_llm.models.modeling_utils.PretrainedConfig,
    drafter_config: Namespace,  # DrafterConfig
    redrafter_num_beams: int,
    redrafter_draft_len_per_beam: int,
    redrafter_greedy_search: bool,
) -> tensorrt_llm.models.modeling_utils.PretrainedConfig:
    tllm_config = copy.deepcopy(tllm_base_model_config)

    tllm_config.base_model_architecture = tllm_config.architecture
    tllm_config.architecture = "ReDrafterForQWenLM"
    setattr(tllm_config, "redrafter_num_layers",
            drafter_config.num_draft_layers)
    setattr(tllm_config, "redrafter_hidden_size", drafter_config.hidden_size)
    setattr(tllm_config, "redrafter_exit_dim", drafter_config.exit_dim)
    setattr(tllm_config, "redrafter_is_rnn", drafter_config.rnn)

    # These three configs look like runtime parameters. But for TensorRT-LLM
    # implementation, they are required to be provided at engine build time and
    # TensorRT needs to unroll loops with set number of loop iterations.
    setattr(tllm_config, "redrafter_num_beams", redrafter_num_beams)
    setattr(tllm_config, "redrafter_draft_len_per_beam",
            redrafter_draft_len_per_beam)
    setattr(tllm_config, "redrafter_greedy_search", redrafter_greedy_search)

    return tllm_config


def convert_and_save(
    rank: int,
    tp_size: int,
    hf_base_model: Qwen2ForCausalLM,
    hf_drafter_model: Optional[AutoModel],
    dtype: str,
    use_parallel_embedding: bool,
    embedding_sharding_dim: int,
    output_dir: str,
) -> None:
    mapping = Mapping(
        world_size=tp_size,
        rank=rank,
        tp_size=tp_size,
    )
    weights =  convert_hf_qwen2(
        hf_base_model,
        mapping,
        rank,
        dtype=dtype,
        use_parallel_embedding=use_parallel_embedding,
        sharding_dim=embedding_sharding_dim,
        # use_weight_only=args.use_weight_only,
        # plugin_weight_only_quant_type=plugin_weight_only_quant_type,
        # use_smooth_quant=args.smoothquant,
        # per_channel=args.per_channel,
        # per_token=args.per_token,
        # int8_kv_cache=args.int8_kv_cache,
        # act_range=convert_args['act_range'],
        # qkv_para=convert_args['llama_qkv_para'],
        # smoother=convert_args['llama_smoother']
    )

    if hf_drafter_model is not None:
        drafter_weights = hf_drafter(
            hf_drafter_model,
            mapping,
            dtype=str_dtype_to_torch(dtype),
            additional_tllm_prefix=(DRAFTER_TLLM_WEIGHT_PREFIX
                                    if hf_drafter_model is not None else ""),
        )
        weights.update(drafter_weights)

    safetensors.torch.save_file(
        weights, os.path.join(output_dir, f"rank{rank}.safetensors"))


def multi_worker_convert_and_save(
    workers: int,
    tp_size: int,
    hf_base_model: Qwen2ForCausalLM,
    hf_drafter_model: Optional[AutoModel],
    dtype: str,
    use_parallel_embedding: bool,
    embedding_sharding_dim: int,
    output_dir: str,
) -> None:
    with ThreadPoolExecutor(max_workers=workers) as p:
        futures = [
            p.submit(
                convert_and_save,
                rank,
                tp_size,
                hf_base_model,
                hf_drafter_model,
                dtype,
                use_parallel_embedding,
                embedding_sharding_dim,
                output_dir,
            ) for rank in range(tp_size)
        ]
        exceptions = []
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                traceback.print_exc()
                exceptions.append(e)
        assert len(
            exceptions
        ) == 0, "Checkpoint conversion failed, please check error log."


def create_and_save_config(args):
    mapping = Mapping(
        world_size=args.tp_size,
        tp_size=args.tp_size,
        pp_size=1,
    )
    base_model_hf_config = AutoConfig.from_pretrained(args.model_dir)
    tllm_model_config = hf_qwen2_config(
        base_model_hf_config,
        dtype=args.dtype,
        mapping=mapping,
    )

    if args.drafter_model_dir:
        # TODO: When ReDrafter is added to Transformers
        # drafter_hf_config = AutoConfig.from_pretrained(args.drafter_model_dir)
        with open(Path(args.drafter_model_dir, "config.json")) as fp:
            drafter_hf_config = Namespace(**json.load(fp))
        tllm_model_config = hf_redrafter_config(
            tllm_base_model_config=tllm_model_config,
            drafter_config=drafter_hf_config,
            redrafter_num_beams=args.redrafter_num_beams,
            redrafter_draft_len_per_beam=args.redrafter_draft_len_per_beam,
            redrafter_greedy_search=args.redrafter_greedy_search,
        )

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(tllm_model_config.to_dict(), f, indent=4)
    return drafter_hf_config


def main():
    args = parse_arguments()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    drafter_hf_config = create_and_save_config(args)

    hf_base_model = Qwen2ForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype="auto",
    )

    hf_drafter_model: Optional[AutoModel] = None
    if args.drafter_model_dir:
        # TODO: When ReDrafter is added to Transformers
        # hf_drafter_model = AutoModel.from_pretrained(
        #     args.drafter_model_dir,
        #     torch_dtype="auto",
        # )
        ckpt_file = Path(args.drafter_model_dir, "model.safetensors")
        if not Path.exists(ckpt_file):
            ckpt_file = Path(args.drafter_model_dir, "model.pt")
        print(f"Loading drafter from {ckpt_file}")
        if str(ckpt_file).endswith(".safetensors"):
            drafter_ckpt = {}
            with safetensors.safe_open(ckpt_file, framework="pt",
                                       device="cpu") as f:
                key: str = None
                for key in f.keys():
                    drafter_ckpt[key] = f.get_tensor(key)
        else:
            drafter_ckpt = torch.load(ckpt_file, map_location='cpu')
        hf_drafter_model = Namespace(**{
            "named_parameters": drafter_ckpt,
            "config": drafter_hf_config
        })

    multi_worker_convert_and_save(
        args.workers,
        args.tp_size,
        hf_base_model,
        hf_drafter_model,
        args.dtype,
        args.use_parallel_embedding,
        args.embedding_sharding_dim,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
