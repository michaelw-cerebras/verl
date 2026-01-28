# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Individual Contributor: Michael Wang
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
"""
FSDP Workers for GKD (Generalized Knowledge Distillation) Training.

This module provides FSDP-based workers for on-policy distillation training,
migrated from the Megatron backend for simpler deployment and better LoRA support.
"""

import logging
import os

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

from verl import DataProto
from verl.single_controller.base.decorator import (
    Dispatch,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import (
    DistProfiler,
    GPUMemoryLogger,
    log_gpu_memory_usage,
    simple_timer,
)
from verl.utils.profiler.performance import gather_timing
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.workers.actor import BasePPOActor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def compute_kl_divergence(
    student_logits: torch.Tensor,
    teacher_topk_logps: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    calc_kl_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute KL divergence loss between student and teacher distributions.

    This is a simplified version that doesn't require vocab parallelism,
    suitable for smaller models where the full vocabulary fits on a single GPU.

    Args:
        student_logits: Student model logits, shape [batch, seq_len, vocab_size]
        teacher_topk_logps: Teacher's top-k log probabilities, shape [batch, seq_len, top_k]
        teacher_topk_indices: Indices of teacher's top-k tokens, shape [batch, seq_len, top_k]
        calc_kl_mask: Boolean mask for which positions to compute KL loss, shape [batch, seq_len]

    Returns:
        per_token_kl_loss: KL loss for each token position, shape [batch, seq_len]
    """
    batch_size, seq_len, vocab_size = student_logits.shape
    topk = teacher_topk_indices.shape[-1]

    # Apply mask to get relevant positions
    # student_logits: [batch, seq_len, vocab_size]
    # We need to compute softmax over vocab dimension
    student_log_probs = F.log_softmax(student_logits, dim=-1)  # [batch, seq_len, vocab_size]

    # Gather student log probs at teacher's top-k indices
    # teacher_topk_indices: [batch, seq_len, topk]
    student_topk_log_probs = torch.gather(
        student_log_probs,
        dim=-1,
        index=teacher_topk_indices
    )  # [batch, seq_len, topk]

    # Teacher probabilities from log probs
    teacher_topk_probs = torch.exp(teacher_topk_logps)  # [batch, seq_len, topk]

    # KL(P||Q) = sum_i P(i) * (log P(i) - log Q(i))
    # where P = teacher, Q = student
    # This encourages student to cover all modes of teacher distribution
    per_token_kl_loss = torch.sum(
        teacher_topk_probs * (teacher_topk_logps - student_topk_log_probs),
        dim=-1,
    )  # [batch, seq_len]

    # Zero out masked positions
    per_token_kl_loss = per_token_kl_loss * calc_kl_mask.float()

    return per_token_kl_loss


class OnPolicyDistillDataParallelActor(BasePPOActor):
    """
    FSDP DataParallel Actor for On-Policy Distillation (GKD).

    This actor computes KL divergence loss between student and teacher distributions
    instead of PPO policy gradient loss.

    Args:
        config: Actor configuration
        actor_module: The FSDP-wrapped actor module
        actor_optimizer: The optimizer for the actor
    """

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
    ):
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1
        self.device_name = get_device_name()

        if torch.distributed.get_rank() == 0:
            print(f"[GKD Actor] use_remove_padding={self.use_remove_padding}")
            print(f"[GKD Actor] use_fused_kernels={self.use_fused_kernels}")

    def _forward_micro_batch_for_logits(
        self,
        micro_batch: dict,
    ) -> torch.Tensor:
        """
        Forward pass to get logits for KL divergence computation.

        Args:
            micro_batch: Dictionary containing input_ids, attention_mask, position_ids, etc.

        Returns:
            logits: Model output logits, shape [batch, seq_len, vocab_size]
        """
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            from verl.utils.model import extract_multi_modal_inputs
            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]

            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            # For simplicity, we don't use remove_padding here
            # This can be added later if needed for performance
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )

            logits = output.logits  # [batch, seq_len, vocab_size]

        return logits

    def _optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_module.parameters(), max_norm=self.config.grad_clip
            )

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # Skip update if grad_norm is not finite
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()

        return grad_norm

    @GPUMemoryLogger(role="gkd actor", logger=logger)
    def update_policy(self, data: DataProto) -> dict:
        """
        Update the policy using KL divergence loss against teacher distribution.

        Args:
            data: DataProto containing:
                - input_ids: [batch, seq_len]
                - attention_mask: [batch, seq_len]
                - position_ids: [batch, seq_len]
                - responses: [batch, response_len]
                - teacher_topk_logps: Teacher's top-k log probs (in non_tensor_batch)
                - teacher_topk_indices: Teacher's top-k token indices (in non_tensor_batch)

        Returns:
            metrics: Dictionary of training metrics
        """
        self.actor_module.train()

        # Select relevant keys
        select_keys = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "responses",
        ]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["teacher_topk_logps", "teacher_topk_indices"]
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Get micro batch size from config
        micro_batch_size = self.config.get("micro_batch_size", None)
        use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        if use_dynamic_bsz:
            max_token_len = self.config.get("max_token_len", 4096) * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
            gradient_accumulation = len(micro_batches)
        else:
            if micro_batch_size is None:
                micro_batch_size = len(data.batch["input_ids"])
            micro_batches = data.split(micro_batch_size)
            gradient_accumulation = len(micro_batches)

        metrics = {}
        self.actor_optimizer.zero_grad()

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            micro_batch_metrics = {}

            # Prepare inputs
            model_inputs = {**micro_batch.batch}
            if has_multi_modal_inputs:
                model_inputs["multi_modal_inputs"] = micro_batch.non_tensor_batch["multi_modal_inputs"]

            # Get teacher knowledge
            teacher_topk_logps = torch.tensor(
                micro_batch.non_tensor_batch["teacher_topk_logps"],
                device=get_device_id(),
                dtype=torch.float32,
            )
            teacher_topk_indices = torch.tensor(
                micro_batch.non_tensor_batch["teacher_topk_indices"],
                device=get_device_id(),
                dtype=torch.long,
            )

            # Create KL mask: only compute loss on response tokens
            responses = model_inputs["responses"]
            attention_mask = model_inputs["attention_mask"]
            response_length = responses.size(1)

            calc_kl_mask = attention_mask.clone().bool()
            calc_kl_mask[:, :(-response_length - 1)] = False

            # Forward pass to get logits
            logits = self._forward_micro_batch_for_logits(model_inputs)

            # Compute KL divergence loss
            kl_losses = compute_kl_divergence(
                student_logits=logits,
                teacher_topk_logps=teacher_topk_logps,
                teacher_topk_indices=teacher_topk_indices,
                calc_kl_mask=calc_kl_mask,
            )

            # Aggregate loss
            masked_kl_losses = kl_losses[calc_kl_mask]
            mean_kl_loss = masked_kl_losses.mean()

            # Scale for gradient accumulation
            if use_dynamic_bsz:
                loss_scale_factor = calc_kl_mask.sum().float() / (
                    sum(mb.batch["attention_mask"].sum().item() for mb in micro_batches)
                )
            else:
                loss_scale_factor = 1.0 / gradient_accumulation

            loss = mean_kl_loss * loss_scale_factor
            loss.backward()

            micro_batch_metrics["actor/kl_loss"] = mean_kl_loss.detach().item() * loss_scale_factor
            append_to_dict(metrics, micro_batch_metrics)

        # Optimizer step
        grad_norm = self._optimizer_step()
        metrics["actor/grad_norm"] = grad_norm.detach().item()

        self.actor_optimizer.zero_grad()
        get_torch_device().empty_cache()

        return metrics


class FSDPOnPolicyDistillActorWorker(ActorRolloutRefWorker):
    """
    FSDP Actor Worker for On-Policy Distillation (GKD).

    This worker owns the trainable FSDP model and optimizer,
    and performs update_actor using KL divergence loss.
    """

    def __init__(self, config: DictConfig, role: str):
        # Ensure we run as actor-only worker
        super().__init__(config, role)
        assert self._is_actor and not self._is_rollout, "Actor worker must be actor-only."

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize the FSDP model and optimizer."""
        from verl.utils.import_utils import import_external_libs
        from verl.utils.fs import copy_to_local
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config import FSDPEngineConfig

        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(
            OmegaConf.create(self.config.model.get("override_config", {}))
        )
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        optim_config = self.config.actor.optim
        fsdp_config = omega_conf_to_dataclass(self.config.actor.fsdp_config, dataclass_type=FSDPEngineConfig)

        local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
        log_gpu_memory_usage("Before init actor model and optimizer", logger=logger)

        (
            self.actor_module_fsdp,
            self.actor_optimizer,
            self.actor_lr_scheduler,
            self.actor_model_config,
        ) = self._build_model_optimizer(
            model_path=local_path,
            fsdp_config=fsdp_config,
            optim_config=optim_config,
            override_model_config=override_model_config,
            use_remove_padding=use_remove_padding,
            use_fused_kernels=use_fused_kernels,
            enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
            trust_remote_code=self.config.model.get("trust_remote_code", False),
            use_liger=self.config.model.get("use_liger", False),
            role="actor",
            enable_activation_offload=self.config.model.get("enable_activation_offload", False),
        )

        # Create distillation actor
        self.actor = OnPolicyDistillDataParallelActor(
            config=self.config.actor,
            actor_module=self.actor_module_fsdp,
            actor_optimizer=self.actor_optimizer,
        )

        log_gpu_memory_usage("After OnPolicyDistillDataParallelActor init", logger=logger)

        self.flops_counter = FlopsCounter(self.actor_model_config)

        # Checkpoint manager
        from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.actor_module_fsdp,
            optimizer=self.actor_optimizer,
            lr_scheduler=self.actor_lr_scheduler,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            checkpoint_config=self.config.actor.checkpoint,
        )

        get_torch_device().empty_cache()
        log_gpu_memory_usage("Actor init_model finished", logger=logger)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @GPUMemoryLogger(role="update_actor", logger=logger)
    @DistProfiler.annotate(color="red")
    def update_actor(self, data: DataProto):
        """Update the actor using KL divergence loss."""
        assert self._is_actor and not self._is_rollout

        with Timer(name="update_policy", logger=None) as timer:
            metrics = self.actor.update_policy(data=data)

        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics["perf/mfu/actor"] = estimated_flops / promised_flops / self.world_size
        metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
        metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

        lr = self.actor_lr_scheduler.get_last_lr()[0]
        metrics["actor/lr"] = lr.item() if torch.is_tensor(lr) else lr
        self.actor_lr_scheduler.step()

        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_actor_weights_info(self):
        """Get actor weights info for synchronization with rollout worker."""
        if hasattr(self, "_weights_info"):
            return self._weights_info

        from verl.utils.model import convert_weight_keys

        # Get state dict
        params = self.actor_module_fsdp.state_dict()
        params = convert_weight_keys(
            params,
            getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        )

        ret = []
        for key, tensor in params.items():
            ret.append((key, tensor.size(), tensor.dtype))

        self._weights_info = ret
        return ret

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self):
        """Synchronize weights from actor to rollout worker."""
        assert self._is_actor
        assert hasattr(self, "_weights_info") and self._weights_info is not None

        from ray.util.collective import collective
        from verl.utils.model import convert_weight_keys
        from torch.distributed.tensor import DTensor

        # Get state dict
        params = self.actor_module_fsdp.state_dict()
        params = convert_weight_keys(
            params,
            getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        )

        device = get_device_id()

        for key, shape, dtype in self._weights_info:
            weight = params[key]
            if isinstance(weight, DTensor):
                weight = weight.full_tensor()
            weight = weight.to(device)
            collective.broadcast(weight, src_rank=0, group_name="actor_rollout")


class FSDPOnPolicyDistillRolloutWorker(ActorRolloutRefWorker):
    """
    FSDP Rollout Worker for On-Policy Distillation (GKD).

    This worker owns the inference engine (vLLM/SGLang) and generates sequences.
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__(config, role)
        assert self._is_rollout and not self._is_actor, "Rollout worker must be rollout-only."

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize the rollout model."""
        from verl.utils.import_utils import import_external_libs
        from verl.utils.fs import copy_to_local
        from verl.utils.config import omega_conf_to_dataclass
        from verl.utils.model import get_generation_config
        from verl.workers.config import RolloutConfig, HFModelConfig
        from verl.workers.rollout import get_rollout_class
        from torch.distributed.device_mesh import init_device_mesh

        import_external_libs(self.config.model.get("external_lib", None))

        local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))

        # Load tokenizer
        from verl.utils import hf_tokenizer, hf_processor
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=self.config.model.get("trust_remote_code", False))
        self.processor = hf_processor(local_path, trust_remote_code=self.config.model.get("trust_remote_code", False))
        self.generation_config = get_generation_config(local_path, trust_remote_code=self.config.model.get("trust_remote_code", False))

        log_gpu_memory_usage("Before init rollout model", logger=logger)

        # Build rollout device mesh
        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)

        infer_tp = self.config.rollout.tensor_model_parallel_size * self.config.rollout.data_parallel_size
        infer_pp = self.config.rollout.pipeline_model_parallel_size
        infer_world_size = infer_tp * infer_pp
        dp = self.world_size // infer_world_size

        assert self.world_size % infer_world_size == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
        )

        rollout_device_mesh = init_device_mesh(
            get_device_name(),
            mesh_shape=(dp, infer_tp, infer_pp),
            mesh_dim_names=["dp", "infer_tp", "infer_pp"],
        )

        rollout_name = self.config.rollout.name
        if rollout_name == "hf":
            self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)
        else:
            is_collect = (
                rollout_device_mesh["infer_tp"].get_local_rank() == 0
                and rollout_device_mesh["infer_pp"].get_local_rank() == 0
            )
            self._register_dispatch_collect_info(
                "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )

        # Build rollout model
        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
        )
        self.rollout_device_mesh = rollout_device_mesh

        log_gpu_memory_usage("After rollout init", logger=logger)
        get_torch_device().empty_cache()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @GPUMemoryLogger(role="generate_sequences", logger=logger)
    @DistProfiler.annotate(color="red")
    def generate_sequences(self, prompts: DataProto):
        """Generate sequences using the rollout engine."""
        assert self._is_rollout and not self._is_actor
        prompts = prompts.to(get_device_id())

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        timing_generate = {}
        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)

        timing_generate = gather_timing(timing_generate)
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"), blocking=False)
    def async_generate_sequences(self, *args, **kwargs):
        """Asynchronous sequence generation."""
        return self.generate_sequences(*args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_weights_info(self, weights_info):
        """Set actor weights info for synchronization."""
        assert self._is_rollout
        self._weights_info = weights_info

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self):
        """Receive synchronized weights from actor worker."""
        import asyncio
        from ray.util.collective import collective

        assert self._is_rollout
        assert hasattr(self, "_weights_info") and self._weights_info is not None

        rollout_name = self.config.rollout.name

        if rollout_name == "vllm":
            inference_model = (
                self.rollout.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
            )
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
            patch_vllm_moe_model_weight_loader(inference_model)
        elif rollout_name == "sglang":
            from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights
            inference_model = self.rollout._engine

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            async def update_weights(inference_engine, params):
                await sgl_update_weights(
                    engine=inference_engine,
                    params_batch=params,
                    device_mesh_key="infer_tp",
                    device_mesh=self.rollout_device_mesh,
                )
                if self.rollout_device_mesh["infer_tp"].get_local_rank() == 0:
                    await inference_engine.flush_cache()
        else:
            raise NotImplementedError(f"Unknown rollout name: {rollout_name}")

        device = get_torch_device().current_device()
        params_to_load = []

        for key, shape, dtype in self._weights_info:
            tensor = torch.empty(shape, dtype=dtype, device=device)
            collective.broadcast(tensor, src_rank=0, group_name="actor_rollout")
            params_to_load.append((key, tensor))

        if rollout_name == "vllm":
            inference_model.load_weights(params_to_load)
        elif rollout_name == "sglang":
            loop.run_until_complete(update_weights(inference_model, params_to_load))
