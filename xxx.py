# grpo_vllm_weight_sync.py
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Any

import ray
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine
from vllm.utils.network_utils import get_ip, get_open_port

# -----------------------------
# Config
# -----------------------------
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # keep one model family for training/inference
DTYPE = torch.bfloat16

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

@dataclass
class GRPOConfig:
    num_rollouts: int = 4          # K rollouts per prompt
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95
    lr: float = 1e-6
    beta_kl: float = 0.02          # GRPO KL term weight (tune)
    clip_eps: float = 0.2          # GRPO clip (tune)
    packed_broadcast: bool = True  # efficient NCCL transfer

CFG = GRPOConfig()


# -----------------------------
# Ray actors
# -----------------------------
@ray.remote(num_gpus=1)
class HFTrainer:
    """
    Owns the training policy (HF model) on GPU:0 and can broadcast
    updated weights to the vLLM engine using NCCLWeightTransferEngine.
    """
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # Make sure pad_token is set for batching if needed
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=DTYPE,
        ).to("cuda:0")
        self.model.train()

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=CFG.lr)

        # NCCL rendezvous info for trainer<->vLLM
        self.master_address = get_ip()
        self.master_port = get_open_port()

    def get_master_address_and_port(self):
        return self.master_address, self.master_port

    def get_weight_metadata(self):
        """Return weight names, dtype strings, shapes (vLLM uses this for update_weights)."""
        names, dtype_names, shapes = [], [], []
        for name, p in self.model.named_parameters():
            names.append(name)
            dtype_names.append(str(p.dtype).split(".")[-1])
            shapes.append(list(p.shape))
        return names, dtype_names, shapes

    def init_weight_transfer_group(self, world_size: int):
        """Initialize NCCL group on the trainer side."""
        self.model_update_group = NCCLWeightTransferEngine.trainer_init(
            dict(
                master_address=self.master_address,
                master_port=self.master_port,
                world_size=world_size,
            )
        )

    def broadcast_weights(self, packed: bool = True):
        """Send current HF weights to vLLM workers."""
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=self.model.named_parameters(),
            group=self.model_update_group,
            packed=packed,
        )

    # -------- GRPO-ish training step (skeleton) --------
    def grpo_update(
        self,
        prompt_texts: List[str],
        completion_texts: List[str],
        rewards: torch.Tensor,
        old_logps: torch.Tensor,
    ):
        """
        Minimal GRPO-like update skeleton.

        Inputs:
          - prompt_texts: len = B*K (prompts replicated per rollout)
          - completion_texts: len = B*K
          - rewards: (B*K,) float tensor
          - old_logps: (B*K,) float tensor (log p_old of sampled completion under old policy)

        This computes:
          - logp_new of sampled completion under current HF model
          - ratio = exp(logp_new - logp_old)
          - clipped policy objective + KL penalty (very simplified)
        """
        device = "cuda:0"
        self.model.train()
        self.opt.zero_grad(set_to_none=True)

        # Tokenize full sequences = prompt + completion
        full_texts = [p + c for p, c in zip(prompt_texts, completion_texts)]
        enc = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]

        # Compute token logprobs
        out = self.model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits[:, :-1, :]                # (N, T-1, V)
        tgt = input_ids[:, 1:]                        # (N, T-1)
        tgt_attn = attn[:, 1:]                        # (N, T-1)

        logp = F.log_softmax(logits, dim=-1)
        tok_logp = torch.gather(logp, dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)  # (N, T-1)
        seq_logp_new = (tok_logp * tgt_attn).sum(dim=-1)                            # (N,)

        # Advantage: normalize rewards per prompt-group if you want (left simple)
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        # GRPO ratio + clip
        ratio = torch.exp(seq_logp_new - old_logps.to(device))
        unclipped = ratio * adv.to(device)
        clipped = torch.clamp(ratio, 1.0 - CFG.clip_eps, 1.0 + CFG.clip_eps) * adv.to(device)
        policy_obj = torch.min(unclipped, clipped).mean()

        # Very rough KL penalty (optional; you can compute proper KL against a ref model)
        # Here we approximate KL via (logp_new - logp_old) just as a placeholder.
        approx_kl = (seq_logp_new - old_logps.to(device)).mean()
        loss = -(policy_obj - CFG.beta_kl * approx_kl)

        loss.backward()
        self.opt.step()

        return {
            "loss": float(loss.detach().cpu()),
            "policy_obj": float(policy_obj.detach().cpu()),
            "approx_kl": float(approx_kl.detach().cpu()),
        }


# vLLM actor pinned to a separate GPU
@ray.remote(num_gpus=1)
class RolloutEngine:
    """
    Owns vLLM engine on GPU:1 and exposes:
      - generate()
      - init_weight_transfer_engine()
      - update_weights()
      - get_world_size()
    """
    def __init__(self, model_name: str):
        # Important: vLLM will occupy this GPU; keep HF model off it.
        self.llm = LLM(
            model=model_name,
            enforce_eager=True,
            tensor_parallel_size=1,
            data_parallel_size=1,
            distributed_executor_backend="ray",
            weight_transfer_config=WeightTransferConfig(backend="nccl"),
            load_format="dummy",   # start with dummy/random weights, then sync from HF
            dtype="bfloat16",
        )

    def get_world_size(self):
        # vLLM world size for weight transfer == number of vLLM ranks
        # with TP=1, DP=1 => 1
        return 1

    def init_weight_transfer_engine(self, kwargs: Dict[str, Any]):
        return self.llm.init_weight_transfer_engine(kwargs)

    def update_weights(self, kwargs: Dict[str, Any]):
        return self.llm.update_weights(kwargs)

    def generate(self, prompts: List[str], sampling_params: SamplingParams):
        return self.llm.generate(prompts, sampling_params)


# -----------------------------
# Reward function (placeholder)
# -----------------------------
def simple_reward_fn(text: str) -> float:
    """
    Replace with your real reward model / verifier.
    For demo: reward longer (but not too long) and penalize empty.
    """
    n = len(text.strip())
    if n == 0:
        return -1.0
    return min(n / 200.0, 1.0)


# -----------------------------
# Main: GRPO loop
# -----------------------------
def main():
    ray.init()

    trainer = HFTrainer.remote(MODEL)
    rollout = RolloutEngine.remote(MODEL)

    # 1) Set up NCCL weight transfer
    master_address, master_port = ray.get(trainer.get_master_address_and_port.remote())

    world_size = ray.get(rollout.get_world_size.remote()) + 1  # +1 trainer rank
    ray.get([
        trainer.init_weight_transfer_group.remote(world_size),
        rollout.init_weight_transfer_engine.remote(dict(
            init_info=dict(
                master_address=master_address,
                master_port=master_port,
                rank_offset=1,
                world_size=world_size,
            )
        )),
    ])

    # 2) Tell vLLM what weights to expect (names/dtypes/shapes), then broadcast weights
    names, dtype_names, shapes = ray.get(trainer.get_weight_metadata.remote())
    ray.get([
        rollout.update_weights.remote(dict(
            update_info=dict(
                names=names,
                dtype_names=dtype_names,
                shapes=shapes,
                packed=CFG.packed_broadcast,
            )
        )),
        trainer.broadcast_weights.remote(packed=CFG.packed_broadcast),
    ])

    # 3) Rollout + GRPO updates
    for step in range(10):
        # replicate prompts for K rollouts each
        prompts_rep = []
        for p in PROMPTS:
            prompts_rep.extend([p] * CFG.num_rollouts)

        sampling = SamplingParams(
            temperature=CFG.temperature,
            top_p=CFG.top_p,
            max_tokens=CFG.max_new_tokens,
        )

        t0 = time.time()
        outputs = ray.get(rollout.generate.remote(prompts_rep, sampling))
        gen_texts = [o.outputs[0].text for o in outputs]

        # Compute rewards
        rewards = torch.tensor([simple_reward_fn(t) for t in gen_texts], dtype=torch.float32)

        # IMPORTANT for real GRPO:
        # old_logps should be the logprob of the sampled completion under the *old* policy.
        # You can get this by having vLLM return prompt+completion token logprobs, or by
        # running the HF model in eval mode before the update to compute logps.
        # Here, we fake it (placeholder) to show the dataflow.
        old_logps = torch.zeros(len(gen_texts), dtype=torch.float32)

        # Update HF policy
        stats = ray.get(trainer.grpo_update.remote(prompts_rep, gen_texts, rewards, old_logps))

        # Sync updated weights HF -> vLLM
        ray.get([
            rollout.update_weights.remote(dict(
                update_info=dict(
                    names=names,
                    dtype_names=dtype_names,
                    shapes=shapes,
                    packed=CFG.packed_broadcast,
                )
            )),
            trainer.broadcast_weights.remote(packed=CFG.packed_broadcast),
        ])

        dt = time.time() - t0
        print(f"[step {step}] dt={dt:.2f}s stats={stats} reward_mean={rewards.mean().item():.3f}")

    ray.shutdown()


if __name__ == "__main__":
    # keep this (spawn-safe + vLLM + ray)
    main()
