# RL0: RLVR with 0 training data, only rollouts.
# Here we try to teach the model to perform 2 digit addition by trial and error.
# The model is given prompts of the form "25+24=" and has to fill in the next tokens.
# The next tokens can be digits and numbers.
# For example, the model might fill in the next tokens as "48" or "49=" or "49+" etc.
# The model is then rewarded for the correct answer using a reward function - Reinforcement Learning with Verifiable Rewards
# See GRPO, Dr GRPO, DAPO and Tulu for more context:
# Dr GRPO: https://arxiv.org/pdf/2503.20783
# DAPO: https://arxiv.org/pdf/2503.14476
# TULU: https://arxiv.org/pdf/2411.15124

import os
import copy
import random
import time
import pandas as pd
import concurrent.futures as futures

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from model import SmallGPT
from config import config
import generate
from dataclasses import dataclass
import argparse
import rewards_math
import gencache
import html2  # local formatting

# Command line args
parser = argparse.ArgumentParser(description="MATH0")
parser.add_argument("--ckpt", default="", help="Path to base checkpoint (.pt)")
parser.add_argument("--suffix", default="", help="To distinguish runs")
parser.add_argument("--notes", default="", help="Notes about the run for logging")
parser.add_argument("--cache", default="", help="Cache of good completions from previous runs")
args = parser.parse_args()

# Tokenizer
# To make things as simple as possible, we build our own tokenizer for the task, which just splits the
# inputs into characters. This approach is more determistic and robust for small data,
# compared to SentencePiece + BPE.
import mathtokenizer
sp = mathtokenizer.mathtokenizer()
pad_id = sp.pad_id()
bos_id = sp.bos_id()
eos_id = sp.eos_id()
seq_len = config['block_size']

# RL configs. A lot of knobs to get this right.
# See 37 lessons from PPO here:
# https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
@dataclass
class GRPOConfig:
    total_updates:  int = 1000  # num roll outs

    num_prompts: int = 100   # number of distinct prompt inputs per rollout
    group_size:  int = 2     # num samples per prompt
    grpo_epochs:    int = 4  # num gradient steps per rollout
    minibatch_size: int = num_prompts * group_size

    max_prompt_len:int = 8
    max_new_tokens: int = seq_len - max_prompt_len - 1  # to account for the prompt

    ent_coef:    float = 0.01  # higher = prefer newer trajectories
    entropy_schedule = 100

    temperature: float = 0.8   # lower = be close to inference time decisions
    top_p:       float = 0.9
    clip_high:float = 2.0  # PPO clip - can multiply advantage by (1+clip_high)
    clip_low:float  = 0.6  # PPO clip - can multiply advantage by (1-clip_low)

    no_kl:          bool = False  # if we dont care about the base model
    kl_coef_init:  float = 0.1  # higher = prefer being close to base model
    target_kl:     float = 0.12
    kl_adapt_rate: float = 2.0
    kl_high_factor:float = 2.0
    grad_clip:     float = 1.0

    group_mean_centered: bool = True
    batch_mean_centered: bool = True
    batch_var_adjusted:  bool = True

    cache_maxsize: int = 100
    table_logging_frequency:int = 25
    print_samples_frequency:int = 10
    num_tables_to_print:int = 4

    def reward_fn(self, s):
        return rewards_math.reward_fn_hardest(s)

cfg = GRPOConfig()

debug_mode = False
if debug_mode:
    cfg.logging_frequency = 2
    cfg.total_updates = 10
    cfg.num_prompts = 10
    cfg.group_size = 2

# Printing of tensors
torch.set_printoptions(precision=3, linewidth=120, threshold=10000, sci_mode=False)

# For plotting:
updates_log = []
steps_log = []
rewards_log = []
percent_highest_log = []
kls_log = []
entropies_log = []
gens_log = []
total_losses_log = []
policy_losses_log = []

prints = []  # html output

if len(args.notes) > 0:
    notes = f"<div class='gray'>{args.notes}</div>"
    prints = [notes] + prints

def get_chart_data():
    out = ""
    out += f"const updates = {updates_log};\n"
    out += f"const steps = {steps_log};\n"
    out += f"const rewards = {rewards_log};\n" 
    out += f"const percent_highest_log = {percent_highest_log};\n"
    out += f"const gens = {gens_log};\n"
    out += f"const KLs = {kls_log};\n"
    out += f"const entropies = {entropies_log};\n"
    out += f"const total_losses = {total_losses_log};\n"
    out += f"const policy_losses = {policy_losses_log};\n"
    out += "const charts = ["
    out += '  {name: "Rewards", x: updates, y: rewards, xLabel: "Updates", yLabel: ""},\n'
    out += '  {name: "Accuracy", x: updates, y: percent_highest_log, xLabel: "Updates", yLabel: ""},\n'
    out += '  {name: "Total loss", x: steps, y: total_losses, xLabel: "Steps", yLabel: ""},\n'
    out += '  {name: "Policy loss", x: steps, y: policy_losses, xLabel: "Steps", yLabel: ""},\n'
    out += '  {name: "KL*10", x: steps, y: KLs, xLabel: "Steps", yLabel: ""},\n'
    out += '  {name: "Entropy", x: steps, y: entropies, xLabel: "Steps", yLabel: ""},\n'
    out += '  {name: "#gens", x: updates, y: gens, xLabel: "Updates", yLabel: ""},\n'
    out += '];\n'
    return out

# writes to log+suffix.html
def printhtml(s, color="black", ignore_terminal=False, add_span=True, reward=None, reward_mid=0.0):
    global prints
    if not ignore_terminal:
        print(s)

    if reward is not None:
        color = "green" if reward >= reward_mid else "red"
    if add_span:
        s = str(s).replace("\n", "<br>")
        s = f"<span class='{color}'>{s}</span>"
    else:
        s = f"<div class='{color}'>{s}</div>"
    
    prints = [s] + prints    
    title = args.suffix + " : " + args.notes
    chartdata = get_chart_data()
    logsdata = "<br>".join(prints)
    out = html2.header + \
          title + \
          html2.header2 + \
          title + "<br>" + str(cfg) + \
          html2.header3 + \
          chartdata +    \
          html2.charthtml + "<br>" + logsdata + "<br></body></html>"
    with open(f"mathlogs{args.suffix}.html", "w") as f:
        f.write(out)

def savemodel(m1, name):
    os.makedirs("checkpoints", exist_ok=True)
    path = f"checkpoints/rl0-{name}-{args.suffix}.pt"
    torch.save(m1.state_dict(), path)
    printhtml(f"💾 Saved model in {path}")

# Device pick (prefer MPS for your macbooks)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# =============== Model ===============
model = None
optimizer = None

# vectorized nucleus sampling (top-p) with temperature
def _sample_top_p(logits: torch.Tensor, top_p: float = 0.95, temperature: float = 0.9):
    # logits: (B, V)
    if temperature is not None and temperature > 0 and temperature != 1.0:
        logits = logits / max(1e-8, float(temperature))
    probs = torch.softmax(logits, dim=-1)  # (B, V)

    # sort by prob desc
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    keep = cumsum <= top_p
    # ensure at least one token kept
    keep[:, 0] = True

    # mask and renormalize
    masked = sorted_probs * keep
    denom = masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    masked = masked / denom

    # sample in sorted space, then map back to vocab ids
    next_sorted = torch.multinomial(masked, num_samples=1)  # (B,1)
    next_ids = torch.gather(sorted_idx, 1, next_sorted).squeeze(1)  # (B,)
    return next_ids

def print_roll(roll):
    nn = len(roll['generations'])
    for i in range(nn):
        printhtml(f"{i}:")
        r = roll["rewards_scalar"][i]
        for k in ["prompts", "generations", "input_ids", "advantages", "prompt_mask",
                  "response_mask", "group_ids", "rewards_scalar"]:
            printhtml(f"{k}: {roll[k][i]}", reward=r)
        printhtml("-------------------")

reason_for_none_rolls = {} # for debugging

# the RLVR step: sample and verify (assign reward).
def get_rolls(old_model, num_prompts=cfg.num_prompts, group_size=cfg.group_size,
              max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature,
              top_p=cfg.top_p, reindex_groups: bool = True):    
    prompts = random.sample(rewards_math.train_prompts, num_prompts)

    enc_prompts = [generate.special_encode(p, sp) for p in prompts]
    if len(enc_prompts) == 0:
        reason_for_none_rolls["no_encoded_prompts"] += 1
        return None

    enc_prompts_per_group = []  # = num_prompts * #groups
    group_ids = []
    for i, p_ids in enumerate(enc_prompts):
        for _ in range(group_size):
            enc_prompts_per_group.append(p_ids)
            group_ids.append(i)

    B = len(enc_prompts_per_group)
    if B == 0:
        reason_for_none_rolls["B_is_0"] += 1
        return None

    prompt_lengths = torch.tensor([len(x) for x in enc_prompts_per_group], device=device, dtype=torch.long)
    max_prompt_len = int(prompt_lengths.max().item())
    # Pre-allocate max prompt + max_new_tokens + 1 (room for EOS)
    T_max = max_prompt_len + max_new_tokens + 1

    input_ids = torch.full((B, T_max), pad_id, dtype=torch.long, device=device)

    # pre-fill encoded prompt ids
    for i, pids in enumerate(enc_prompts_per_group):
        L = len(pids)
        if L > 0:
            input_ids[i, :L] = torch.tensor(pids, dtype=torch.long, device=device)

    # track per-row current length (cursor at last written position + 1)
    cur_len = prompt_lengths.clone()            # where next token will be written
    alive   = torch.ones(B, dtype=torch.bool, device=device)
    gen_len = torch.zeros(B, dtype=torch.long, device=device)  # excludes EOS

    # autoregressive parallel generation for all rows
    for step in range(max_new_tokens):
        if not alive.any():
            break

        # run model up to the last real token per row
        max_used = min(seq_len, int(cur_len.max().item()))
        logits = old_model(input_ids[:, :max_used])          # (B, max_used, V)
        # Next-token logits are at index (cur_len-1) for each row
        idx = cur_len - 1
        next_logits = logits[torch.arange(B, device=device), idx]  # (B, V)

        # sample only for alive rows
        next_tokens = torch.full((B,), eos_id, dtype=torch.long, device=device)
        temp = temperature # + random.randint(-4, 4)*0.1
        if alive.any():
            chosen = _sample_top_p(next_logits[alive], top_p=top_p, temperature=temp)
            next_tokens[alive] = chosen

        # write next token
        input_ids[torch.arange(B, device=device), cur_len] = next_tokens

        # update gen_len if not EOS
        not_eos = (next_tokens != eos_id) & alive
        gen_len[not_eos] += 1

        # rows that just ended
        ended = (next_tokens == eos_id) & alive
        alive = alive & (~ended)

        # advance cursor for rows that received a token (all alive+ended received one)
        wrote = (next_tokens != pad_id)  # should be all True here
        cur_len[wrote] += 1

    # if still alive after budget, force an EOS at cursor (already safe to overwrite)
    if alive.any():
        input_ids[torch.arange(B, device=device)[alive], cur_len[alive]] = eos_id
        # gen_len unchanged; eos not counted
    keep = (gen_len >= 1)
    if not keep.any():
        reason_for_none_rolls["!keep.any"] += 1
        return None

    input_ids   = input_ids[keep]
    prompt_lengths = prompt_lengths[keep]
    gen_len     = gen_len[keep]
    group_ids   = torch.tensor(group_ids, device=device, dtype=torch.long)[keep]

    used_len = (prompt_lengths + gen_len + 1)
    T = int(used_len.max().item())
    input_ids = input_ids[:, :T].contiguous()

    # attention mask, prompt/response masks
    ar = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
    prompt_mask   = (ar < prompt_lengths.unsqueeze(1)).float()
    response_mask = ((ar >= prompt_lengths.unsqueeze(1)) &
                     (ar <  (used_len).unsqueeze(1))).float()
                     #(ar <  (prompt_lengths + gen_len).unsqueeze(1))).float()

    # Decode texts as (prompt_text, response_text)
    prompts_and_completions = []
    token_rewards_input = []
    input_ids_cpu = input_ids.detach().cpu()
    for i in range(input_ids_cpu.size(0)):
        p_text = prompts[group_ids[i].item()]  # original prompt string
        ids = input_ids_cpu[i].tolist()
        genchars = [generate.special_decode(ids[j], sp) for j in range(len(ids))]
        token_rewards_input.append((ids, p_text, genchars, prompt_lengths[i].item()))

    rewards_vector = []
    with futures.ThreadPoolExecutor() as ex:
        rs = list(ex.map(rewards_math.reward_fn_per_token, token_rewards_input))
    rewards_vector = torch.tensor([r[0] for r in rs],
                                  device=device, dtype=torch.float32)
    correct_answers = [r[1] for r in rs]

    for i in range(input_ids_cpu.size(0)):
        p_text = prompts[group_ids[i].item()]  # original prompt string
        ids = input_ids_cpu[i].tolist()
        prompts_and_completions.append( (p_text, sp.decode(ids), correct_answers[i]) )

    # group mean baseline
    # a "group" is a bunch of generations for the same prompt
    # if all the generations are terrible, then we have a couple of choices:
    # 1. drop the group. in which case we don't learn anything useful from the generations.
    # 2. rank the group and reward the top ones (even if they are terrible)
    #    in the hope that the model will learn to generate the correct answer later.
    # configs like group_mean_centered and group_var_adjusted are used to balance these tradeoffs.
    unique_gids, inv = torch.unique(group_ids, return_inverse=True)  # inv: (B,) maps row->group index [0..G-1]
    G = unique_gids.numel()
    inv_T = inv.unsqueeze(1).expand(-1, T)
    resp_mask_f = response_mask.float()

    """
    This is how it looks:
    Rewards_vector: tensor([[ 0.390,  0.390,  0.390,  0.390,  0.390,  0.390,  0.390,  0.390,  0.390,  0.390],
            [ 7.022,  7.022,  7.022,  7.022,  7.022,  7.022,  7.022,  7.022,  7.022,  7.022],
            [21.875, 21.875, 21.875, 21.875, 21.875, 21.875, 21.875, 21.875, 21.875, 21.875],
            [ 0.390,  0.390,  0.390,  0.390,  0.390,  0.390, 15.625,  0.390,  0.390,  0.390]], device='mps:0')
    groupids: tensor([0, 0, 1, 1], device='mps:0')
    inv_t: tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device='mps:0')
    rmask: tensor([[0., 0., 0., 0., 0., 1., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0., 1., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 1., 0., 0.]], device='mps:0')
    sums: tensor([[ 0.000,  0.000,  0.000,  0.000,  0.000,  7.412,  7.412,  0.390,  0.390,  0.000],
            [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000, 37.500, 22.265,  0.000,  0.000]], device='mps:0')
    counts: tensor([[0., 0., 0., 0., 0., 2., 2., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0., 2., 2., 0., 0.]], device='mps:0')
    means: tensor([[ 0.000,  0.000,  0.000,  0.000,  0.000,  3.706,  3.706,  0.390,  0.390,  0.000],
            [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000, 18.750, 11.132,  0.000,  0.000]], device='mps:0')
    means_per_row: tensor([[ 0.000,  0.000,  0.000,  0.000,  0.000,  3.706,  3.706,  0.390,  0.390,  0.000],
            [ 0.000,  0.000,  0.000,  0.000,  0.000,  3.706,  3.706,  0.390,  0.390,  0.000],
            [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000, 18.750, 11.132,  0.000,  0.000],
            [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000, 18.750, 11.132,  0.000,  0.000]], device='mps:0')
    centered = tensor([[  0.000,   0.000,   0.000,   0.000,   0.000,  -3.316,  -3.316,   0.000,   0.000,   0.000],
            [  0.000,   0.000,   0.000,   0.000,   0.000,   3.316,   3.316,   0.000,   0.000,   0.000],
            [  0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   3.125,  10.743,   0.000,   0.000],
            [  0.000,   0.000,   0.000,   0.000,   0.000,   0.000,  -3.125, -10.742,   0.000,   0.000]], device='mps:0')
    idx: tensor([0, 1, 2, 3], device='mps:0')    
    """

    centered = rewards_vector * response_mask
    if cfg.group_mean_centered:
        sums = torch.zeros((G, T), device=device, dtype=torch.float32).scatter_add_(0, inv_T, rewards_vector * resp_mask_f)
        counts = torch.zeros((G, T), device=device, dtype=torch.float32).scatter_add_(0, inv_T, resp_mask_f)
        means = sums / counts.clamp_min(1.0)         
        means_per_row = means[inv, :]
        centered = (rewards_vector - means_per_row) * response_mask

    abs_token = centered.abs()
    group_var = torch.zeros(G, device=device, dtype=torch.float32).scatter_add_(
        0, inv, abs_token.sum(dim=1)
    )
    keep_group = group_var > -1  # keep all
    if cfg.group_mean_centered:
        # --- 3) DROP GROUPS with no variability (token-wise) ---
        keep_group = group_var > float(1e-4)

    keep_mask = keep_group[inv]    # (B,)
    if not keep_mask.any():
        reason_for_none_rolls["empty_after_norm"] += 1
        # nothing useful this round
        return None

    idx = keep_mask.nonzero(as_tuple=False).squeeze(-1)

    # filter everything consistently
    input_ids     = input_ids[idx]
    prompt_mask   = prompt_mask[idx]
    response_mask = response_mask[idx]
    group_ids     = group_ids[idx]
    gen_lengths   = gen_len[idx]
    rewards_vector = rewards_vector[idx]
    centered       = centered[idx]
    prompts_and_completions          = [prompts_and_completions[i] for i in idx.tolist()]

    # optional: reindex group ids to 0..K-1 for downstream neatness
    if reindex_groups:
        kept_unique = torch.unique(group_ids)
        mapping = {int(g.item()): i for i, g in enumerate(kept_unique)}
        group_ids = torch.tensor([mapping[int(g.item())] for g in group_ids], device=device, dtype=torch.long)

    advantages  = centered * response_mask
    #print(f"avs: {advantages}")
    rewards_scalar = (advantages.sum(-1) / response_mask.sum(-1).clamp_min(1)).detach()

    return {
        "prompts":      [p for (p, _, _) in prompts_and_completions],
        "generations":  [g for (_, g, _) in prompts_and_completions],
        "answers":      [p for (_, _, p) in prompts_and_completions],
        "input_ids":     input_ids,
        "advantages":    advantages,
        "prompt_mask":   prompt_mask,
        "response_mask": response_mask,
        "group_ids":     group_ids,
        "rewards_vector": rewards_vector,
        "rewards_scalar": rewards_scalar,
        "gen_lengths":    gen_lengths,
    }

def make_minibatches(batch_dict, mb_size):
    B = batch_dict["input_ids"].size(0)
    idx = torch.randperm(B, device=device)
    for start in range(0, B, mb_size):
        mb_idx = idx[start:start+mb_size]
        yield {k: (v[mb_idx] if torch.is_tensor(v) and v.size(0) == B else v)
               for k, v in batch_dict.items()}

def token_logprobs(model, input_ids, ids):
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=input_ids.device.type, enabled=False):
        logits = model(input_ids)                  # (B, T, V)
        logits = logits[:, :-1, :]                # align with next-token targets
        logp   = F.log_softmax(logits, dim=-1)    # (B, T-1, V)
        lp     = logp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
        # optional numeric clamp on logprobs (not logits)
        lp = lp.clamp(min=-50.0, max=50.0)
        return lp

# the main function that does the PPO/GRPO step:
# think of this as token-wise weighting of a training sample.
# if the weights are negative, the backprop will penalize the model for those tokens.
# otherwise, the backprop will reward the model for those tokens.
# how to weight the tokens is a bit of an art, 
# and this is where PPO, GRPO, Dr GRPO, DAPO etc come in.
def grpo_step(model, optimizer, old_model, ref_model, batch,
              clip_high=0.10, clip_low=0.10, ent_coef=0.01, kl_coef=0.05,
              grad_clip=1.0, pad_id=pad_id):
    input_ids   = batch["input_ids"][:, :seq_len]
    advantages  = batch["advantages"][:, :seq_len]
    mask = batch["response_mask"][:, :seq_len].float()
    if not mask.any():
        return None, None

    model.train(); old_model.eval(); ref_model.eval()

    logits  = model(input_ids)
    ologits = old_model(input_ids)
    rlogits = ref_model(input_ids)

    # Causal shift
    ids      = input_ids  [:, 1:]
    adv      = advantages [:, 1:]
    mask      = mask [:, 1:]
    logits   = logits     [:, :-1, :]
    ologits  = ologits [:, :-1, :]
    rlogits  = rlogits [:, :-1, :]

    logp      = F.log_softmax(logits,  dim=-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    old_logp  = F.log_softmax(ologits, dim=-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    entropy  = -(F.log_softmax(logits, dim=-1) * F.softmax(logits, dim=-1)).sum(-1)

    # masked, whitened advantages
    mask_sum = mask.sum().clamp_min(1)
    adv_mask = adv * mask
    mean     = (adv_mask.sum() / mask_sum).detach()
    centered = adv * mask
    if cfg.batch_mean_centered:
        centered = (adv - mean) * mask
    
    advn     = (centered / 1.0).clamp(-5.0, 5.0)
    if cfg.batch_var_adjusted:
        var      = (centered.pow(2).sum() / mask_sum).detach().clamp_min(1e-6)
        advn     = (centered / var.sqrt()).clamp(-5.0, 5.0)
    #print(advn)
    #print()
    #print(f"generations: {batch['generations']}")
    #print(f"advn: {advn}")
    
    # PPO
    ratio    = torch.exp((logp - old_logp).clamp(-20, 20))
    surr1    = ratio * advn
    surr2    = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high) * advn
    surr_min = torch.min(surr1, surr2) * mask
    policy_loss = -1.0 * surr_min.sum() / mask_sum

    # KL(πθ || πref) directly in loss (exact KL via distributions)
    logp_all  = F.log_softmax(logits,  dim=-1)
    logp_ref  = F.log_softmax(rlogits, dim=-1)
    p_all     = logp_all.exp()
    kl_token  = (p_all * (logp_all - logp_ref)).sum(dim=-1)  # (B, T-1)
    kl_mean   = (kl_token * mask).sum() / mask_sum
    entropy_bonus = (entropy * mask).sum() / mask_sum

    if cfg.no_kl:
        kl_coef = 0.0
    loss = policy_loss - ent_coef * entropy_bonus + kl_coef * kl_mean
    # reduce loss =
    #   increase surr_min = increase adv = increase rewards
    #   increase entropy = favor more new trajectories
    #   reduce kl_mean = be closer to the base ref. 

    logp_before = token_logprobs(model, input_ids, ids)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    logp_after = token_logprobs(model, input_ids, ids)

    with torch.no_grad():
        r = ratio * mask
        ratio_mean = r.sum() / mask_sum
        ratio_max  = (r[r > 0].max() if (r > 0).any() else torch.tensor(1.0, device=r.device))

    return {
        "total_loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "entropy_bonus": float(entropy_bonus.item()),
        "kl_div": float(kl_mean.item()),
        "ratio_mean": float(ratio_mean.item()),
        "ratio_max": float(ratio_max.item()),
    }, {
        "input_ids": input_ids,
        "tokenmask": mask,
        "norm_adv": advn,
        "surr_min": surr_min,
        "old_probs": old_logp,
        "new_probs": logp,
        "before_probs": logp_before,
        "post_backprop": logp_after - logp_before,
        "kl_token": kl_token,
        "ratio": ratio
    }

def print_grpo_stats(update, step, stats, logs):
    s = ""
    for k, v in stats.items():
        s += f"{k}: {v:.3f} | "
    
    B, T = logs["input_ids"].shape

    printhtml(f"Update {update} / Step {step} / Minibatch size {B}: {s}", "bold")
    for idx in range(min(cfg.num_tables_to_print, B)):
        prompt = ""
        generation = ""
        tokens = []

        ids = logs["input_ids"][idx].tolist()        
        prompt_done = False
        for i in range(T):
            ch = sp.id_to_piece(ids[i])

            if i == 0: 
                prompt += ch
                continue  # skip the first token

            tokens.append(ch)
            if logs["tokenmask"][idx][i-1] < 1.0 and not prompt_done:
                prompt += ch
            else:
                generation += ch
                prompt_done = True

        table_data = {
            "ids"  : ids[1:],
            "token": tokens
        }

        c = "gray"
        mx = 0.0
        mn = 0.0
        for k, v in logs.items():
            if k != "input_ids":
                table_data[k] = v[idx].tolist()

        df = pd.DataFrame(table_data)
        gen = prompt + generation
        gen = gen.replace("_", " ").replace("<s>", " ").replace("▁", " ")

        htmlout = f"[{prompt}]: [{gen}]"
        htmlout += "</s>"
        htmlout += df.to_html(index=True, float_format="{:.3f}".format, justify="center")
        printhtml(htmlout, ignore_terminal=True, add_span=False, color=c)
    printhtml("-------------------")

@torch.no_grad()
def print_samples(model, dataset="train"):
    model.eval()
    prompts = rewards_math.train_prompts if dataset == "train" else rewards_math.test_prompts
    num_prompts_to_print = 2
    if dataset == "test":
        num_prompts_to_print = 100
    num_prompts_to_print = min(num_prompts_to_print, len(prompts))
    top_p = cfg.top_p
    temperature=cfg.temperature
    printhtml(f"Samples from the model at temp = {temperature:0.3f}, top_p = {top_p:.3f}:")
    num_correct = 0
    num_total = 0
    for p in random.sample(prompts, num_prompts_to_print):
        s = generate.sample_sequence_top_p(model, p, sp, cfg.max_new_tokens,
                                           temperature=temperature, top_p=top_p)
        r = cfg.reward_fn(s)
        printhtml(f"({r:.3f})  [{p}]: [{s}]", reward=r, reward_mid=rewards_math.mid2)
        if r >= rewards_math.highest:
            num_correct += 1
        num_total += 1
    accuracy = num_correct * 100.0 / (num_total+0.01)
    printhtml(f"Accuracy: {accuracy:.3f}% ({num_correct}/{num_total})", color="green" if accuracy > 0.5 else "red")

# =============== Optimizer (policy only) ===============
def build_param_groups(model, base_lr=5e-5, head_lr=2.5e-5, wd=0.01):
    """
    Smaller LR on the policy head keeps GRPO steps gentle.
    Adjust names to your MoEGPT module layout.
    """
    base, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if "lm_head" in n or n.endswith("head.weight") or n.endswith("head.bias"):
            head.append(p)
        else:
            base.append(p)
    groups = [
        {"params": base, "lr": base_lr, "weight_decay": wd},
        {"params": head, "lr": head_lr, "weight_decay": wd},
    ]
    if not head:
        printhtml("⚠️ No head parameters matched; using single LR group.", "yellow")
        groups = [{"params": model.parameters(), "lr": base_lr, "weight_decay": wd}]
    return groups

def get_reward_stats(rewards_scalar):
    nn = len(rewards_scalar)
    num_highest = 0
    for i in range(nn):
        num_highest += (rewards_scalar[i] >= rewards_math.highest)

    return (
        float(rewards_scalar.mean().item()), # avg
        float(rewards_scalar.min().item()), # min
        float(rewards_scalar.max().item()), # max
        round(100.0 * float(num_highest.item()) / (nn+0.001), 3) # % highest
    )

# =============== the GRPO outer loop ===============
# will sample some generations, verify the outputs, and call grpo_step to 
# update the model.
def grpo(model, optimizer):
    global updates_log, steps_log
    global rewards_log, kls_log, gens_log, percent_highest_log
    global total_losses_log, policy_losses_log, entropies_log
    global cfg

    # reference model snapshot (refreshed once per update)
    ref_model = copy.deepcopy(model).to(device)
    for p in ref_model.parameters(): p.requires_grad = False
    ref_model.eval()

    # behavior model for ratios (refreshed each step)
    old_model = copy.deepcopy(model).to(device)
    for p in old_model.parameters(): p.requires_grad = False
    old_model.eval()

    kl_coef = cfg.kl_coef_init
    step = 0

    cache = gencache.GenCache()
    if len(args.cache) > 0:
        cache.load(args.cache)
    cache.maxsize = cfg.cache_maxsize

    try:
        for update in range(cfg.total_updates):
            torch.mps.synchronize()
            t_update_start = time.perf_counter()

            # set reference to current policy at start of update (line 3 in Algo 1)
            ref_model.load_state_dict(model.state_dict())
            ref_model.eval()

            # --- collect rollout: G samples per prompt using old_model snapshot
            # refresh old_model right before sampling (behavior policy)
            old_model.load_state_dict(model.state_dict())
            old_model.eval()

            torch.mps.synchronize()
            t0 = time.perf_counter()
            
            if (update % 50 > 60) and len(cache.items) > 0: # noop
                roll, _ = cache.get()
                printhtml("got from cache", "yellow")
            else:
                roll = get_rolls(old_model)

            torch.mps.synchronize()
            t1 = time.perf_counter()

            if debug_mode:
                printhtml("roll:")
                printhtml(roll)

            if roll is None and len(cache.items) > 0:
                roll, _ = cache.get()
                printhtml("got a None roll. getting from cache", "yellow")
            if roll is None:
                printhtml("got a None roll, but nothing in cache. skipping", "yellow")
                continue

            avg_reward, min_reward, max_reward, percent_highest = get_reward_stats(roll["rewards_scalar"])
            if percent_highest > 5.0:
                cache.insert(roll, percent_highest)
            if avg_reward < rewards_math.low and len(cache.items) > 0:
                # reduce exploration and get a good roll from cache
                cfg.ent_coef = max(0.05, cfg.ent_coef / 1.5)
                prev_avg_reward = avg_reward
                roll, avg_reward = cache.get()
                if roll is None:
                    printhtml(f"no good rolls to swap a {prev_avg_reward:.3f}. cfg.ent_coef = {cfg.ent_coef:0.3f}", "red")
                    continue
                else:
                    printhtml(f"swapped {prev_avg_reward:.3f} with {avg_reward:.3f}. reduced ent_coef to {cfg.ent_coef:0.3f}", "red")

            # --- GRPO epochs (μ) over the same rollout
            epoch_kls = []
            for epoch in range(cfg.grpo_epochs):
                for mb in make_minibatches(
                    {"input_ids": roll["input_ids"],
                    "advantages": roll["advantages"],
                    "response_mask": roll["response_mask"]
                    },
                    cfg.minibatch_size
                ):
                    torch.mps.synchronize()
                    t2 = time.perf_counter()

                    step += 1
                    stats, logs = grpo_step(
                        model, optimizer, old_model, ref_model, mb,
                        clip_high=cfg.clip_high,
                        clip_low=cfg.clip_low,
                        ent_coef=cfg.ent_coef,
                        kl_coef=kl_coef,
                        grad_clip=cfg.grad_clip,
                        pad_id=pad_id
                    )
                    if stats is None:
                        continue

                    if debug_mode:
                        printhtml("DEBUG:")
                        printhtml(f"stats: {stats}")
                        printhtml(f"logs: {logs}")

                    steps_log += [step]
                    total_losses_log += [round(float(stats["total_loss"]), 3)]
                    policy_losses_log += [round(float(stats["policy_loss"]), 3)]
                    entropies_log += [round(float(stats["entropy_bonus"]), 3)]
                    kls_log += [round(10*float(stats["kl_div"]), 3)]

                    torch.mps.synchronize()
                    t3 = time.perf_counter()

                    if update % cfg.table_logging_frequency == 1:
                        printhtml(f"--------------------    {step}    -----------------------", "gray")
                        print_grpo_stats(update, step, stats, logs)
                        printhtml(f"--------------------    {step}    -----------------------", "gray")

                epoch_kls.append(stats["kl_div"])
                mean_kl_epoch = sum(epoch_kls) / max(1, len(epoch_kls))
                if mean_kl_epoch > cfg.kl_high_factor * cfg.target_kl:
                    printhtml(f"⚠️ {update} / {step}, KL={mean_kl_epoch:.4f}", "yellow")
                    break

            # --- adapt KL coefficient (β)
            mean_kl = sum(epoch_kls) / max(1, len(epoch_kls)) if epoch_kls else 0.0
            hi = cfg.target_kl * 1.2
            lo = cfg.target_kl * 0.8
            if mean_kl > hi:
                kl_coef *= cfg.kl_adapt_rate
            elif mean_kl < lo:
                kl_coef /= cfg.kl_adapt_rate
            if mean_kl > cfg.kl_high_factor * cfg.target_kl:
                kl_coef = max(kl_coef * (cfg.kl_adapt_rate ** 2), 1e-4)
            kl_coef = float(min(1.0, max(1e-6, kl_coef)))

            updates_log += [update]
            rewards_log += [round(float(avg_reward), 3)]
            percent_highest_log += [percent_highest]

            num_gens = len(roll['generations'])
            gens_log += [num_gens]

            torch.mps.synchronize()
            t_update_end = time.perf_counter()

            printhtml(f"{update} of {cfg.total_updates} | step: {step} | "
                      f"#cache: {len(cache.items)} | #gens: {num_gens} | "
                      f"AVG REWARD: {avg_reward:.3f} (min={min_reward:.3f}, max={max_reward:.3f}, accuracy={percent_highest:.1f}%) | "
                      f"KL: {mean_kl:.3f}, KL_coeff: {kl_coef:.3f} | "
                      f"gen time: {t1-t0:.3f}s, update time: {t_update_end-t_update_start:.3f}s",
                         reward=avg_reward, reward_mid=rewards_math.mid2)

            if update % cfg.print_samples_frequency == 1:
                print_samples(model)
                savemodel(model, f"{update}")
                printhtml(reason_for_none_rolls)
            if update % cfg.entropy_schedule == (cfg.entropy_schedule-1):
                cfg.ent_coef = max(0.05, cfg.ent_coef / 1.5)
                printhtml(f"new ent_coef = {cfg.ent_coef:.3f}")

    except KeyboardInterrupt:
        print("Interrupted.. cleaning up")
    
    # final save
    printhtml("COMPLETED!", "green")
    print_samples(model, "test")
    savemodel(model, "final")
    cache.save(f"checkpoints/cache-{args.suffix}")
    return {"done": True}

if __name__ == "__main__":
    cfg = GRPOConfig()

    configs = { # group, batch, batchvar, KL
        'train-fff-noKL-0.5': (False, False, False, True, 0.5),
        'train-fff-noKL-0.1': (False, False, False, True, 0.1),
        'train-fff-yesKL-0.1': (False, False, False, False, 0.1),
        'train-tff-noKL-0.5': (True, False, False, True, 0.5),
        'train-tff-noKL-0.1': (True, False, False, True, 0.1),
        'train-tff-yesKL-0.1': (True, False, False, False, 0.1),
        'train-ftf-yesKL-0.1': (False, True, False, False, 0.1),
        'train-ftt-yesKL-0.1': (False, True, True, False, 0.1),
    }
    
    for k, v in configs.items():
        print(k, v)
        args.suffix = k
        #args.ckpt = f"checkpoints/zerovec-final-{k}.pt"
        cfg.group_mean_centered = v[0]
        cfg.batch_mean_centered = v[1]
        cfg.batch_var_adjusted  = v[2]
        cfg.no_kl = v[3]
        cfg.ent_coef = v[4]

        updates_log = []
        steps_log = []
        rewards_log = []
        percent_highest_log = []
        kls_log = []
        entropies_log = []
        gens_log = []
        total_losses_log = []
        policy_losses_log = []
        prints = []

        model = SmallGPT(
            vocab_size=config["vocab_size"],
            n_layer=2, n_head=4, n_embd=64,
            block_size=seq_len,
        ).to(device)
        if len(args.ckpt) > 0: # load from a checkpoint
            printhtml(f"Loading checkpoint from {args.ckpt}", "bold")
            state_dict = torch.load(args.ckpt, map_location=device)
            model.load_state_dict(state_dict, strict=True)
        else: # we dont need the KL term
            cfg.no_kl = True
            cfg.ent_coef = 0.5
            cfg.entropy_schedule = 100
            printhtml("starting from scratch, so no KL and high entropy", "bold")

        optimizer = AdamW(model.parameters(), lr=5e-5)

        printhtml(f"Using device: {device}")
        printhtml(model)
        total_params = sum(p.numel() for p in model.parameters())
        printhtml(f"params: {total_params/1000000:.2f}M.")

        reason_for_none_rolls = {
            "no_encoded_prompts": 0,
            "B_is_0": 0,
            "empty_after_norm": 0,
            "!keep.any": 0,
        }
        _ = grpo(model, optimizer)
