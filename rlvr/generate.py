# inference utils

import torch
import sentencepiece as spm
from config import config
from termcolor import colored
from torch.amp import autocast, GradScaler
import random
import rewards_math
import copy

def gray(s):
  return "\033[3m" + colored(s, "grey") + "\033[0m"

def printcolor(s, color):
    print(colored(s, color), end="", flush=True)

def sample_top_p(logits: torch.Tensor, top_p: float = 0.95, temperature: float = 0.9):
    # logits: (B, V)
    if temperature is not None and temperature > 0 and temperature != 1.0:
        logits = logits / max(1e-8, float(temperature))
    probs = torch.softmax(logits, dim=-1)  # (B, V)
    # sort by prob desc
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    
    #print(f"sorted probs:\t{sorted_probs}")
    #print(f"sorted idx:\t{sorted_idx}")

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

@torch.no_grad()
def sample_sequence(model, prompt, sp,
                    max_new_tokens=config['block_size'], temperature=1.0, add_bos=False):
    model.eval()
    input_ids = sp.encode(prompt, out_type=int, add_bos=add_bos, add_eos=False)
    input_tensors = torch.tensor(input_ids, dtype=torch.long, device=config['device']).unsqueeze(0)

    log_probs = []
    output_ids = []
    generated = input_tensors.clone()

    eos_id = sp.eos_id()

    num_tokens = input_tensors.size(1)
    for _ in range(max_new_tokens):
        if num_tokens >= config["block_size"]:
            break

        with autocast(device_type=config['device']):
            logits = model(generated)
            logits = logits[:, -1, :]  # last token
            next_token_p = sample_top_p(logits, top_p=1.0, temperature=0.8)
            probs = torch.softmax(logits, dim=-1)
            #print(colored(f"probs before temp:\t {probs}", "yellow"))
            logits = logits / temperature  # last token
            probs = torch.softmax(logits, dim=-1)
            #print(colored(f"probs after temp:\t {probs}", "green"))
            dist = torch.distributions.Categorical(probs)
            next_token = dist.sample()
            #print()
            #print(
            #    colored(f"\t\tnext token: {sp.decode(next_token.item())} ({next_token.item()})", "red"),
            #    colored(f"|   next_token_p: {sp.decode(next_token_p.item())} ({next_token_p.item()})", "blue")
            #)
            log_prob = dist.log_prob(next_token)
            num_tokens += 1

        if next_token.item() == eos_id:
            break

        log_probs.append(log_prob)
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        output_ids.append(next_token.item())

    all_log_probs = torch.tensor([], device=config['device'])
    if len(log_probs) > 0: # no generation
        all_log_probs = torch.stack(log_probs)

    return input_ids, output_ids, generated.squeeze(0), all_log_probs

def special_encode(p, sp):
    return sp.encode(p, out_type=int, add_bos=False, add_eos=False)

def special_decode(id, sp):
    #if id in [3, 4, 5]: # _1, _2, _3 are treated as unknowns
    #    id = 28
    #if id in [20]:
    #    id = 23
    #if id in [2]:
    #    id = 20
    return sp.decode(id)


@torch.no_grad()
def sample_sequence_top_p(model, prompt, sp, max_new_tokens=config['block_size'],
                          top_p=0.9, temperature=1.0):
    model.eval()
    input_ids = special_encode(prompt, sp)
    input_tensors = torch.tensor(input_ids, dtype=torch.long, device=config['device']).unsqueeze(0)
    generated = input_tensors.clone()

    eos_id = sp.eos_id()

    #print(f"\n\ninput_ids:\t{input_ids}")

    num_tokens = input_tensors.size(1)
    for _ in range(max_new_tokens):
        if num_tokens >= config["block_size"]:
            break

        with autocast(device_type=config['device']):
            logits = model(generated)
            logits = logits[:, -1, :]  # last token
            #print(f"logits:\t{logits}")
            next_token = sample_top_p(logits, top_p=top_p, temperature=temperature)
            #print(colored(f"\t\t next_id: '{sp.decode(next_token.item())}'  ({next_token.item()})", "blue"))
            num_tokens += 1

        if next_token.item() == eos_id:
            break

        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

    return special_decode(generated.squeeze(0).tolist(), sp)

@torch.no_grad()
def sample_sequence_as_string(model, prompt, sp,
                              max_new_tokens=config['block_size'], temperature=1.0):
    input_ids, output_ids, seq, log_probs = sample_sequence(
        model, prompt, sp, max_new_tokens, temperature, add_bos=True if prompt == "" else False)
    return sp.decode(seq.tolist())


@torch.no_grad()
def generate_tokens(model, prompt, sp, max_new_tokens=40, temperature=1.0):
    prompt = prompt.strip()
    add_bos = True if prompt == "" else False
    input_ids, output_ids, seq, log_probs = sample_sequence(
        model, prompt, sp, max_new_tokens, temperature, add_bos=add_bos)
    print(gray(prompt + ' = '), end="", flush=True)
    print(gray(input_ids), end="", flush=True)
    print()
    print(colored(sp.decode(output_ids), "blue"), end="", flush=True)
    print(gray(output_ids), end="", flush=True)
    print()
    return seq, log_probs

def get_rolls(old_model, prompts, group_size, max_new_tokens,
              temperature=0.9, top_p=0.95, reindex_groups: bool = True):
    enc_prompts = [sp.encode(p, out_type=int, add_bos=False, add_eos=False) for p in prompts]
    if len(enc_prompts) == 0:
        print("len(enc)==0")
        return None

    enc_prompts_per_group = []  # num_prompts * #groups
    group_ids = []
    for i, p_ids in enumerate(enc_prompts):
        for _ in range(group_size):
            enc_prompts_per_group.append(p_ids)
            group_ids.append(i)

    B = len(enc_prompts_per_group)
    if B == 0:
        print("B==0")
        return None

    prompt_lengths = torch.tensor([len(x) for x in enc_prompts_per_group], device=device, dtype=torch.long)
    max_prompt_len = int(prompt_lengths.max().item())
    # Pre-allocate full workspace: max prompt + max_new_tokens + 1 (room for EOS)
    T_max = max_prompt_len + max_new_tokens + 1

    input_ids = torch.full((B, T_max), pad_id, dtype=torch.long, device=device)

    # pre-fill encoded prompt ids
    for i, pids in enumerate(enc_prompts_per_group):
        L = len(pids)
        if L > 0:
            input_ids[i, :L] = torch.tensor(pids, dtype=torch.long, device=device)

    print(input_ids)
    # Track per-row current length (cursor at last written position + 1)
    cur_len = prompt_lengths.clone()            # where next token will be written
    alive   = torch.ones(B, dtype=torch.bool, device=device)
    gen_len = torch.zeros(B, dtype=torch.long, device=device)  # excludes EOS

    # Autoregressive generation for all rows
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
        if alive.any():
            chosen = sample_top_p(next_logits[alive], top_p=top_p, temperature=temperature)
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
        print("!keep.any")
        return None

    input_ids   = input_ids[keep]
    prompt_lengths = prompt_lengths[keep]
    gen_len     = gen_len[keep]
    group_ids   = torch.tensor(group_ids, device=device, dtype=torch.long)[keep]

    used_len = (prompt_lengths + gen_len + 1)
    T = int(used_len.max().item())
    input_ids = input_ids[:, :T].contiguous()

    print(input_ids)

    # attention mask, prompt/response masks
    ar = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
    prompt_mask   = (ar < prompt_lengths.unsqueeze(1)).float()
    response_mask = ((ar >= prompt_lengths.unsqueeze(1)) &
                     (ar <  (used_len).unsqueeze(1))).float()

    # Decode texts as (prompt_text, response_text)
    prompts_and_completions = []
    token_rewards_input = []
    input_ids_cpu = input_ids.detach().cpu()
    for i in range(input_ids_cpu.size(0)):
        p_text = prompts[group_ids[i].item()]  # original prompt string
        plen = prompt_lengths[i]
        ids = input_ids_cpu[i].tolist()
        gen = sp.decode(ids)
        print(f"{p_text}: {gen}: {ids}")
        chars = [sp.decode(ids[j]) for j in range(len(ids))]
        prompts_and_completions.append( (p_text, gen) )
        token_rewards_input.append((chars, plen))

    rewards_vector = []
    try:
        import concurrent.futures as futures
        with futures.ThreadPoolExecutor() as ex:
            rs = list(ex.map(rewards_math.reward_fn_per_token, token_rewards_input))
        rewards_vector = torch.tensor(rs, device=device, dtype=torch.float32)
    except Exception:
        print("here2")
        rewards_vector = torch.tensor([rewards_math.reward_fn_per_token(x) for x in token_rewards_input],
                                        device=device, dtype=torch.float32)

    # group mean baseline
    unique_gids, inv = torch.unique(group_ids, return_inverse=True)  # inv: (B,) maps row->group index [0..G-1]
    G = unique_gids.numel()
    inv_T = inv.unsqueeze(1).expand(-1, T)
    resp_mask_f = response_mask.float()

    print(f"Rewards_vector: {rewards_vector}")
    print(f"groupids: {group_ids}")
    print(f"inv_t: {inv_T}")
    print(f"rmask: {resp_mask_f}")

    sums = torch.zeros((G, T), device=device, dtype=torch.float32).scatter_add_(0, inv_T, rewards_vector * resp_mask_f)
    print(f"sums: {sums}")
    # counts[g,t] = number of rows in group g that are valid (mask==1) at time t
    counts = torch.zeros((G, T), device=device, dtype=torch.float32).scatter_add_(0, inv_T, resp_mask_f)
    print(f"counts: {counts}")
    means = sums / counts.clamp_min(1.0)         
    print(f"means: {means}")

    means_per_row = means[inv, :]
    print(f"means_per_row: {means_per_row}")


    centered = (rewards_vector - means_per_row) * response_mask

    print(f"centered = {centered}")

    # --- 3) DROP GROUPS with no variability (token-wise) ---
    # group variability = sum over rows in group of |centered_token| within mask
    abs_token = centered.abs()
    # accumulate per-group scalar variability
    group_var = torch.zeros(G, device=device, dtype=torch.float32).scatter_add_(
        0, inv, abs_token.sum(dim=1)
    )
    keep_group = group_var > float(1e-4)
    keep_mask = keep_group[inv]    # (B,)

    if not keep_mask.any():
        print("!keep mask . any")
        return None

    idx = keep_mask.nonzero(as_tuple=False).squeeze(-1)
    print(f"idx: {idx}")

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

    advantages  = centered * response_mask                       # (B,T)

    rewards_scalar = rewards_vector.mean(-1)
    print(rewards_scalar)
    return {
        "prompts":      [p for (p, _,) in prompts_and_completions],
        "generations":  [g for (_, g,) in prompts_and_completions],
        "input_ids":     input_ids,
        "advantages":    advantages,
        "prompt_mask":   prompt_mask,
        "response_mask": response_mask,
        "group_ids":     group_ids,
        "rewards_vector": rewards_vector,
        "rewards_scalar": rewards_scalar,
        "gen_lengths":    gen_lengths,
    }

if __name__ == "__main__":
    from model import SmallGPT
    import argparse
    parser = argparse.ArgumentParser(description="GENERATE")
    parser.add_argument("--ckpt", default="", help="Path to base checkpoint (.pt)")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.load("tokenizer" + config['data_prefix'] + ".model")
    pad_id = 1
    eos_id = 2
    seq_len = config['block_size']

    torch.set_printoptions(precision=3, linewidth=120, threshold=10000, sci_mode=False)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = SmallGPT(
        vocab_size=config["vocab_size"],
        n_layer=2, n_head=4, n_embd=64,
        block_size=seq_len,
    ).to(device)

    if len(args.ckpt) > 0: # load from a checkpoint
        print(f"Loading checkpoint from {args.ckpt}")
        state_dict = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state_dict, strict=True)
    else:
        print("using uninitialized random model")

    import rewards_math
    prompts2 = [
        "12+3=",
        "8+14="
    ]
    import rewards_math
    import numpy as np
    n = len(prompts2)
    total = 0.0
    num_highest = 0.0
    i = 0
    for p in rewards_math.prompts:
        for i in range(3):
            s = sample_sequence_top_p(model, p, sp, seq_len - len(p),
                                    temperature=0.8, top_p=0.9)
            r = rewards_math.reward_fn_hardest(s)
            c = "red"
            if r > rewards_math.mid:
                c = "green"
            printcolor(f"{p}: {s} ({r:.3f})\n", c)
        







