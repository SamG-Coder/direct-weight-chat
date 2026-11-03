"""Generic causal-transformer inference; no fact/rule compiler is imported here.

The generation loop does only tokenization, forward passes, next-token selection,
KV caching, and detokenization. It never chooses a sentence or computes an answer.
"""
from . import offline

import json
import re
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Failed to initialize NumPy:.*")
import torch
import torch.nn.functional as F

DEFAULT_MODEL = Path(__file__).resolve().parent / "model"


class Tokenizer:
    def __init__(self, config):
        self.config = config
        self.labels = config["vocabulary"]
        self.ids = {label.lower(): index for index, label in enumerate(self.labels)}
        self.pointers = {}
        self.reverse = {}
        self.used = [0, 0]
        self.max_phrase = max(len(label.split()) for label in self.labels)

    def encode(self, text):
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]", text)
        output, position = [], 0
        while position < len(words):
            for length in range(min(self.max_phrase, len(words) - position), 0, -1):
                key = " ".join(words[position:position + length]).lower()
                if key in self.ids and not key.startswith("<"):
                    output.append(self.ids[key])
                    position += length
                    break
            else:
                original = words[position]
                key = original.lower()
                if key not in self.pointers:
                    section = 0 if key[0] in "aeiou" else 1
                    if self.used[section] >= 64:
                        raise ValueError("The fixed pointer vocabulary is full; reset the conversation")
                    tid = self.config["dynamic_start"] + 64 * section + self.used[section]
                    self.used[section] += 1
                    self.pointers[key], self.reverse[tid] = tid, original
                output.append(self.pointers[key])
                position += 1
        return output

    def token_text(self, token):
        return self.reverse.get(token, self.labels[token])

    def decode(self, tokens):
        text = " ".join(self.token_text(token) for token in tokens if token != self.config["special"]["eos"])
        text = re.sub(r"\s+([.,?!:;])", r"\1", text)
        text = re.sub(r"(?<!\w)-\s+(?=\d)", "-", text)
        text = re.sub(r"\bi\b", "I", text)
        text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
        return text


class Model:
    def __init__(self, directory=DEFAULT_MODEL, device="auto"):
        self.directory = Path(directory)
        self.config = json.loads((self.directory / "config.json").read_text(encoding="utf-8"))
        if self.config["format"] != "direct-causal-transformer-f32-v1":
            raise ValueError("Unexpected model format")
        torch.set_num_threads(8)
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        weights = self.directory / self.config["weights_file"]
        if weights.stat().st_size != self.config["weight_bytes"]:
            raise ValueError("Incorrect parameter file size")
        mapped = torch.from_file(str(weights), shared=False, size=self.config["parameter_count"], dtype=torch.float32)
        self.storage = mapped.to(self.device)
        self.w, position = {}, 0
        for name, item in self.config["tensors"].items():
            if item["offset_bytes"] != position:
                raise ValueError("Tensor layout has a gap or overlap")
            start, size = position // 4, item["nbytes"] // 4
            self.w[name] = self.storage[start:start + size].reshape(item["shape"])
            position += item["nbytes"]
        assert position == weights.stat().st_size
        self.reset()

    def reset(self):
        self.tokenizer = Tokenizer(self.config)
        self.history = [self.config["special"]["bos"]]

    def norm(self, x, prefix):
        return F.layer_norm(x, (self.config["width"],), self.w[prefix + ".weight"],
                            self.w[prefix + ".bias"], self.config["norm_epsilon"])

    def linear(self, x, prefix):
        return F.linear(x, self.w[prefix + ".weight"], self.w[prefix + ".bias"])

    @torch.inference_mode()
    def forward(self, ids, cache=None, inspect=False):
        offset = 0 if cache is None else cache[0][0].shape[2]
        if offset + len(ids) > self.config["context_length"]:
            raise ValueError("Context is full; reset the conversation")
        tokens = torch.tensor([ids], device=self.device, dtype=torch.long)
        positions = torch.arange(offset, offset + len(ids), device=self.device)
        h = self.w["token_embedding"][tokens] + self.w["position_embedding"][positions][None, :, :]
        new_cache, snapshots = [], []
        heads = self.config["heads"]
        head_dim = self.config["width"] // heads
        for layer in range(self.config["layers"]):
            prefix = f"blocks.{layer}"
            x = self.norm(h, prefix + ".ln1")
            q, k, v = self.linear(x, prefix + ".attn.qkv").chunk(3, dim=-1)
            q, k, v = [value.reshape(1, len(ids), heads, head_dim).transpose(1, 2) for value in (q, k, v)]
            if cache is not None:
                k = torch.cat((cache[layer][0], k), dim=2)
                v = torch.cat((cache[layer][1], v), dim=2)
            new_cache.append((k, v))
            if cache is None:
                attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            elif len(ids) == 1:
                attention = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            else:
                allowed = torch.arange(k.shape[2], device=self.device)[None, :] <= positions[:, None]
                attention = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
            attention = attention.transpose(1, 2).contiguous().reshape(1, len(ids), self.config["width"])
            h = h + self.linear(attention, prefix + ".attn.out")
            h = h + self.linear(F.relu(self.linear(self.norm(h, prefix + ".ln2"), prefix + ".mlp.up")), prefix + ".mlp.down")
            if inspect:
                snapshots.append(((h[0, -1, 0::2] - h[0, -1, 1::2]) / 2).cpu().tolist())
        logits = self.linear(self.norm(h, "final_norm"), "lm_head")
        if not torch.isfinite(logits).all():
            raise RuntimeError("The model produced nonfinite token logits")
        return logits[0], new_cache, snapshots

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens=64, trace=False):
        prefix = self.history + [self.config["special"]["user"]] + self.tokenizer.encode(prompt) + [self.config["special"]["assistant"]]
        if len(prefix) + max_new_tokens > self.config["context_length"]:
            raise ValueError("Insufficient context space; reduce output length or reset the conversation")
        started = time.perf_counter()
        logits, cache, _ = self.forward(prefix)
        output, token_trace = [], []
        stopped = False
        for step in range(max_new_tokens):
            next_logits = logits[-1]
            token = int(next_logits.argmax().item())
            if trace:
                scores, indices = next_logits.topk(3)
                token_trace.append({"step": step, "token_id": token, "token": self.tokenizer.token_text(token),
                                    "top3": [{"token": self.tokenizer.token_text(int(i)), "logit": float(s)}
                                             for i, s in zip(indices.tolist(), scores.tolist())]})
            output.append(token)
            if token == self.config["special"]["eos"]:
                stopped = True
                break
            logits, cache, _ = self.forward([token], cache)
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.history = prefix + output
        text = self.tokenizer.decode(output)
        result = {"prompt": prompt, "text": text, "token_ids": output, "stopped_on_eos": stopped,
                  "generated_tokens": len(output), "seconds": time.perf_counter() - started,
                  "device": self.device, "prefix_tokens": len(prefix)}
        if trace:
            result["token_trace"] = token_trace
        return result
