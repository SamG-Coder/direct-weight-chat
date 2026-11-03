"""Evaluate the frozen transformer without changing or fitting its weights."""
import argparse
import hashlib
import json
import random
import socket
import statistics
import sys
import time
from pathlib import Path

from .engine import Model, DEFAULT_MODEL

import torch


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="auto")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((DEFAULT_MODEL / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        if digest(DEFAULT_MODEL / name) != expected:
            raise RuntimeError(f"Integrity failure: {name}")
    for name, expected in manifest["source_sha256"].items():
        if digest(DEFAULT_MODEL.parent / name) != expected:
            raise RuntimeError(f"Construction source changed: {name}")
    model = Model(device=args.device)
    assert not any(name in sys.modules for name in ("chat1gb_autoregressive.knowledge", "chat1gb_autoregressive.build", "chat1gb.program"))
    count = sum(t.numel() for t in model.w.values())
    assert count == 250_000_000
    assert (DEFAULT_MODEL / "weights.f32").stat().st_size == 1_000_000_000
    result = {"device": model.device, "quick": args.quick, "weight_sha256": manifest["sha256"]["weights.f32"],
              "parameter_count": count, "nonzero_parameters": manifest["nonzero_parameters"],
              "checks": {}, "cases": [], "general_chat_probes": [], "version": "fifth analytic build"}
    result["checks"]["integrity_and_standard_tensor_layout"] = True
    result["checks"]["runtime_imports_no_facts_or_response_program"] = True
    with socket.socket() as connection:
        try:
            connection.connect(("127.0.0.1", 9))
        except PermissionError:
            result["checks"]["offline_socket_guard"] = True
        else:
            raise RuntimeError("Offline guard did not reject connection")

    def check(prompt, expected, category, reset=True, trace=False):
        if reset:
            model.reset()
        item = model.generate(prompt, 40, trace)
        item.update(expected=expected, correct=item["text"] == expected, category=category)
        result["cases"].append(item)
        return item

    base = [
        ("Hello", "Hello."),
        ("Explain a cat.", "Cat is an animal. It can move."),
        ("Explain a neural network.", "Neural network is a model. It can learn."),
        ("What can a fish do?", "Fish can swim."),
        ("What does a tree need?", "Tree needs sunlight."),
        ("What does a computer have?", "Computer has processor."),
        ("What does a fire produce?", "Fire produces heat."),
        ("Describe ice.", "Ice is cold."),
        ("Where is a moon?", "Moon exists in space."),
        ("A dax is a vehicle. A dax can fly. Explain a dax.", "Dax is a vehicle. It can fly."),
        ("A zorb needs sunlight. What does a zorb need?", "Zorb needs sunlight."),
        ("A dax can swim. A dax can fly. What can a dax do?", "Dax can fly."),
        ("A dax is a bird. What can a dax do?", "Dax can fly."),
        ("A dax is a bird. A dax can swim. What can a dax do?", "Dax can swim."),
        ("An ombre is an observatory. An ombre can shimmer. Explain an ombre.", "Ombre is an observatory. It can shimmer."),
    ]
    for prompt, expected in base:
        check(prompt, expected, "functional", trace=True)
    print(f"Functional: {sum(c['correct'] for c in result['cases'])}/{len(base)}", flush=True)

    # These combinations and word forms are chosen only after the weights were frozen.
    # They are prompts to the model, not examples used to fit any parameter.
    rng = random.Random(240391)
    novel_count = 6 if args.quick else 40
    for i in range(novel_count):
        name = "vel" + "".join(rng.choice("bcdfghklmnprstvwz") for _ in range(5))
        category, article = rng.choice([("animal", "an"), ("vehicle", "a"), ("machine", "a"), ("observatory", "an")])
        verb = rng.choice(["swim", "fly", "pivot", "glimmer", "zindle", "shimmer"])
        prompt = f"A {name} is {article} {category}. A {name} can {verb}. Explain a {name}."
        check(prompt, f"{name.capitalize()} is {article} {category}. It can {verb}.", "novel_composition")

    arithmetic_count = 8 if args.quick else 120
    operands = [(0, 0), (9, 1), (123, 456), (999, 999)]
    operands += [(rng.randrange(1000), rng.randrange(1000)) for _ in range(arithmetic_count - len(operands))]
    for a, b in operands:
        check(f"{a} + {b}", f"{a+b}.", "unseen_arithmetic")
    model.reset()
    for a, b in [(123, 456), (11, 22), (999, 1), (72, 128), (0, 0)]:
        check(f"{a} + {b}", f"{a+b}.", "multi_turn_arithmetic", reset=False)
    model.reset()
    for prompt, expected in [
        ("A vesk is a bird. Explain a vesk.", "Vesk is a bird. It can fly."),
        ("What can a vesk do?", "Vesk can fly."),
        ("A vesk can swim. What can a vesk do?", "Vesk can swim."),
        ("Explain a computer.", "Computer is a machine. It can calculate."),
        ("Tell me more.", "Computer can calculate."),
    ]:
        check(prompt, expected, "multi_turn_facts", reset=False)

    # Prefix invariance verifies causal attention rather than future-token access.
    model.reset()
    special = model.config["special"]
    common = [special["bos"], special["user"]] + model.tokenizer.encode("A dax is a bird.")
    left = common + model.tokenizer.encode("What can a dax do?") + [special["assistant"]]
    right = common + model.tokenizer.encode("A dax can swim.") + [special["assistant"]]
    a, _, _ = model.forward(left)
    b, _, _ = model.forward(right)
    difference = float((a[:len(common)] - b[:len(common)]).abs().max().item())
    result["checks"]["causal_prefix_max_logit_difference"] = difference
    assert difference < 0.01, difference

    # Full-prefix and cached decoding must choose the same tokens at each step.
    model.reset()
    prefix = [special["bos"], special["user"]] + model.tokenizer.encode("A dax is a bird. Explain a dax.") + [special["assistant"]]
    logits, cache, _ = model.forward(prefix)
    generated = []
    for _ in range(16):
        full, _, _ = model.forward(prefix + generated)
        token = int(logits[-1].argmax().item())
        assert token == int(full[-1].argmax().item())
        generated.append(token)
        if token == special["eos"]:
            break
        logits, cache, _ = model.forward([token], cache)
    result["checks"]["cached_and_full_prefix_tokens_match"] = True

    # Ablate neural retrieval in memory. No response or fact code in the host
    # can compensate for the disabled transformer layers.
    prompt = "A kelv is a vehicle. A kelv can shimmer. Explain a kelv."
    model.reset()
    original = model.generate(prompt, 24)
    saved = {}
    for layer in range(3, 10):
        key = f"blocks.{layer}.attn.out.weight"
        saved[key] = model.w[key]
        model.w[key] = torch.zeros_like(model.w[key])
    model.reset()
    ablated = model.generate(prompt, 24)
    model.w.update(saved)
    assert ablated["text"] != original["text"]
    result["checks"]["retrieval_ablation"] = {"original": original["text"], "ablated": ablated["text"]}
    saved_head = model.w["lm_head.weight"]
    model.w["lm_head.weight"] = torch.zeros_like(saved_head)
    model.reset()
    blank = model.generate("Hello", 4)
    model.w["lm_head.weight"] = saved_head
    assert blank["text"] == ""
    result["checks"]["output_head_ablation_removes_text"] = True

    if not args.quick:
        for prompt in [
            "Write an original poem about a rainy city.",
            "Explain why a fish cannot live on land.",
            "Write a Python function that sorts a file.",
            "If Mara is taller than Eli and Eli is taller than Jo, who is tallest?",
            "What is the capital of France?",
            "Do not explain a cat.",
            "A dax cannot fly. What can a dax do?",
            "A dax is a zorb. A zorb is a bird. What can a dax do?",
            "1234 + 5678",
            "17 * 24",
            "Tell me something surprising about the ocean.",
        ]:
            model.reset()
            result["general_chat_probes"].append(model.generate(prompt, 40, True))

    result["summary"] = {}
    for category in sorted({item["category"] for item in result["cases"]}):
        subset = [item for item in result["cases"] if item["category"] == category]
        result["summary"][category] = {"correct": sum(item["correct"] for item in subset), "total": len(subset)}
    result["summary"]["all"] = {"correct": sum(item["correct"] for item in result["cases"]), "total": len(result["cases"])}
    result["summary"]["median_complete_response_ms"] = statistics.median(item["seconds"] for item in result["cases"]) * 1000
    assert digest(DEFAULT_MODEL / "weights.f32") == manifest["sha256"]["weights.f32"]
    result["checks"]["weight_hash_unchanged_after_all_tests"] = True
    evidence = DEFAULT_MODEL.parent / "evidence"
    evidence.mkdir(exist_ok=True)
    path = evidence / f"verification-{model.device}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"Integrity, causality, cache parity, ablations and offline checks passed. Saved {path}", flush=True)
    for failed in (row for row in result["cases"] if not row["correct"]):
        print(f"RETAINED FAILURE: {failed['prompt']} -> {failed['text']!r}; expected {failed['expected']!r}", flush=True)


if __name__ == "__main__":
    main()
