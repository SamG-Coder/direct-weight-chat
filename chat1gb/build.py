"""Compile a neural conversation program into exactly one GB of float32 tensors.

No random initialization, examples fitted to targets, gradients, optimizer,
pretrained checkpoint, or network access. Python 3.14 + existing NumPy suffice.
"""
from chat1gb import offline

import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from chat1gb.program import program

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "chat1gb" / "model"
TARGET_PARAMETERS = 250_000_000
VOCABULARY = 258
BITS = 9
NAME_TOKEN, SUM_TOKEN = 256, 257


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def encode_reply(text):
    tokens = []
    while text:
        if text.startswith("{name}"):
            tokens.append(NAME_TOKEN)
            text = text[6:]
        elif text.startswith("{sum}"):
            tokens.append(SUM_TOKEN)
            text = text[5:]
        else:
            tokens.extend(text[0].encode("utf-8"))
            text = text[1:]
    return tokens + [0]


def bits(tokens):
    values = np.asarray(list(tokens), dtype=np.uint16)
    return (((values[..., None] >> np.arange(BITS)) & 1).astype(np.float32) * 2) - 1


def hadamard(size):
    out = np.ones((1, 1), dtype=np.float32)
    while len(out) < size:
        out = np.block([[out, out], [out, -out]])
    return out


def choose_dimensions(features, concepts, replies, minimum_length):
    # Every byte is a named neural parameter. Search actual hidden widths and
    # decoder capacities, rather than adding a padding tensor or trailing bytes.
    constant = concepts + replies * concepts + 3 * replies + VOCABULARY * (BITS + 1) + 53 + 6 * BITS
    for use_bias in (True, False):
        coefficient = features + concepts + int(use_bias)
        for length in range(max(512, minimum_length), 8193):
            remaining = TARGET_PARAMETERS - constant - replies * length * BITS
            if remaining > 0 and remaining % coefficient == 0:
                return remaining // coefficient, length, use_bias
    raise RuntimeError("No exact one-GB architecture found for this program")


def main():
    started = time.monotonic()
    if MODEL.exists():
        raise SystemExit(f"Refusing to overwrite a model: {MODEL}")
    p = program()
    feature_names = sorted({term for terms in p["groups"].values() for term in terms})
    concept_names = list(p["groups"])
    fidx = {name: i for i, name in enumerate(feature_names)}
    cidx = {name: i for i, name in enumerate(concept_names)}
    count_f, count_g, count_r = len(fidx), len(cidx), len(p["replies"])
    sequences = [encode_reply(r["text"]) for r in p["replies"]]
    hidden, length, use_bias = choose_dimensions(count_f, count_g, count_r, max(map(len, sequences)))
    period = 1 << (count_f - 1).bit_length()

    shapes = {
        "encoder.weight": (hidden, count_f),
        **({"encoder.bias": (hidden,)} if use_bias else {}),
        "concept.weight": (count_g, hidden),
        "concept.bias": (count_g,),
        "rule.weight": (count_r, count_g),
        "rule.bias": (count_r,),
        "reply.bits": (count_r, length, BITS),
        "byte_decoder.weight": (VOCABULARY, BITS),
        "byte_decoder.bias": (VOCABULARY,),
        "name_write.weight": (count_r,),
        "name_clear.weight": (count_r,),
        "name_default.bits": (6, BITS),
        "adder.sum.weight": (1, 21),
        "adder.sum.bias": (1,),
        "adder.hinge.weight": (2, 1),
        "adder.hinge.bias": (2,),
        "adder.digit.weight": (1, 3),
        "adder.digit.bias": (1,),
        "adder.carry.weight": (1, 2),
        "adder.carry.bias": (1,),
        "adder.readout.weight": (10, 1),
        "adder.readout.bias": (10,),
    }
    total = sum(math.prod(shape) for shape in shapes.values())
    if total != TARGET_PARAMETERS:
        raise RuntimeError(f"Incorrect parameter budget: {total}")
    MODEL.mkdir(parents=True)
    weights_path = MODEL / "weights.f32"
    storage = np.memmap(weights_path, mode="w+", dtype="<f4", shape=(total,))
    tensors, metadata, offset = {}, {}, 0
    for name, shape in shapes.items():
        size = math.prod(shape)
        tensors[name] = storage[offset:offset + size].reshape(shape)
        metadata[name] = {"shape": list(shape), "offset_bytes": offset * 4, "nbytes": size * 4}
        offset += size

    selectors = np.zeros((count_g, count_f), dtype=np.float32)
    for name, terms in p["groups"].items():
        for term in terms:
            selectors[cidx[name], fidx[term]] = 1
    q = hadamard(period)[:count_f]
    projected = selectors @ q
    counts = np.bincount(np.arange(hidden) % period, minlength=period).astype(np.float32)

    # Q Q^T = period I. Repeated, scaled columns provide an overcomplete frame.
    # The second layer is its explicit inverse composed with lexical OR gates.
    # No target examples, estimates, regression, or fitted coefficients are used.
    for start in range(0, hidden, 8192):
        end = min(start + 8192, hidden)
        positions = np.arange(start, end)
        basis = positions % period
        scale = np.asarray([0.5, 1.0, 2.0, 4.0], dtype=np.float32)[(positions // period) % 4]
        tensors["encoder.weight"][start:end] = q[:, basis].T * scale[:, None]
        tensors["concept.weight"][:, start:end] = projected[:, basis] / (period * counts[basis] * scale)[None, :]
        if start % 131072 == 0:
            print(f"Writing dense neural layers: {end:,}/{hidden:,} hidden units", flush=True)
    if use_bias:
        tensors["encoder.bias"][:] = 0
    tensors["concept.bias"][:] = 0
    tensors["rule.weight"][:] = 0
    for rule in p["rules"]:
        rid = rule["reply"]
        for name in rule["required"]:
            tensors["rule.weight"][rid, cidx[name]] = 1
        for name in rule["forbidden"]:
            tensors["rule.weight"][rid, cidx[name]] = -4
        tensors["rule.bias"][rid] = -len(rule["required"]) + 0.5 + rule["priority"] / 1000

    tensors["reply.bits"][:] = -1
    for rid, sequence in enumerate(sequences):
        tensors["reply.bits"][rid, :len(sequence)] = bits(sequence)
    tensors["byte_decoder.weight"][:] = bits(range(VOCABULARY))
    tensors["byte_decoder.bias"][:] = 0
    tensors["name_write.weight"][:] = [float(r["write_name"]) for r in p["replies"]]
    tensors["name_clear.weight"][:] = [float(r["clear_name"]) for r in p["replies"]]
    tensors["name_default.bits"][:] = bits(b"friend")

    # The neural decimal adder handles a + b + carry for one column at a time.
    tensors["adder.sum.weight"][:] = [list(range(10)) + list(range(10)) + [1]]
    tensors["adder.sum.bias"][:] = 0
    tensors["adder.hinge.weight"][:] = 1
    tensors["adder.hinge.bias"][:] = [-9, -10]
    tensors["adder.digit.weight"][:] = [[1, -10, 10]]
    tensors["adder.digit.bias"][:] = 0
    tensors["adder.carry.weight"][:] = [[1, -1]]
    tensors["adder.carry.bias"][:] = 0
    tensors["adder.readout.weight"][:] = np.arange(10, dtype=np.float32)[:, None] * 2
    tensors["adder.readout.bias"][:] = -(np.arange(10, dtype=np.float32) ** 2)
    storage.flush()

    tensor_stats = {}
    for name, tensor in tensors.items():
        if not np.isfinite(tensor).all():
            raise RuntimeError(f"Nonfinite parameter in {name}")
        tensor_stats[name] = {"parameters": tensor.size, "nonzero": int(np.count_nonzero(tensor)),
                              "min": float(tensor.min()), "max": float(tensor.max())}
    metadata_config = {
        "format": "offline-neural-chat-f32-v1", "name": "1 GB Neural Chat Experiment",
        "dtype": "float32", "byte_order": "little", "weights_file": "weights.f32",
        "parameter_count": total, "weight_bytes": total * 4,
        "features": feature_names, "max_ngram": max(len(t.split()) for t in feature_names if not t.startswith("@")),
        "concepts": concept_names, "reply_labels": [r["name"] for r in p["replies"]],
        "architecture": {"input_features": count_f, "hidden_width": hidden, "concepts": count_g,
                         "reply_states": count_r, "max_reply_tokens": length, "byte_bits": BITS,
                         "byte_vocabulary": VOCABULARY, "encoder_bias": use_bias,
                         "activation": "capped_relu", "hadamard_period": period},
        "tokens": {"eos": 0, "copy_name": NAME_TOKEN, "neural_sum": SUM_TOKEN},
        "max_input_characters": 8192, "max_operand_digits": 12,
        "tensors": metadata,
    }
    (MODEL / "model.json").write_text(json.dumps(metadata_config, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "construction": "direct compilation of explicitly authored neural circuits",
        "training_steps": 0, "optimizer_steps": 0, "gradient_updates": 0,
        "pretrained_weights_used": False, "training_data_used": False, "internet_used": False,
        "random_parameter_initialization": False, "padding_bytes": 0,
        "parameter_count": total, "weight_bytes": weights_path.stat().st_size,
        "weight_gb_decimal": weights_path.stat().st_size / 1e9,
        "weight_gib_binary": weights_path.stat().st_size / (1024 ** 3),
        "sha256": {"weights.f32": digest(weights_path), "model.json": digest(MODEL / "model.json")},
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path)
                          for path in (ROOT / "chat1gb/program.py", ROOT / "chat1gb/build.py", ROOT / "chat1gb/offline.py")},
        "tensor_statistics": tensor_stats,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "construction_seconds": time.monotonic() - started,
        "limitations": [
            "An explicitly programmed neural conversation controller, not a pretrained or trained language model.",
            "Most parameters implement a redundant, overcomplete dense frame; size does not imply independent knowledge.",
            "Responses are finite authored byte memories selected by neural gates, with name copying and neural addition.",
            "Lexical feature extraction and operand/name parsing are ordinary host code.",
            "Context is the previous reply state plus an in-memory name slot; there is no general dialogue understanding.",
            "No claim of benchmark-level general reasoning or human-like conversational ability.",
        ],
    }
    (MODEL / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Created {weights_path}", flush=True)
    print(f"{total:,} parameters / {weights_path.stat().st_size:,} bytes; no padding", flush=True)
    print(f"Architecture: {count_f} -> {hidden} -> {count_g} -> {count_r}; decoder capacity {length}", flush=True)
    print(f"SHA-256: {manifest['sha256']['weights.f32']}", flush=True)


if __name__ == "__main__":
    main()
