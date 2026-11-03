"""Write an ordinary 18-block decoder transformer's parameters analytically.

No optimizer, loss, training examples, random initialization, downloaded data,
pretrained parameters, answer table, or runtime sentence renderer.
"""
from . import offline

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import spec as S
from .knowledge import FACTS, make_vocabulary

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "model"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def put_pair(row, feature, value):
    row[2 * feature] = value
    row[2 * feature + 1] = -value


def add_reader(row, expression, scale=1):
    for feature, value in expression.items():
        row[2 * feature] += scale * value / 2
        row[2 * feature + 1] -= scale * value / 2


def expr(*items):
    out = {}
    for item in items:
        for feature, value in item.items():
            out[feature] = out.get(feature, 0) + value
    return out


class Compiler:
    def __init__(self, tensors, ids):
        self.w, self.ids = tensors, ids
        self.used = [0] * S.LAYERS

    def unit(self, layer, expression, targets, threshold=0):
        row = self.used[layer]
        if row >= S.FFN:
            raise RuntimeError(f"Layer {layer} exceeds {S.FFN} ReLU units")
        self.used[layer] += 1
        prefix = f"blocks.{layer}.mlp"
        add_reader(self.w[prefix + ".up.weight"][row], expr(expression, {S.ONE: -threshold}))
        for feature, value in targets.items():
            self.w[prefix + ".down.weight"][2 * feature, row] += value
            self.w[prefix + ".down.weight"][2 * feature + 1, row] -= value

    def linear(self, layer, expression, targets):
        self.unit(layer, expression, targets)
        self.unit(layer, {f: -v for f, v in expression.items()}, {f: -v for f, v in targets.items()})

    def clear(self, layer, features):
        for feature in features:
            self.linear(layer, {feature: 1}, {feature: -1})

    def gate(self, layer, positive=(), negative=(), targets=None):
        expression = {}
        for feature in positive:
            expression[feature] = expression.get(feature, 0) + 2
        for feature in negative:
            expression[feature] = expression.get(feature, 0) - 2
        self.unit(layer, expression, targets or {}, threshold=2 * len(positive) - 1)

    def word(self, layer, word, targets, start=S.CODE, positive=(), negative=()):
        expression = {start + i: bit for i, bit in enumerate(S.code(self.ids[word]))}
        for feature in positive:
            expression[feature] = expression.get(feature, 0) + 2
        for feature in negative:
            expression[feature] = expression.get(feature, 0) - 2
        self.unit(layer, expression, targets, threshold=S.CODE_BITS + 2 * len(positive) - 1)

    def copy(self, layer, source, gate, destination, count=1, scale=1):
        for i in range(count):
            self.unit(layer, {source + i: 1, gate: 4}, {destination + i: scale}, threshold=4)
            self.unit(layer, {source + i: -1, gate: 4}, {destination + i: -scale}, threshold=4)

    def emit(self, layer, word, positive=(), negative=()):
        self.gate(layer, positive, negative, {S.OUTPUT + i: bit for i, bit in enumerate(S.code(self.ids[word]))})

    def clip(self, layer, features):
        for feature in features:
            self.unit(layer, {feature: 2, S.ONE: 1}, {feature: 1})
            self.unit(layer, {feature: 2, S.ONE: -1}, {feature: -1})
            self.unit(layer, {S.ONE: 1}, {feature: -1})
            self.linear(layer, {feature: 1}, {feature: -1})

    def attention(self, layer, head, queries, keys, sources, destinations):
        prefix = f"blocks.{layer}.attn"
        qkv = self.w[prefix + ".qkv.weight"]
        offset = head * S.HEAD_DIM
        for i, expression in enumerate(queries):
            add_reader(qkv[offset + i], expression)
        for i, expression in enumerate(keys):
            add_reader(qkv[S.WIDTH + offset + i], expression)
        for i, (source, destination) in enumerate(zip(sources, destinations)):
            add_reader(qkv[2 * S.WIDTH + offset + i], {source: 1})
            self.w[prefix + ".out.weight"][2 * destination, offset + i] += 1
            self.w[prefix + ".out.weight"][2 * destination + 1, offset + i] -= 1

    def latest(self, layer, head, candidate, sources, destinations, first=False):
        self.attention(layer, head, [{S.ONE: 8}],
                       [{candidate: 65536, S.POS: (-1 if first else 1) * 8 * 128}], sources, destinations)


def shapes():
    out = {"token_embedding": (S.VOCAB, S.WIDTH), "position_embedding": (S.CONTEXT, S.WIDTH)}
    for layer in range(S.LAYERS):
        p = f"blocks.{layer}"
        for norm in ("ln1", "ln2"):
            out[p + f".{norm}.weight"] = (S.WIDTH,)
            out[p + f".{norm}.bias"] = (S.WIDTH,)
        out[p + ".attn.qkv.weight"] = (3 * S.WIDTH, S.WIDTH)
        out[p + ".attn.qkv.bias"] = (3 * S.WIDTH,)
        out[p + ".attn.out.weight"] = (S.WIDTH, S.WIDTH)
        out[p + ".attn.out.bias"] = (S.WIDTH,)
        out[p + ".mlp.up.weight"] = (S.FFN, S.WIDTH)
        out[p + ".mlp.up.bias"] = (S.FFN,)
        out[p + ".mlp.down.weight"] = (S.WIDTH, S.FFN)
        out[p + ".mlp.down.bias"] = (S.WIDTH,)
    out["final_norm.weight"], out["final_norm.bias"] = (S.WIDTH,), (S.WIDTH,)
    out["lm_head.weight"], out["lm_head.bias"] = (S.VOCAB, S.WIDTH), (S.VOCAB,)
    return out


def main():
    started = time.monotonic()
    if MODEL.exists():
        raise SystemExit(f"Refusing to overwrite {MODEL}")
    labels, dynamic_start, number_start = make_vocabulary(S.VOCAB)
    ids = {word: i for i, word in enumerate(labels)}
    layouts = shapes()
    count = sum(math.prod(shape) for shape in layouts.values())
    assert count == 250_000_000, count
    MODEL.mkdir(parents=True)
    file = MODEL / "weights.f32"
    storage = np.memmap(file, mode="w+", dtype="<f4", shape=(count,))
    storage[:] = 0
    tensors, metadata, offset = {}, {}, 0
    for name, shape in layouts.items():
        size = math.prod(shape)
        tensors[name] = storage[offset:offset + size].reshape(shape)
        metadata[name] = {"shape": list(shape), "offset_bytes": offset * 4, "nbytes": size * 4}
        offset += size
    gamma = S.ANCHOR_VALUE * math.sqrt(2 / S.WIDTH)
    for name, tensor in tensors.items():
        if name.endswith(("ln1.weight", "ln2.weight")) or name == "final_norm.weight":
            tensor[:] = gamma

    relation_words = {"is": 0, "can": 1, "needs": 2, "need": 2, "has": 3, "have": 3,
                      "produces": 4, "produce": 4, "exists": 6, "lives": 6, "located": 6}
    query_words = {"can": (1,), "need": (2,), "needs": (2,), "has": (3,), "have": (3,),
                   "produces": (4,), "produce": (4,), "describe": (5,), "color": (5,), "colour": (5,),
                   "where": (6,), "hello": (7,), "hi": (7,), "hey": (7,), "+": (8,), "plus": (8,),
                   "add": (8,), "sum": (8,), "calculate": (8,), "explain": (0, 9), "about": (0, 9), "more": (1,)}
    adjectives = {fact["trait"] for fact in FACTS.values() if "trait" in fact}
    for tid, label in enumerate(labels):
        row = tensors["token_embedding"][tid]
        put_pair(row, S.ANCHOR, S.ANCHOR_VALUE)
        put_pair(row, S.ONE, 1)
        for bit, value in enumerate(S.code(tid)):
            put_pair(row, S.CODE + bit, value)
        if tid == ids["<bos>"]:
            put_pair(row, S.BOS_FLAG, 1)
        if label in ("<user>", "<assistant>"):
            put_pair(row, S.ROLE_CANDIDATE, 1)
            put_pair(row, S.ROLE_USER, float(label == "<user>"))
        if label in ("<assistant>", "it"):
            put_pair(row, S.PHASE_CANDIDATE, 1)
            put_pair(row, S.PHASE_VALUE, float(label == "it"))
        if label in query_words or label == "<user>":
            put_pair(row, S.QUERY_CANDIDATE, 1)
            for q in query_words.get(label, (0,)):
                put_pair(row, S.QUERY_FLAGS + q, 1)
        if label in relation_words:
            put_pair(row, S.REL_FLAGS + relation_words[label], 1)
        if label in ("a", "an"):
            put_pair(row, S.ARTICLE, 1)
        if label in ("in", "on"):
            put_pair(row, S.PREPOSITION, 1)
        if label in adjectives:
            put_pair(row, S.ADJECTIVE, 1)
        put_pair(row, S.VOWEL, float(label[0] in "aeiou"))
        if label in FACTS or dynamic_start <= tid < number_start:
            put_pair(row, S.ENTITY, 1)
        if dynamic_start <= tid < number_start:
            put_pair(row, S.VOWEL, float(tid - dynamic_start < 64))
        if tid >= number_start:
            put_pair(row, S.NUMBER, 1)
            put_pair(row, S.NUMBER_VALUE, (tid - number_start) / 1000)
        if label in FACTS:
            values = FACTS[label]
            for relation, key in enumerate(("type", "can", "needs", "has", "produces", "trait", "location")):
                if key not in values:
                    continue
                base = S.slot(S.FACTS, relation)
                for bit, value in enumerate(S.code(ids[values[key]])):
                    put_pair(row, base + bit, value)
                put_pair(row, base + 14, 1)
                put_pair(row, base + 15, float(values[key][0] in "aeiou"))
    for position in range(S.CONTEXT):
        row = tensors["position_embedding"][position]
        put_pair(row, S.POS, position / 128)
        put_pair(row, S.POS_SQ, (position / 128) ** 2)
    for tid in range(S.VOCAB):
        add_reader(tensors["lm_head.weight"][tid], {S.OUTPUT + i: 8 * bit for i, bit in enumerate(S.code(tid))})

    c = Compiler(tensors, ids)
    # Block 0: causal role, clause phase, and three previous-token pointers.
    c.latest(0, 0, S.ROLE_CANDIDATE, [S.ROLE_USER], [S.USER_STATE])
    c.latest(0, 1, S.PHASE_CANDIDATE, [S.PHASE_VALUE], [S.PHASE_STATE])
    sources = list(range(S.CODE, S.CODE + 14)) + [S.ENTITY] + list(range(S.REL_FLAGS, S.REL_FLAGS + 7)) + [S.ARTICLE, S.PREPOSITION]
    for shift, destination in enumerate(S.PREV, 1):
        sharpness = 16
        queries = [{S.POS: 8 * 2 * sharpness * 128 ** 2, S.ONE: -8 * 2 * sharpness * shift * 128},
                   {S.ONE: -8 * sharpness * 128 ** 2}]
        c.attention(0, shift + 1, queries, [{S.POS: 1}, {S.POS_SQ: 1}], sources, list(range(destination, destination + 24)))
    for src, dst in ((S.ENTITY, S.LIVE_ENTITY), (S.QUERY_CANDIDATE, S.LIVE_QUERY), (S.NUMBER, S.LIVE_NUMBER)):
        c.gate(0, [src, S.USER_STATE], targets={dst: 1})

    # Block 1: prompt subject/facts, request, and first/last numeric operands.
    sources = list(range(S.CODE, S.CODE + 14)) + [S.ENTITY] + list(range(S.FACTS, S.FACTS + 112))
    destinations = list(range(S.SUBJECT, S.SUBJECT + 14)) + [S.SUBJECT_PRESENT] + list(range(S.CONTEXT_FACTS, S.CONTEXT_FACTS + 112))
    for chunk in range(2):
        c.latest(1, chunk, S.LIVE_ENTITY, sources[chunk * 64:(chunk + 1) * 64], destinations[chunk * 64:(chunk + 1) * 64])
    c.latest(1, 2, S.LIVE_QUERY, list(range(S.QUERY_FLAGS, S.QUERY_FLAGS + 10)), list(range(S.REQUEST, S.REQUEST + 10)))
    c.latest(1, 3, S.LIVE_NUMBER, [S.NUMBER_VALUE, S.POS, S.NUMBER], [S.NUM_FIRST, S.NUM_FIRST_POS, S.NUM_FIRST_PRESENT], first=True)
    c.latest(1, 4, S.LIVE_NUMBER, [S.NUMBER_VALUE, S.POS, S.NUMBER], [S.NUM_LAST, S.NUM_LAST_POS, S.NUM_LAST_PRESENT])
    cases = []
    p1, p2, p3 = S.PREV
    for relation in range(7):
        pos = [p1 + 15 + (0 if relation == 5 else relation), p2 + 14, S.USER_STATE]
        neg = []
        if relation == 0:
            neg = [S.ADJECTIVE]
        elif relation == 5:
            pos += [S.ADJECTIVE]
        cases.append((relation, p2, pos, neg))
    cases += [(0, p3, [p1 + 22, p2 + 15, p3 + 14, S.USER_STATE], []),
              (6, p3, [p1 + 23, p2 + 21, p3 + 14, S.USER_STATE], []),
              (6, p3, [p1 + 23, p2 + 15, p3 + 14, S.USER_STATE], [])]
    for number, (relation, subject, pos, neg) in enumerate(cases):
        c.gate(1, pos, neg, {S.GATES + number: 1})
        gate = S.GATES + number
        c.copy(2, subject, gate, S.FACT_SUBJECT, 14)
        c.copy(2, S.CODE, gate, S.FACT_OBJECT, 14)
        c.copy(2, S.VOWEL, gate, S.FACT_VOWEL)
        c.unit(2, {gate: 1}, {S.FACT_VALID: 1, S.FACT_RELATION + relation: 1})

    # Blocks 3-9: retrieve exact subject/relation facts from preceding user text.
    # A BOS sentinel defeats nonmatching memories instead of guessing a nearest fact.
    for relation in range(7):
        layer = 3 + relation
        queries = [{S.SUBJECT + bit: 8 * 64} for bit in range(14)] + [{S.ONE: 8 * 64}]
        keys = [{S.FACT_SUBJECT + bit: 1} for bit in range(14)] + [
            {S.FACT_RELATION + relation: 4, S.FACT_VALID: 1, S.BOS_FLAG: 18.5}]
        c.attention(layer, 0, queries, keys, list(range(S.FACT_OBJECT, S.FACT_OBJECT + 14)) + [S.FACT_VALID, S.FACT_VOWEL], list(range(S.RETURN, S.RETURN + 16)))
        destination = S.slot(S.CONTEXT_FACTS, relation)
        c.copy(layer, destination, S.RETURN + 14, destination, 16, scale=-1)
        for i in range(16):
            c.linear(layer, {S.RETURN + i: 1}, {destination + i: 1, S.RETURN + i: -1})

    # Block 10: current/previous word detectors and valid arithmetic bounds.
    c.clear(10, range(S.STATES, S.STATES + 37))
    for i, word in enumerate(S.STATE_WORDS):
        c.word(10, word, {S.STATES + i: 1})
    previous_words = ("<assistant>", "a", "an", "can", "needs", "has", "produces", "in", "is")
    for i, word in enumerate(previous_words, 17):
        c.word(10, word, {S.STATES + i: 1}, start=p1)
    for src, dst in ((S.NUM_FIRST, S.STATES + 26), (S.NUM_LAST, S.STATES + 27)):
        expression = {S.ONE: 1000, src: -1000}
        c.unit(10, expression, {dst: 1})
        c.unit(10, expression, {dst: -1}, threshold=1)
    difference = {S.NUM_LAST_POS: 128, S.NUM_FIRST_POS: -128}
    c.unit(10, difference, {S.STATES + 28: 1})
    c.unit(10, difference, {S.STATES + 28: -1}, threshold=1)

    # Block 11: next-token grammar and semantic selection, not sentence selection.
    c.clear(11, range(S.GATES, S.GATES + 28))
    state = {word: S.STATES + i for i, word in enumerate(S.STATE_WORDS)}
    for relation in range(7):
        c.gate(11, [state["<assistant>"], S.REQUEST + relation, S.slot(S.CONTEXT_FACTS, relation) + 14], targets={S.GATES: 1})
    c.gate(11, [state["<assistant>"], S.REQUEST + S.Q_HELLO], targets={S.GATES + 1: 1})
    c.emit(11, "hello", [state["<assistant>"], S.REQUEST + S.Q_HELLO])
    c.gate(11, [state["<assistant>"], S.REQUEST + S.Q_ADD, S.STATES + 26, S.STATES + 27,
                S.STATES + 28, S.NUM_FIRST_PRESENT, S.NUM_LAST_PRESENT], targets={S.GATES + 2: 1})
    c.linear(11, {S.NUM_FIRST: 1000, S.NUM_LAST: 1000}, {S.MATH_SUM: 1})
    for relation, word in enumerate(("is", "can", "needs", "has", "produces", "is", "exists")):
        c.emit(11, word, [S.STATES + 17, S.ENTITY, S.REQUEST + relation])
    for current, following in (("i", "do"), ("do", "not"), ("not", "know"), ("know", "."),
                               ("hello", "."), ("it", "can"), ("exists", "in")):
        c.emit(11, following, [state[current]])
    c.emit(11, "an", [state["is"], S.REQUEST + S.Q_CLASS, S.CONTEXT_FACTS + 15])
    c.emit(11, "a", [state["is"], S.REQUEST + S.Q_CLASS], [S.CONTEXT_FACTS + 15])
    for word in ("a", "an"):
        c.gate(11, [state[word]], targets={S.GATES + 3: 1})
    for word, relation in (("can", 1), ("needs", 2), ("has", 3), ("produces", 4), ("in", 6)):
        c.gate(11, [state[word]], targets={S.GATES + 3 + relation: 1})
    c.gate(11, [state["is"], S.REQUEST + S.Q_TRAIT], targets={S.GATES + 8: 1})
    for previous in range(18, 25):
        c.emit(11, ".", [S.STATES + previous])
    c.emit(11, ".", [S.STATES + 25, S.REQUEST + S.Q_TRAIT])
    c.emit(11, ".", [S.NUMBER, S.STATES + 17])
    c.emit(11, "it", [state["."], S.REQUEST + S.Q_MULTI, S.slot(S.CONTEXT_FACTS, 1) + 14], [S.PHASE_STATE])

    # Block 12: copy subject/object token codes into the output representation.
    c.emit(12, "i", [state["<assistant>"]], [S.GATES, S.GATES + 1, S.GATES + 2])
    c.copy(12, S.SUBJECT, S.GATES, S.OUTPUT, 14)
    for relation in range(7):
        c.copy(12, S.slot(S.CONTEXT_FACTS, relation), S.GATES + 3 + relation, S.OUTPUT, 14)
    c.clip(13, range(S.OUTPUT, S.OUTPUT + 14))

    # Blocks 14-16: a directly written piecewise-linear integer-to-token circuit.
    # Its input is the sum of numeric values retrieved by attention, not Python math.
    initial = S.code(number_start)
    c.unit(14, {S.ONE: 1}, {S.MATH_BITS + bit: value for bit, value in enumerate(initial)})
    for boundary in range(1, 1999):
        before, after = S.code(number_start + boundary - 1), S.code(number_start + boundary)
        delta = {S.MATH_BITS + bit: a - b for bit, (a, b) in enumerate(zip(after, before)) if a != b}
        c.unit(14, {S.MATH_SUM: 1}, delta, threshold=boundary - 1)
        c.unit(14, {S.MATH_SUM: 1}, {k: -v for k, v in delta.items()}, threshold=boundary)
    c.clip(15, range(S.MATH_BITS, S.MATH_BITS + 14))
    c.copy(16, S.MATH_BITS, S.GATES + 2, S.OUTPUT, 14)
    c.clip(17, range(S.OUTPUT, S.OUTPUT + 14))

    storage.flush()
    nonzero = 0
    for tensor in tensors.values():
        if not np.isfinite(tensor).all():
            raise RuntimeError("A nonfinite parameter was written")
        nonzero += int(np.count_nonzero(tensor))
    config = {
        "format": "direct-causal-transformer-f32-v1", "layers": S.LAYERS, "width": S.WIDTH,
        "heads": S.HEADS, "ffn_width": S.FFN, "context_length": S.CONTEXT, "vocabulary_size": S.VOCAB,
        "activation": "relu", "norm": "layer_norm", "norm_epsilon": 1e-5,
        "tied_embeddings": False, "dtype": "float32", "parameter_count": count,
        "weight_bytes": file.stat().st_size, "weights_file": file.name, "tensors": metadata,
        "vocabulary": labels, "dynamic_start": dynamic_start, "dynamic_count": 128,
        "number_start": number_start, "special": {"eos": ids["<eos>"], "bos": ids["<bos>"],
            "user": ids["<user>"], "assistant": ids["<assistant>"]},
    }
    (MODEL / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "parameter_count": count,
        "weight_bytes": file.stat().st_size, "padding_bytes": 0, "nonzero_parameters": nonzero,
        "nonzero_fraction": nonzero / count, "training_steps": 0, "optimizer_steps": 0,
        "pretrained_weights_used": False, "internet_used": False, "random_initialization_used": False,
        "complete_response_bank": False, "construction": "analytically compiled token-level circuits inside a causal transformer",
        "relu_units_used_per_block": c.used, "atomic_entity_records": len(FACTS),
        "sha256": {"weights.f32": sha(file), "config.json": sha(MODEL / "config.json")},
        "source_sha256": {name: sha(ROOT / name) for name in ("build.py", "spec.py", "knowledge.py", "offline.py")},
        "build_seconds": time.monotonic() - started,
        "limitations": ["Sparse directly programmed circuits; most standard transformer connections remain zero.",
                        "Small authored lexicon and fact vocabulary; general language understanding is unproven.",
                        "New input words are represented by fixed pointer-token slots.",
                        "Arithmetic circuit is intended for two non-negative operands no greater than 999.",
                        "Prompt facts use limited syntactic patterns; no weight updates occur during a conversation."],
    }
    (MODEL / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Created {file}: {count:,} parameters, {file.stat().st_size:,} bytes", flush=True)
    print(f"Nonzero parameters: {nonzero:,} ({nonzero / count:.3%}); no response bank", flush=True)
    print(f"ReLU units by block: {c.used}", flush=True)
    print(f"SHA-256: {manifest['sha256']['weights.f32']}", flush=True)


if __name__ == "__main__":
    main()
