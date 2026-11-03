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
            self.unit(layer, expr({source + i: 1}, {gate: 4}), {destination + i: scale}, threshold=4)
            self.unit(layer, expr({source + i: -1}, {gate: 4}), {destination + i: -scale}, threshold=4)

    def copy_unless(self, layer, source, gate, suppress, destination, count=1, scale=1):
        for i in range(count):
            self.unit(layer, expr({source + i: 1}, {gate: 4}, {suppress: -4}), {destination + i: scale}, threshold=4)
            self.unit(layer, expr({source + i: -1}, {gate: 4}, {suppress: -4}), {destination + i: -scale}, threshold=4)

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

    relation_words = {"is": 0, "isn't": 0, "can": 1, "cannot": 1, "can't": 1, "needs": 2, "need": 2, "has": 3, "have": 3,
                      "produces": 4, "produce": 4, "exists": 6, "lives": 6, "located": 6}
    query_words = {"can": (1,), "cannot": (1,), "can't": (1,), "need": (2,), "needs": (2,), "has": (3,), "have": (3,),
                   "produces": (4,), "produce": (4,), "describe": (5,), "color": (5,), "colour": (5,),
                   "where": (6,), "hello": (7,), "hi": (7,), "hey": (7,), "+": (8,), "plus": (8,),
                   "add": (8,), "sum": (8,), "calculate": (8,), "-": (8,), "explain": (0, 9), "about": (0, 9), "more": (1,)}
    adjectives = {fact["trait"] for fact in FACTS.values() if "trait" in fact}
    object_words = set(FACTS) | {value for facts in FACTS.values() for value in facts.values()}
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
        if label == "-":
            put_pair(row, S.TOKEN_SUBTRACT, 1)
        if label in ("not", "cannot", "can't", "isn't", "doesn't"):
            put_pair(row, S.NEGATION, 1)
        if label in ("does", "doesn't"):
            put_pair(row, S.AUXILIARY, 1)
        if label in ("a", "an"):
            put_pair(row, S.ARTICLE, 1)
        if label in ("in", "on"):
            put_pair(row, S.PREPOSITION, 1)
        if label in adjectives:
            put_pair(row, S.ADJECTIVE, 1)
        put_pair(row, S.VOWEL, float(label[0] in "aeiou"))
        if label in FACTS or dynamic_start <= tid < number_start:
            put_pair(row, S.ENTITY, 1)
        if label in object_words or tid >= dynamic_start:
            put_pair(row, S.OBJECT_ALLOWED, 1)
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
                # Authored facts are positive. Slot 15 now stores polarity;
                # article choice is derived from the selected object's code.
    for position in range(S.CONTEXT):
        row = tensors["position_embedding"][position]
        put_pair(row, S.POS, position / 128)
        put_pair(row, S.POS_SQ, (position / 128) ** 2)
    for tid in range(S.VOCAB):
        add_reader(tensors["lm_head.weight"][tid], {S.OUTPUT + i: 8 * bit for i, bit in enumerate(S.code(tid))})

    c = Compiler(tensors, ids)
    # Block 0: causal role, clause phase, and four previous-token pointers.
    c.latest(0, 0, S.ROLE_CANDIDATE, [S.ROLE_USER], [S.USER_STATE])
    c.latest(0, 1, S.PHASE_CANDIDATE, [S.PHASE_VALUE], [S.PHASE_STATE])
    sources = list(range(S.CODE, S.CODE + 14)) + [S.ENTITY] + list(range(S.REL_FLAGS, S.REL_FLAGS + 7)) + [S.ARTICLE, S.PREPOSITION, S.NEGATION, S.AUXILIARY]
    for shift, destination in enumerate(S.PREV, 1):
        sharpness = 16
        queries = [{S.POS: 8 * 2 * sharpness * 128 ** 2, S.ONE: -8 * 2 * sharpness * shift * 128},
                   {S.ONE: -8 * sharpness * 128 ** 2}]
        c.attention(0, shift + 1, queries, [{S.POS: 1}, {S.POS_SQ: 1}], sources, list(range(destination, destination + 26)))
    c.latest(0, 6, S.ROLE_USER, [S.POS], [S.USER_START])
    for src, dst in ((S.ENTITY, S.LIVE_ENTITY), (S.QUERY_CANDIDATE, S.LIVE_QUERY), (S.NUMBER, S.LIVE_NUMBER)):
        c.gate(0, [src, S.USER_STATE], targets={dst: 1})

    # Block 1: prompt subject/facts, request, and first/last numeric operands.
    sources = list(range(S.CODE, S.CODE + 14)) + [S.ENTITY] + list(range(S.FACTS, S.FACTS + 112))
    destinations = list(range(S.SUBJECT, S.SUBJECT + 14)) + [S.SUBJECT_PRESENT] + list(range(S.CONTEXT_FACTS, S.CONTEXT_FACTS + 112))
    for chunk in range(2):
        c.latest(1, chunk, S.LIVE_ENTITY, sources[chunk * 64:(chunk + 1) * 64], destinations[chunk * 64:(chunk + 1) * 64])
    c.latest(1, 2, S.LIVE_QUERY, list(range(S.QUERY_FLAGS, S.QUERY_FLAGS + 10)) + [S.TOKEN_SUBTRACT],
             list(range(S.REQUEST, S.REQUEST + 10)) + [S.REQUEST_SUBTRACT])
    # Prefer the newest user turn, then its earliest number. The role boundary
    # is itself retrieved by attention, not parsed into arithmetic by the host.
    c.attention(1, 3, [{S.ONE: 8}], [{S.LIVE_NUMBER: 65536, S.USER_START: 16384 * 128, S.POS: -8 * 128}],
                [S.NUMBER_VALUE, S.POS, S.NUMBER], [S.NUM_FIRST, S.NUM_FIRST_POS, S.NUM_FIRST_PRESENT])
    c.latest(1, 4, S.LIVE_NUMBER, [S.NUMBER_VALUE, S.POS, S.NUMBER], [S.NUM_LAST, S.NUM_LAST_POS, S.NUM_LAST_PRESENT])
    cases = []
    p1, p2, p3, p4 = S.PREV
    for relation in range(7):
        pos = [p1 + 15 + (0 if relation == 5 else relation), p2 + 14, S.USER_STATE]
        neg = []
        if relation == 0:
            neg = [S.ADJECTIVE]
        elif relation == 5:
            pos += [S.ADJECTIVE]
        cases.append((relation, p2, pos, neg, p1 + 24))
        # X REL not OBJECT; including copular traits. Explicit auxiliaries
        # use separate patterns below, so this pattern excludes them.
        signed_pos = [p1 + 24, p2 + 15 + (0 if relation == 5 else relation), p3 + 14, S.USER_STATE]
        if relation == 5:
            signed_pos += [S.ADJECTIVE]
        signed_neg = [p1 + 25] + ([S.ADJECTIVE] if relation == 0 else [])
        cases.append((relation, p3, signed_pos, signed_neg, None))
        if relation in (2, 3, 4):
            # X does not REL OBJECT / X doesn't REL OBJECT.
            cases.append((relation, p4, [p1 + 15 + relation, p2 + 24, p3 + 25, p4 + 14, S.USER_STATE], [], None))
            cases.append((relation, p3, [p1 + 15 + relation, p2 + 24, p2 + 25, p3 + 14, S.USER_STATE], [], None))
    cases += [(0, p3, [p1 + 22, p2 + 15, p3 + 14, S.USER_STATE], [], p2 + 24),
              (6, p3, [p1 + 23, p2 + 21, p3 + 14, S.USER_STATE], [], p2 + 24),
              (6, p3, [p1 + 23, p2 + 15, p3 + 14, S.USER_STATE], [], p2 + 24),
              (0, p4, [p1 + 22, p2 + 24, p3 + 15, p4 + 14, S.USER_STATE], [p2 + 25], None),
              (6, p4, [p1 + 23, p2 + 24, p3 + 15, p4 + 14, S.USER_STATE], [p2 + 25], None)]
    # Compile syntax alternatives into shared relation/subject/polarity gates,
    # rather than allocating one semantic register per surface pattern.
    for relation, subject, pos, neg, polarity_source in cases:
        subject_gate = S.GATES + 7 + (p2, p3, p4).index(subject)
        targets = {S.GATES + relation: 1, subject_gate: 1, S.GATES + 21: 1}
        if polarity_source is None:
            targets[S.GATES + 20] = 1
        c.gate(1, pos + [S.OBJECT_ALLOWED], neg, targets)
        if polarity_source is not None:
            c.gate(1, pos + [S.OBJECT_ALLOWED, polarity_source], neg, {S.GATES + 20: 1})
    for i, subject in enumerate((p2, p3, p4)):
        c.copy(2, subject, S.GATES + 7 + i, S.FACT_SUBJECT, 14)
    c.copy(2, S.CODE, S.GATES + 21, S.FACT_OBJECT, 14)
    for relation in range(7):
        c.linear(2, {S.GATES + relation: 1}, {S.FACT_RELATION + relation: 1})
    c.linear(2, {S.GATES + 21: 1}, {S.FACT_VALID: 1})
    c.linear(2, {S.GATES + 20: 1}, {S.FACT_NEGATIVE: 1})

    # Release local parsing scratch and initialize a neural graph cursor.
    # All writes within this MLP read its pre-MLP input, so clearing the old
    # lookbacks here does not interfere with the fact copies above.
    c.clear(2, range(S.RETURN, S.RETURN + 112))
    for bit in range(14):
        c.linear(2, {S.SUBJECT + bit: 1}, {S.CURRENT + bit: 1})
    # Preserve the grammar's previous-token predicates before recycling PREV.
    # GATES[0:10] belong to fact parsing; [10:19] are free until block 11.
    previous_words = ("<assistant>", "a", "an", "can", "needs", "has", "produces", "in", "is")
    for i, word in enumerate(previous_words):
        c.word(2, word, {S.GATES + 10 + i: 1}, start=p1)
    c.word(2, "-", {S.GATES + 19: 1}, start=p1)
    for alias, canonical in (("cannot", "can"), ("need", "needs"), ("have", "has"), ("produce", "produces")):
        c.word(2, alias, {S.GATES + 10 + previous_words.index(canonical): 1}, start=p1)
    for i, word in enumerate(("do", "does", "not")):
        c.word(2, word, {S.GATES + 22 + i: 1}, start=p1)

    # Blocks 3,5,7,9: parallel associative lookup of seven predicates for the
    # current graph node, then follow its class edge. Four lookups support an
    # explicit terminal property after up to three unfamiliar class links.
    # Blocks 4,6,8: consult permanent atomic properties of the new cursor.
    # No prompt order or entity identity is compiled into this algorithm.
    for layer in range(3, 10):
        queries = [{S.CURRENT + bit: 8 * 8192} for bit in range(14)] + [{S.ONE: 8 * 8192}, {S.ONE: 8}]
        if layer % 2:
            for relation in range(7):
                scratch = S.slot(S.RETURN, relation)
                keys = [{S.FACT_SUBJECT + bit: 1} for bit in range(14)] + [
                    {S.FACT_RELATION + relation: 4, S.FACT_VALID: 1, S.BOS_FLAG: 18.5}, {S.POS: 128}]
                c.attention(layer, relation, queries, keys,
                            list(range(S.FACT_OBJECT, S.FACT_OBJECT + 14)) + [S.FACT_VALID, S.FACT_NEGATIVE],
                            list(range(scratch, scratch + 16)))
                destination = S.slot(S.CONTEXT_FACTS, relation)
                if layer == 3:
                    # A direct prompt assertion overrides the same subject's
                    # permanent fact. Later ancestors fill missing properties.
                    c.copy(layer, destination, scratch + 14, destination, 16, scale=-1)
                    for i in range(16):
                        c.linear(layer, {scratch + i: 1}, {destination + i: 1})
                else:
                    c.copy_unless(layer, scratch, scratch + 14, destination + 14, destination, 16)
            c.clear(layer, range(S.CURRENT, S.CURRENT + 14))
            # Negative class membership is stored as information but cannot
            # license traversal of the positive class edge.
            c.copy_unless(layer, S.RETURN, S.RETURN + 14, S.RETURN + 15, S.CURRENT, 14)
        else:
            # A sentinel needs a zero identity code, or its own token bits
            # add to the threshold and can defeat an exact memory match.
            keys = [{S.CODE + bit: 1, S.BOS_FLAG: -S.code(ids["<bos>"])[bit]} for bit in range(14)] + [{S.BOS_FLAG: 13}, {S.POS: 128}]
            for chunk in range(2):
                begin, end = chunk * 64, min(112, (chunk + 1) * 64)
                c.attention(layer, chunk, queries, keys,
                            list(range(S.FACTS + begin, S.FACTS + end)),
                            list(range(S.RETURN + begin, S.RETURN + end)))
            # The answer's class remains the closest class, not a distant one.
            for relation in range(1, 7):
                scratch, destination = S.slot(S.RETURN, relation), S.slot(S.CONTEXT_FACTS, relation)
                c.copy_unless(layer, scratch, scratch + 14, destination + 14, destination, 16)
        c.clear(layer, range(S.RETURN, S.RETURN + 112))

    # Block 10: current/previous word detectors and valid arithmetic bounds.
    # Reinitialize coordinates after scratch reuse so tiny roundoff cannot
    # break the all-zero output code's deterministic EOS tie.
    c.clear(10, range(S.OUTPUT, S.OUTPUT + 14))
    c.clear(10, range(S.MATH_BITS, S.MATH_BITS + 14))
    c.clear(10, range(S.STATES, S.STATES + 37))
    for i, word in enumerate(S.STATE_WORDS):
        c.word(10, word, {S.STATES + i: 1})
    for alias, canonical in (("cannot", "can"), ("need", "needs"), ("have", "has"), ("produce", "produces")):
        c.word(10, alias, {S.STATES + S.STATE_WORDS.index(canonical): 1})
    for i in range(len(previous_words)):
        c.linear(10, {S.GATES + 10 + i: 1}, {S.STATES + 17 + i: 1})
    c.word(10, "-", {S.STATES + 29: 1})
    c.linear(10, {S.GATES + 19: 1}, {S.STATES + 30: 1})
    c.word(10, "does", {S.STATES + 31: 1})
    for i in range(3):
        c.linear(10, {S.GATES + 22 + i: 1}, {S.STATES + 32 + i: 1})
    for relation in range(7):
        c.gate(10, [S.REQUEST + relation, S.slot(S.CONTEXT_FACTS, relation) + 15],
               [S.PHASE_STATE], {S.OUTPUT_NEGATIVE: 1})
    c.gate(10, [S.PHASE_STATE, S.slot(S.CONTEXT_FACTS, 1) + 15], targets={S.OUTPUT_NEGATIVE: 1})
    # A lexical property classifier, not a response bank. Pointer IDs carry
    # the tokenizer's vowel class; all decisions still occur in this MLP.
    for tid, label in enumerate(labels):
        if label[0] in "aeiou" or dynamic_start <= tid < dynamic_start + 64:
            c.word(10, label, {S.CLASS_VOWEL: 1}, start=S.CONTEXT_FACTS)
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
    # General bounded arithmetic: addition or |a-b|, plus a signed-result bit.
    # No table of operand pairs is constructed. Numeric values arrive through
    # attention and all operations below become matrix/ReLU tensor operations.
    c.unit(11, {S.NUM_FIRST: 1000, S.NUM_LAST: 1000, S.REQUEST_SUBTRACT: -2000}, {S.MATH_SUM: 1})
    for sign in (1, -1):
        c.unit(11, {S.NUM_FIRST: sign * 1000, S.NUM_LAST: -sign * 1000, S.REQUEST_SUBTRACT: 2000},
               {S.MATH_SUM: 1}, threshold=2000)
    negative = {S.NUM_LAST: 1000, S.NUM_FIRST: -1000, S.REQUEST_SUBTRACT: 2000}
    c.unit(11, negative, {S.NEGATIVE_RESULT: 1}, threshold=2000)
    c.unit(11, negative, {S.NEGATIVE_RESULT: -1}, threshold=2001)
    for relation, word in enumerate(("is", "can", "needs", "has", "produces", "is", "exists")):
        c.emit(11, word, [S.STATES + 17, S.ENTITY, S.REQUEST + relation], [S.OUTPUT_NEGATIVE])
        negative_word = "is" if relation in (0, 5, 6) else "cannot" if relation == 1 else "does"
        c.emit(11, negative_word, [S.STATES + 17, S.ENTITY, S.REQUEST + relation, S.OUTPUT_NEGATIVE])
    for current, following in (("i", "do"), ("do", "not"), ("know", "."),
                               ("hello", "."), ("exists", "in")):
        c.emit(11, following, [state[current]])
    c.emit(11, "know", [state["not"], S.STATES + 32])
    c.emit(11, "not", [S.STATES + 31])
    c.emit(11, "not", [state["is"], S.OUTPUT_NEGATIVE])
    c.emit(11, "can", [state["it"]], [S.OUTPUT_NEGATIVE])
    c.emit(11, "cannot", [state["it"], S.OUTPUT_NEGATIVE])
    for relation, word in ((2, "need"), (3, "have"), (4, "produce")):
        c.emit(11, word, [state["not"], S.STATES + 33, S.REQUEST + relation])
    for article, positive, negative in (("an", [S.CLASS_VOWEL], []), ("a", [], [S.CLASS_VOWEL])):
        c.emit(11, article, [state["is"], S.REQUEST + S.Q_CLASS] + positive, [S.OUTPUT_NEGATIVE] + negative)
        c.emit(11, article, [state["not"], S.REQUEST + S.Q_CLASS, S.OUTPUT_NEGATIVE] + positive, negative)
    c.emit(11, "in", [state["not"], S.REQUEST + S.Q_LOCATION, S.OUTPUT_NEGATIVE])
    for word in ("a", "an"):
        c.gate(11, [state[word]], targets={S.GATES + 3: 1})
    for word, relation in (("can", 1), ("needs", 2), ("has", 3), ("produces", 4), ("in", 6)):
        c.gate(11, [state[word]], targets={S.GATES + 3 + relation: 1})
    c.gate(11, [state["is"], S.REQUEST + S.Q_TRAIT], [S.OUTPUT_NEGATIVE], {S.GATES + 8: 1})
    c.gate(11, [state["not"], S.REQUEST + S.Q_TRAIT, S.OUTPUT_NEGATIVE], targets={S.GATES + 8: 1})
    for previous in range(18, 25):
        c.emit(11, ".", [S.STATES + previous])
    c.emit(11, ".", [S.STATES + 25, S.REQUEST + S.Q_TRAIT], [S.OUTPUT_NEGATIVE])
    c.emit(11, ".", [S.STATES + 34, S.REQUEST + S.Q_TRAIT, S.OUTPUT_NEGATIVE])
    c.emit(11, ".", [S.NUMBER, S.STATES + 17])
    c.emit(11, ".", [S.NUMBER, S.STATES + 30])
    c.emit(11, "it", [state["."], S.REQUEST + S.Q_MULTI, S.slot(S.CONTEXT_FACTS, 1) + 14], [S.PHASE_STATE])

    # Block 12: copy subject/object token codes into the output representation.
    # A negative result emits the minus token, then its magnitude on the next
    # forward pass. The host only formats the spacing of these generated tokens.
    c.emit(12, "-", [S.GATES + 2, S.NEGATIVE_RESULT])
    c.gate(12, [S.GATES + 2], [S.NEGATIVE_RESULT], {S.GATES + 21: 1})
    c.gate(12, [S.STATES + 29, S.REQUEST + S.Q_ADD], targets={S.GATES + 21: 1})
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
    c.copy(16, S.MATH_BITS, S.GATES + 21, S.OUTPUT, 14)
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
