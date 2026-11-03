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
from itertools import product
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

    def pattern(self, layer, words, targets, positive=(), negative=(), any_of=()):
        """Conjoin lexical identities and Boolean flags into one ReLU gate."""
        expression = expr(*({start + i: bit for i, bit in enumerate(S.code(self.ids[word]))}
                            for start, word in words),
                          *({f: 2} for f in positive), *({f: -2} for f in negative),
                          *({f: 2 for f in group} for group in any_of))
        # Each any_of group is a one-hot lexical category, so its sum is Boolean.
        self.unit(layer, expression, targets,
                  threshold=S.CODE_BITS * len(words) + 2 * (len(positive) + len(any_of)) - 1)

    def copy(self, layer, source, gate, destination, count=1, scale=1):
        for i in range(count):
            self.unit(layer, expr({source + i: 1}, {gate: 4}), {destination + i: scale}, threshold=4)
            self.unit(layer, expr({source + i: -1}, {gate: 4}), {destination + i: -scale}, threshold=4)

    def copy_unless(self, layer, source, gate, suppress, destination, count=1, scale=1):
        for i in range(count):
            self.unit(layer, expr({source + i: 1}, {gate: 4}, {suppress: -4}), {destination + i: scale}, threshold=4)
            self.unit(layer, expr({source + i: -1}, {gate: 4}, {suppress: -4}), {destination + i: -scale}, threshold=4)

    def copy_when(self, layer, source, destination, positive=(), negative=(), count=1, scale=1):
        """Copy a bounded code under a conjunction, using two ReLUs per bit."""
        conditions = expr(*({f: 4} for f in positive), *({f: -4} for f in negative))
        for i in range(count):
            self.unit(layer, expr({source + i: 1}, conditions), {destination + i: scale}, threshold=4 * len(positive))
            self.unit(layer, expr({source + i: -1}, conditions), {destination + i: -scale}, threshold=4 * len(positive))

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

    def previous(self, layer, head, shift, sources, destinations):
        """Address p-shift using two bounded sinusoidal phases.

        The fast phase separates adjacent tokens; the slow phase separates
        its 64-token aliases. Within this context, every nonzero address error
        has a score gap of at least 32768*(1-cos(2*pi/128)) > 39. Neither the
        features nor the scores grow quadratically with absolute position.
        """
        queries, keys = [], []
        for i, period in enumerate(S.LOOKBACK_PERIODS):
            cosine, sine = S.LOOKBACK_POSITION[2*i:2*i+2]
            angle = 2 * math.pi * shift / period
            a = 8 * S.LOOKBACK_SHARPNESS * math.cos(angle)
            b = 8 * S.LOOKBACK_SHARPNESS * math.sin(angle)
            queries += [{cosine: a, sine: b}, {cosine: -b, sine: a}]
            keys += [{cosine: 1}, {sine: 1}]
        self.attention(layer, head, queries, keys, sources, destinations)

    def latest_in_clause(self, layer, head, candidate, group, group_scale, sources, destinations, query_group=None):
        """Prefer an event in this causal clause, otherwise its zero sentinel.

        Group IDs are nondecreasing boundary positions. A one-position older
        group loses more score than any validity/recency bonus can supply.
        Cancel the query's common group offset before adding token recency.
        """
        penalty = 131072 * group_scale
        self.attention(layer, head, [{S.ONE: 8}, {group if query_group is None else query_group: -8 * penalty}, {S.ONE: 8}],
                       [{group: penalty}, {S.ONE: 1}, {candidate: 65536, S.POS: 8 * 128}],
                       sources, destinations)


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
    request_introducers = set(('what where describe explain tell about color colour more '
                              'hello hi hey add plus sum calculate a an can does is + -').split())
    nonassertive_heads = set(('what who where when why how which can does is are could would should will may if '
                             'tell describe explain write compare calculate add sum').split())
    for tid, label in enumerate(labels):
        row = tensors["token_embedding"][tid]
        put_pair(row, S.ANCHOR, S.ANCHOR_VALUE)
        put_pair(row, S.ONE, 1)
        for bit, value in enumerate(S.code(tid)):
            put_pair(row, S.CODE + bit, value)
        if tid == ids["<bos>"]:
            put_pair(row, S.BOS_FLAG, 1)
        if label in ("<bos>", "<user>", ".", "?", "!"):
            put_pair(row, S.CLAUSE_BOUNDARY, 1)
        if label in request_introducers or label in FACTS or tid >= dynamic_start:
            put_pair(row, S.LEXICAL_REQUEST, 1)
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
        if label in nonassertive_heads:
            put_pair(row, S.QUESTION_HEAD_WORD, 1)
        if label in FACTS or dynamic_start <= tid < number_start:
            put_pair(row, S.ENTITY, 1)
        if label in object_words or tid >= dynamic_start:
            put_pair(row, S.OBJECT_ALLOWED, 1)
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
        for i, period in enumerate(S.LOOKBACK_PERIODS):
            angle = 2 * math.pi * position / period
            put_pair(row, S.LOOKBACK_POSITION[2*i], math.cos(angle))
            put_pair(row, S.LOOKBACK_POSITION[2*i+1], math.sin(angle))
    for tid in range(S.VOCAB):
        add_reader(tensors["lm_head.weight"][tid], {S.OUTPUT + i: 8 * bit for i, bit in enumerate(S.code(tid))})
    # Valid token codes have an exact nearest-neighbor margin of 16 logits.
    # A one-logit EOS prior resolves an otherwise all-zero code robustly,
    # instead of allowing tiny accumulated floating-point residue to emit text.
    tensors["lm_head.bias"][ids["<eos>"]] = 1

    c = Compiler(tensors, ids)
    # Block 0: causal role, clause phase, and four previous-token pointers.
    c.latest(0, 0, S.ROLE_CANDIDATE, [S.ROLE_USER], [S.USER_STATE])
    c.latest(0, 1, S.PHASE_CANDIDATE, [S.PHASE_VALUE], [S.PHASE_STATE])
    sources = list(range(S.CODE, S.CODE + 14)) + [S.ENTITY] + list(range(S.REL_FLAGS, S.REL_FLAGS + 7)) + [S.ARTICLE, S.PREPOSITION, S.NEGATION, S.AUXILIARY, S.NUMBER]
    for shift, destination in enumerate(S.PREV, 1):
        c.previous(0, shift + 1, shift, sources, list(range(destination, destination + 27)))
    c.latest(0, 6, S.ROLE_USER, [S.POS], [S.USER_START])
    c.latest(0, 7, S.CLAUSE_BOUNDARY, [S.POS], [S.CLAUSE_START])
    for src, dst in ((S.ENTITY, S.LIVE_ENTITY), (S.QUERY_CANDIDATE, S.LIVE_QUERY), (S.NUMBER, S.LIVE_NUMBER)):
        c.gate(0, [src, S.USER_STATE], targets={dst: 1})
    p1, p2, p3, p4 = S.PREV
    c.linear(0, {S.QUESTION_HEAD_WORD: 1}, {S.QUESTION_HEAD_TEMP: 1})
    # Predicate events are separate from object events. A later coordinated
    # object can inherit an explicitly changed predicate even when its subject
    # is omitted. A local 'and not OBJECT' does not change this shared register.
    for relation in range(7):
        flag = S.REL_FLAGS + relation
        c.gate(0, [flag, S.USER_STATE], targets={S.LEXICAL_RELATION + relation: 1, S.LEXICAL_VALID: 1})
        c.gate(0, [flag, S.USER_STATE, S.NEGATION], targets={S.LEXICAL_NEGATIVE: 1})
        c.word(0, 'not', {S.LEXICAL_RELATION + relation: 1, S.LEXICAL_VALID: 1, S.LEXICAL_NEGATIVE: 1},
               positive=[p1 + 15 + relation, S.USER_STATE])
        if relation in (2, 3, 4):
            c.gate(0, [flag, S.USER_STATE, p1 + 24], targets={S.LEXICAL_NEGATIVE: 1})
    c.gate(0, [S.PREPOSITION, S.USER_STATE], targets={S.LEXICAL_RELATION + 6: 1, S.LEXICAL_VALID: 1})
    c.gate(0, [S.PREPOSITION, S.USER_STATE, p1 + 24], targets={S.LEXICAL_NEGATIVE: 1})
    for relation in (0, 6):
        c.gate(0, [S.PREPOSITION, S.USER_STATE, p1 + 15 + relation, p2 + 24],
               [p1 + 24], {S.LEXICAL_NEGATIVE: 1})
    for boundary in ("<user>", ".", "?", "!"):
        c.word(0, boundary, {S.CLAUSE_FIRST: 1}, start=p1, positive=[S.USER_STATE])
    # Transparent modifiers do not choose the request kind. Read the lexical
    # head after a bounded modifier prefix, including unsupported commands.
    # Enumerating this grammar compiles identity conjunctions, not responses.
    modifiers = ("please", "briefly", "simply", "just")
    for length in range(1, len(S.PREV)):
        for prefix in product(modifiers, repeat=length):
            for boundary in ("<user>", ".", "?", "!"):
                words = list(zip(S.PREV[:length], reversed(prefix))) + [(S.PREV[length], boundary)]
                c.pattern(0, words, {S.CLAUSE_FIRST: 1}, positive=[S.USER_STATE])
    c.word(0, "and", {S.LOCAL_AND: 1}, positive=[S.USER_STATE])
    c.word(0, "or", {S.LOCAL_OR: 1}, positive=[S.USER_STATE])
    # A +/- token immediately after a number is binary. Other occurrences
    # are unary and must not replace the expression's binary operator.
    for sign in ("+", "-"):
        c.word(0, sign, {S.LIVE_QUERY: -1}, positive=[S.USER_STATE], negative=[p1 + 26])
    # Conditional sign reversal using the previous token's identity. Numeric
    # embeddings are bounded by the vocabulary (< 16), so a 16-unit identity
    # penalty suppresses every nonmatching or binary-minus case.
    unary_minus = expr({S.NUMBER_VALUE: 1, p2 + 26: -16},
                       {p1 + i: 16 * bit for i, bit in enumerate(S.code(ids["-"]))})
    c.unit(0, unary_minus, {S.NUMBER_VALUE: -2}, threshold=16 * S.CODE_BITS)

    # Bind an inverted-question subject only after a clause boundary. Local
    # identity conjunctions distinguish "Can X ..." from "X can Y" without
    # host parsing. The same code works for any entity/pointer-token identity.
    for verb, marker in (("can", S.Q_PREFIX_CAN), ("does", S.Q_PREFIX_DOES), ("is", S.Q_PREFIX_IS)):
        for article in (False, True):
            verb_at, boundary_at = (p2, p3) if article else (p1, p2)
            for boundary in ("<user>", ".", "?", "!", "and", "or"):
                conditions = [S.ENTITY, S.USER_STATE] + ([p1 + 22] if article else [])
                expression = expr({verb_at + i: bit for i, bit in enumerate(S.code(ids[verb]))},
                                  {boundary_at + i: bit for i, bit in enumerate(S.code(ids[boundary]))},
                                  *({f: 2} for f in conditions))
                c.unit(0, expression, {marker: 1}, threshold=28 + 2 * len(conditions) - 1)
            verb_index = 1 if article else 0
            for length in range(1, len(S.PREV) - verb_index - 1):
                for prefix in product(modifiers, repeat=length):
                    for boundary in ("<user>", ".", "?", "!", "and", "or"):
                        words = ([(S.PREV[verb_index], verb)] +
                                 list(zip(S.PREV[verb_index + 1:verb_index + 1 + length], reversed(prefix))) +
                                 [(S.PREV[verb_index + 1 + length], boundary)])
                        conditions = [S.ENTITY, S.USER_STATE] + ([p1 + 22] if article else [])
                        c.pattern(0, words, {marker: 1}, positive=conditions)
    # These embedding features have already been consumed by block-0 readers.
    c.clear(0, list(range(25, 30)) + list(range(40, 49)) + [50, S.CLAUSE_BOUNDARY])

    # Block 1: prompt subject/facts, request, and first/last numeric operands.
    sources = list(range(S.CODE, S.CODE + 14)) + list(range(S.FACTS, S.FACTS + 112))
    destinations = list(range(S.SUBJECT, S.SUBJECT + 14)) + list(range(S.CONTEXT_FACTS, S.CONTEXT_FACTS + 112))
    for chunk in range(2):
        c.latest(1, chunk, S.LIVE_ENTITY, sources[chunk * 64:(chunk + 1) * 64], destinations[chunk * 64:(chunk + 1) * 64])
    c.latest(1, 2, S.LIVE_QUERY, list(range(S.QUERY_FLAGS, S.QUERY_FLAGS + 10)) + [S.TOKEN_SUBTRACT],
             list(range(S.REQUEST, S.REQUEST + 10)) + [S.REQUEST_SUBTRACT])
    # Prefer the newest user turn, then its earliest number. The role boundary
    # is itself retrieved by attention, not parsed into arithmetic by the host.
    c.attention(1, 3, [{S.ONE: 8}], [{S.LIVE_NUMBER: 65536, S.USER_START: 16384 * 128, S.POS: -8 * 128}],
                [S.NUMBER_VALUE, S.POS, S.NUMBER], [S.NUM_FIRST, S.NUM_FIRST_POS, S.NUM_FIRST_PRESENT])
    c.latest(1, 4, S.LIVE_NUMBER, [S.NUMBER_VALUE, S.POS, S.NUMBER], [S.NUM_LAST, S.NUM_LAST_POS, S.NUM_LAST_PRESENT])
    for shift, destinations in ((1, S.Q_PREV1), (2, S.Q_PREV2), (3, S.Q_PREV3)):
        c.previous(1, 4 + shift, shift, [S.Q_PREFIX_CAN, S.Q_PREFIX_DOES, S.Q_PREFIX_IS], list(destinations))
    # Temporary read into Q_EVENT_VALID; its old contents are cleared in this
    # MLP before the query parser writes the actual event-validity bit.
    c.latest(1, 8, S.CLAUSE_FIRST, [S.LEXICAL_REQUEST, S.QUESTION_HEAD_TEMP],
             [S.Q_EVENT_VALID, S.QUESTION_CLAUSE])
    c.latest_in_clause(1, 9, S.LEXICAL_VALID, S.CLAUSE_START, 128,
                       list(range(S.LEXICAL_RELATION, S.LEXICAL_RELATION + 7)) + [S.LEXICAL_NEGATIVE, S.LEXICAL_VALID],
                       list(range(S.PREDICATE_CONTEXT, S.PREDICATE_CONTEXT + 7)) + [S.PREDICATE_NEGATIVE, S.PREDICATE_VALID])
    # The clause of the preceding token remains available at a punctuation
    # boundary, where the current token already begins a new clause ID.
    c.previous(1, 10, 1, [S.CLAUSE_START], [S.ASSERTION_GROUP_TEMP])
    # Query events use CURRENT temporarily; they are read by block 2 before
    # it initializes the normal graph cursor. No completed answer is stored.
    c.clear(1, list(range(25, 51)) + [52, 54, 55, S.Q_PREFIX_CAN, S.Q_PREFIX_DOES, S.Q_PREFIX_IS])
    c.clear(1, [S.Q_EVENT_VALID, S.REQUEST_SUPPORTED, S.Q_EVENT_GROUP, S.Q_GROUP])
    c.linear(1, {S.Q_EVENT_VALID: 1}, {S.REQUEST_SUPPORTED: 1})
    c.linear(1, {S.CLAUSE_START: 1 / 64}, {S.Q_EVENT_GROUP: 1})
    c.clear(1, [S.LOCAL_ADJECTIVE])
    c.linear(1, {S.ADJECTIVE: 1}, {S.LOCAL_ADJECTIVE: 1})
    c.linear(1, {S.CLAUSE_FIRST: 1}, {S.Q_EVENT_CANDIDATE: 1})
    c.clear(1, [S.CLAUSE_FIRST, S.LEXICAL_REQUEST])
    c.clear(1, [S.QUESTION_HEAD_TEMP, S.QUESTION_CLAUSE] + list(range(S.LEXICAL_RELATION, S.LEXICAL_NEGATIVE + 1)))
    # Only the first two old lexical buffers are read by block 2. Reuse the
    # remaining ones for a delayed event and its previous-clause address.
    c.clear(1, range(S.PREV[2], S.PREV[3] + 27))
    # Preserve the natural position/128 scale. Dividing by another 64 before
    # writing over a +/-1 residual loses low bits in float32 cancellation;
    # multiplying that error by a clause penalty can overwhelm recency.
    c.linear(1, {S.ASSERTION_GROUP_TEMP: 1}, {S.ASSERTION_PREVIOUS_GROUP: 1})
    c.clear(1, [S.ASSERTION_GROUP_TEMP])
    for terminator in ('and', 'or', 'but', '.', '?', '!', ',', '<user>', '<assistant>', '<eos>'):
        c.word(1, terminator, {S.ASSERTION_TERMINATOR: 1})
    for offset, destination in zip(S.PREV[:3], S.Q_PREV_JOIN):
        for word in ("and", "or"):
            c.word(1, word, {destination: 1}, start=offset)
    # The same semantic predicate field serves affirmative and negative
    # questions. Negation is an independent bit, not a separate answer rule.
    query_cases = [(1, p1, [S.Q_PREV1[0]], [], False),
                   (1, p2, [S.Q_PREV2[0], p1 + 24], [p1 + 25], True)]
    for relation in (2, 3, 4):
        query_cases += [(relation, p2, [S.Q_PREV2[1], p1 + 15 + relation], [], False),
                        (relation, p3, [S.Q_PREV3[1], p1 + 15 + relation, p2 + 24], [p2 + 25], True)]
    for relation in (0, 5):
        adjective = [S.ADJECTIVE] if relation == 5 else []
        not_adjective = [S.ADJECTIVE] if relation == 0 else []
        query_cases += [(relation, p1, [S.Q_PREV1[2]] + adjective, not_adjective, False),
                        (relation, p2, [S.Q_PREV2[2], p1 + 24] + adjective, not_adjective + [p1 + 25], True)]
    query_cases += [(0, p2, [S.Q_PREV2[2], p1 + 22], [], False),
                    (6, p2, [S.Q_PREV2[2], p1 + 23], [], False),
                    (0, p3, [S.Q_PREV3[2], p1 + 22, p2 + 24], [p2 + 25], True),
                    (6, p3, [S.Q_PREV3[2], p1 + 23, p2 + 24], [p2 + 25], True)]
    for relation, subject, conditions, excluded, negated in query_cases:
        conditions += [S.OBJECT_ALLOWED, S.USER_STATE]
        targets = {S.Q_EVENT_VALID: 1, S.Q_EVENT_CANDIDATE: 1, S.REQUEST + relation: 1}
        if negated:
            targets[S.Q_EVENT_NEGATIVE] = 1
        c.gate(1, conditions, excluded, targets)
        c.copy_when(1, subject, S.CURRENT, positive=conditions, negative=excluded, count=14)
        c.copy_when(1, S.REQUEST, S.REQUEST, positive=conditions, negative=excluded, count=10, scale=-1)
    for reset in ("<bos>", "<user>", "what", "where", "describe", "explain"):
        c.word(1, reset, {S.Q_EVENT_CANDIDATE: 1}, negative=[S.CLAUSE_FIRST])
    # Mark user numbers after the turn's first number. The clipped ramp is
    # evaluated neurally; no host-side token counting or expression parser.
    after_first = {S.POS: 128, S.NUM_FIRST_POS: -128, S.NUMBER: 8192, S.USER_STATE: 8192}
    c.unit(1, after_first, {S.NUMBER_AFTER_FIRST: 1}, threshold=16384)
    c.unit(1, after_first, {S.NUMBER_AFTER_FIRST: -1}, threshold=16385)
    cases = []
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
    for relation in (2, 3, 4):
        cases += [(relation, p3, [p1 + 22, p2 + 15 + relation, p3 + 14, S.USER_STATE], [], p2 + 24),
                  (relation, p4, [p1 + 22, p2 + 24, p3 + 15 + relation, p4 + 14, S.USER_STATE], [p2 + 25], None),
                  (relation, p4, [p1 + 22, p2 + 15 + relation, p3 + 24, p3 + 25, p4 + 14, S.USER_STATE], [], None)]
    for relation in (0, 6):
        cases.append((6, p4, [p1 + 22, p2 + 23, p3 + 15 + relation, p4 + 14, S.USER_STATE], [], p3 + 24))
    # Full assertions are ready after block 1. Block 2 can retrieve their
    # subject for coordinated objects instead of following the most recent
    # entity token (which might be an object or a later question subject).
    for relation, subject, pos, neg, polarity_source in cases:
        pos = pos + [S.OBJECT_ALLOWED]
        neg = neg + [S.Q_PREV2[1], S.QUESTION_CLAUSE]
        targets = {S.FACT_RELATION + relation: 1, S.FACT_VALID: 1}
        if polarity_source is None:
            targets[S.FACT_NEGATIVE] = 1
        c.gate(1, pos, neg, targets)
        c.copy_when(1, subject, S.FACT_SUBJECT, pos, neg, count=14)
        c.copy_when(1, S.CODE, S.FACT_OBJECT, pos, neg, count=14)
        if polarity_source is not None:
            c.gate(1, pos + [polarity_source], neg, {S.FACT_NEGATIVE: 1})

    # Shared-object syntax produces provisional metadata, not an independently
    # valid fact. Its subject and validity must come from a full assertion in
    # this same clause at block 2. Each layout is a reusable lexical/category
    # conjunction, independent of entity identities or expected test answers.
    rels = lambda offset: tuple(range(offset + 15, offset + 22))
    layouts = [
        (p1, [], [], (), False),                              # and OBJECT
        (p2, [p1 + 24], [p1 + 25] + list(rels(p1)), (), True), # and not OBJECT
        (p2, [p1 + 22], [], (), False),                       # and a OBJECT
        (p2, [p1 + 23], [], (), False),                       # and in OBJECT
        (p3, [p2 + 24, p1 + 22], [p2 + 25] + list(rels(p2)), (), True),
        (p3, [p2 + 24, p1 + 23], [p2 + 25] + list(rels(p2)), (), True),
        (p2, [], [], (rels(p1),), False),                     # and REL OBJECT
        (p3, [p2 + 24], [p2 + 25] + list(rels(p2)), (rels(p1),), False),
        (p3, [p2 + 25], [], (rels(p1),), False),               # and doesn't REL OBJECT
        (p4, [p3 + 25, p2 + 24], [p2 + 25], (rels(p1),), False),
        (p3, [p1 + 22], [], (rels(p2),), False),
        (p3, [p1 + 23], [], (rels(p2),), False),
        (p4, [p2 + 24, p1 + 22], [p2 + 25], (rels(p3),), False),
        (p4, [p2 + 24, p1 + 23], [p2 + 25], (rels(p3),), False),
    ]
    for join_at, positive, negative, any_of, local_negative in layouts:
        for lexical_relation in (0, 1, 2, 3, 4, 6):
            for relation in ((0, 5) if lexical_relation == 0 else (lexical_relation,)):
                pos = positive + [S.PREDICATE_CONTEXT + lexical_relation, S.PREDICATE_VALID, S.OBJECT_ALLOWED, S.USER_STATE]
                neg = negative + [S.QUESTION_CLAUSE]
                if lexical_relation == 0:
                    (pos if relation == 5 else neg).append(S.ADJECTIVE)
                targets = {S.ASSERTION_CONTINUATION: 1, S.FACT_RELATION + relation: 1}
                if local_negative:
                    targets[S.FACT_NEGATIVE] = 1
                c.pattern(1, [(join_at, 'and')], targets, pos, neg, any_of)
                if not local_negative:
                    c.pattern(1, [(join_at, 'and')], {S.FACT_NEGATIVE: 1},
                              pos + [S.PREDICATE_NEGATIVE], neg, any_of)
    c.latest(2, 0, S.Q_EVENT_CANDIDATE,
             list(range(S.CURRENT, S.CURRENT + 14)) + list(range(S.CODE, S.CODE + 14)) + [S.Q_EVENT_VALID] + list(range(S.REQUEST, S.REQUEST + 7)) + [S.Q_EVENT_NEGATIVE, S.Q_EVENT_GROUP],
             list(S.Q_SUBJECT_TEMP) + list(range(S.Q_OBJECT, S.Q_OBJECT + 14)) + [S.Q_MODE] + list(S.Q_REL_TEMP) + [S.Q_QUERY_NEGATIVE, S.Q_GROUP])
    # Use the previous-token clause coordinate for both query and key. At a
    # period the current clause ID has advanced, but its delayed object and
    # its frame still belong to the clause just completed. Full fact events
    # occur after a subject/verb, so their previous/current clause IDs agree.
    c.latest_in_clause(2, 1, S.FACT_VALID, S.ASSERTION_PREVIOUS_GROUP, 128,
                       list(range(S.FACT_SUBJECT, S.FACT_SUBJECT + 14)) + [S.FACT_VALID],
                       list(range(S.CURRENT, S.CURRENT + 14)) + [S.ASSERTION_FRAME_VALID])
    # The token after a provisional object decides whether it is complete.
    # In 'X has wings and Y needs water', Y is followed by a verb and must not
    # be committed as X's object. This delay is causal and uses generic tensor
    # lookback; no Python lookahead, answer correction or fact dictionary.
    c.previous(2, 2, 1,
                [S.ASSERTION_CONTINUATION] + list(range(S.FACT_RELATION, S.FACT_RELATION + 7)) +
                [S.FACT_NEGATIVE] + list(range(S.CODE, S.CODE + 14)),
                list(range(S.ASSERTION_PREVIOUS, S.ASSERTION_PREVIOUS + 23)))
    continuation = [S.ASSERTION_PREVIOUS, S.ASSERTION_FRAME_VALID, S.ASSERTION_TERMINATOR]
    c.copy_when(2, S.CURRENT, S.FACT_SUBJECT, continuation, [S.Q_MODE], count=14)
    c.copy_when(2, S.ASSERTION_PREVIOUS + 9, S.FACT_OBJECT, continuation, [S.Q_MODE], count=14)
    c.copy_when(2, S.ASSERTION_PREVIOUS + 1, S.FACT_RELATION, continuation, [S.Q_MODE], count=7)
    c.copy_when(2, S.ASSERTION_PREVIOUS + 8, S.FACT_NEGATIVE, continuation, [S.Q_MODE])
    c.gate(2, continuation, [S.Q_MODE], {S.FACT_VALID: 1})

    # Release local parsing scratch and initialize a neural graph cursor.
    # All writes within this MLP read its pre-MLP input, so clearing the old
    # lookbacks here does not interfere with the fact copies above.
    c.clear(2, range(S.RETURN, S.RETURN + 112))
    c.clear(2, S.LOOKBACK_POSITION)
    c.clear(2, range(S.CURRENT, S.CURRENT + 14))
    c.copy_unless(2, S.SUBJECT, S.ONE, S.Q_MODE, S.CURRENT, 14)
    c.copy(2, S.SUBJECT, S.Q_MODE, S.SUBJECT, 14, scale=-1)
    for bit, source in enumerate(S.Q_SUBJECT_TEMP):
        c.copy(2, source, S.Q_MODE, S.SUBJECT + bit)
        c.copy(2, source, S.Q_MODE, S.CURRENT + bit)
    c.copy(2, S.CONTEXT_FACTS, S.Q_MODE, S.CONTEXT_FACTS, 112, scale=-1)
    c.copy(2, S.REQUEST, S.Q_MODE, S.REQUEST, 10, scale=-1)
    for relation, source in enumerate(S.Q_REL_TEMP):
        c.copy(2, source, S.Q_MODE, S.REQUEST + relation)
    c.clear(2, range(S.Q_OBJECT, S.Q_OBJECT + 14))
    c.copy(2, S.Q_OBJECT, S.Q_MODE, S.Q_OBJECT, 14)
    # Shared-subject continuations inherit the predicate unless a new one is
    # explicitly present. A locally written not applies only to that atom.
    continuations = [(None, [S.Q_PREV_JOIN[0]], [], False),
                     (None, [S.Q_PREV_JOIN[1], p1 + 24], [p1 + 25], True),
                     (0, [S.Q_PREV_JOIN[1], p1 + 22], [], False),
                     (6, [S.Q_PREV_JOIN[1], p1 + 23], [], False),
                     (0, [S.Q_PREV_JOIN[2], p1 + 22, p2 + 24], [p2 + 25], True),
                     (6, [S.Q_PREV_JOIN[2], p1 + 23, p2 + 24], [p2 + 25], True)]
    for relation in range(7):
        extra = [S.LOCAL_ADJECTIVE] if relation == 5 else []
        excluded = [S.LOCAL_ADJECTIVE] if relation == 0 else []
        rel_flag = p1 + 15 + (0 if relation == 5 else relation)
        continuations += [(relation, [S.Q_PREV_JOIN[1], rel_flag] + extra, excluded, False),
                          (relation, [S.Q_PREV_JOIN[2], rel_flag, p2 + 24] + extra, excluded + [p2 + 25], True)]
    for relation, conditions, excluded, negated in continuations:
        conditions += [S.Q_MODE, S.OBJECT_ALLOWED]
        targets = {S.Q_EVENT_VALID: 1, S.Q_EVENT_CANDIDATE: 1}
        if relation is not None:
            targets[S.REQUEST + relation] = 1
            for r, source in enumerate(S.Q_REL_TEMP):
                c.copy_when(2, source, S.REQUEST + r, conditions, excluded, scale=-1)
        if negated:
            targets[S.Q_QUERY_NEGATIVE] = 1
        c.gate(2, conditions, excluded, targets)
        c.copy_when(2, S.Q_OBJECT, S.Q_OBJECT, conditions, excluded, count=14, scale=-1)
        c.copy_when(2, S.CODE, S.Q_OBJECT, conditions, excluded, count=14)
        c.copy_when(2, S.Q_QUERY_NEGATIVE, S.Q_QUERY_NEGATIVE, conditions, excluded, scale=-1)
    c.clear(2, [S.Q_JOIN_AND, S.Q_JOIN_OR, S.OUTPUT_NEGATIVE])
    c.gate(2, [S.LOCAL_AND, S.Q_MODE], targets={S.Q_JOIN_AND: 1})
    c.gate(2, [S.LOCAL_OR, S.Q_MODE], targets={S.Q_JOIN_OR: 1})
    c.clear(2, list(S.Q_SUBJECT_TEMP) + list(S.Q_BIND_TEMP))
    # Preserve the grammar's previous-token predicates before recycling PREV.
    # GATES[0:10] held lexical predicate/frame reads; [10:19] now hold grammar.
    previous_words = ("<assistant>", "a", "an", "can", "needs", "has", "produces", "in", "is")
    for i, word in enumerate(previous_words):
        c.word(2, word, {S.GATES + 10 + i: 1}, start=p1)
    c.word(2, "-", {S.GATES + 19: 1}, start=p1)
    for alias, canonical in (("cannot", "can"), ("need", "needs"), ("have", "has"), ("produce", "produces")):
        c.word(2, alias, {S.GATES + 10 + previous_words.index(canonical): 1}, start=p1)
    for i, word in enumerate(("do", "does", "not")):
        c.word(2, word, {S.GATES + 22 + i: 1}, start=p1)

    # Block 3 binds the newest atom, including shared-subject continuations.
    # Blocks 3,5,7,9: parallel associative lookup of seven predicates for the
    # current graph node, then follow its class edge. Four lookups support an
    # explicit terminal property after up to three unfamiliar class links.
    # Blocks 4,6,8: consult permanent atomic properties of the new cursor.
    # No prompt order or entity identity is compiled into this algorithm.
    for layer in range(3, 11):
        if layer == 3:
            c.attention(3, 7, [{S.ONE: 8}],
                        [{S.NUMBER_AFTER_FIRST: 65536, S.USER_START: 16384 * 128, S.POS: -8 * 128}],
                        [S.POS], [S.NUM_SECOND_POS])
            c.latest(3, 8, S.Q_EVENT_CANDIDATE,
                     list(range(S.Q_OBJECT, S.Q_OBJECT + 14)) + list(range(S.REQUEST, S.REQUEST + 7)) + [S.Q_QUERY_NEGATIVE, S.Q_MODE],
                     list(S.Q_BIND_TEMP))
            c.clear(3, range(S.Q_OBJECT, S.Q_OBJECT + 14))
            for bit, source in enumerate(S.Q_BIND_TEMP[:14]):
                c.linear(3, {source: 1}, {S.Q_OBJECT + bit: 1})
            mode = S.Q_BIND_TEMP[-1]
            c.copy(3, S.REQUEST, mode, S.REQUEST, 10, scale=-1)
            for relation, source in enumerate(S.Q_BIND_TEMP[14:21]):
                c.copy(3, source, mode, S.REQUEST + relation)
            c.clear(3, [S.Q_QUERY_NEGATIVE, S.Q_MODE, S.Q_EVENT_CANDIDATE])
            c.linear(3, {S.Q_BIND_TEMP[-2]: 1}, {S.Q_QUERY_NEGATIVE: 1})
            c.linear(3, {mode: 1}, {S.Q_MODE: 1})
            c.clear(3, S.Q_BIND_TEMP)
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
        elif layer < 10:
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
        if layer % 2 == 0:
            # A separate associative head asks for an exact proposition,
            # retaining all event objects rather than one object per predicate.
            truth_queries = ([{S.SUBJECT + bit: 8 * 8192} for bit in range(14)] +
                             [{S.Q_OBJECT + bit: 8 * 8192} for bit in range(14)] +
                             [{S.REQUEST + r: 8 * 8192 * 4} for r in range(7)] +
                             [{S.ONE: 8 * 8192}, {S.ONE: 8}])
            truth_keys = ([{S.FACT_SUBJECT + bit: 1} for bit in range(14)] +
                          [{S.FACT_OBJECT + bit: 1} for bit in range(14)] +
                          [{S.FACT_RELATION + r: 1} for r in range(7)] +
                          [{S.FACT_VALID: 1, S.BOS_FLAG: 32}, {S.POS: 128}])
            c.attention(layer, 2, truth_queries, truth_keys, [S.FACT_VALID, S.FACT_NEGATIVE],
                        [S.Q_EXPLICIT_FOUND, S.Q_EXPLICIT_NEGATIVE])
            # Permanent atomic properties use the same subject/object match;
            # a missing value never constitutes evidence of a negative fact.
            for relation in range(7):
                slot = S.slot(S.FACTS, relation)
                static_queries = ([{S.SUBJECT + bit: 8 * 8192} for bit in range(14)] +
                                  [{S.Q_OBJECT + bit: 8 * 8192} for bit in range(14)] +
                                  [{S.REQUEST + relation: 8 * 8192}, {S.ONE: 8 * 8192}, {S.ONE: 8}])
                static_keys = ([{S.CODE + bit: 1, S.BOS_FLAG: -S.code(ids["<bos>"])[bit]} for bit in range(14)] +
                               [{slot + bit: 1} for bit in range(14)] +
                               [{S.ONE: 4, S.BOS_FLAG: -4}, {slot + 14: 1, S.BOS_FLAG: 32}, {S.POS: 128}])
                c.attention(layer, 3 + relation, static_queries, static_keys, [slot + 14], [S.Q_STATIC_FOUND])
            c.copy_unless(layer, S.Q_EXPLICIT_FOUND, S.Q_EXPLICIT_FOUND, S.Q_FOUND, S.Q_FOUND, 2)
            c.gate(layer, [S.Q_STATIC_FOUND], [S.Q_EXPLICIT_FOUND, S.Q_FOUND], {S.Q_FOUND: 1})
            c.clear(layer, [S.Q_EXPLICIT_FOUND, S.Q_EXPLICIT_NEGATIVE, S.Q_STATIC_FOUND])
            if layer < 10:
                c.copy(layer, S.SUBJECT, S.Q_MODE, S.SUBJECT, 14, scale=-1)
                c.copy(layer, S.CURRENT, S.Q_MODE, S.SUBJECT, 14)
        if layer < 10:
            c.clear(layer, range(S.RETURN, S.RETURN + 112))

    # Block 10: current/previous word detectors and valid arithmetic bounds.
    # Reinitialize coordinates after scratch reuse so tiny roundoff cannot
    # break the all-zero output code's deterministic EOS tie.
    c.clear(10, range(S.OUTPUT, S.OUTPUT + 14))
    c.clear(10, range(S.MATH_BITS, S.MATH_BITS + 14))
    c.clear(10, range(S.STATES, S.STATES + 37))
    c.clear(10, range(S.ATOM_TRUE, S.COMPOSED_KNOWN + 1))
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
        for sign in (1, -1):
            expression = {S.ONE: 1000, src: sign * 1000}
            c.unit(10, expression, {dst: 1})
            c.unit(10, expression, {dst: -1}, threshold=1)
        c.unit(10, {S.ONE: 1}, {dst: -1})
    difference = {S.NUM_LAST_POS: 128, S.NUM_FIRST_POS: -128}
    c.unit(10, difference, {S.STATES + 35: 1})
    c.unit(10, difference, {S.STATES + 35: -1}, threshold=1)
    # Exactly two operands: the earliest number after the first must also be
    # the last. Larger counts become nonpositive validity, never an answer.
    c.unit(10, {S.ONE: 1}, {S.STATES + 28: 1})
    for sign in (1, -1):
        c.unit(10, {S.NUM_LAST_POS: sign * 128, S.NUM_SECOND_POS: -sign * 128}, {S.STATES + 28: -1})
    current_turn = {S.NUM_FIRST_POS: 128, S.USER_START: -128}
    c.unit(10, current_turn, {S.STATES + 36: 1})
    c.unit(10, current_turn, {S.STATES + 36: -1}, threshold=1)
    # Form the signed result before the next layer extracts magnitude/sign.
    c.linear(10, {S.NUM_FIRST: 1000, S.NUM_LAST: 1000}, {S.MATH_SUM: 1})
    c.copy(10, S.NUM_LAST, S.REQUEST_SUBTRACT, S.MATH_SUM, scale=-2000)

    # Block 11: next-token grammar and semantic selection, not sentence selection.
    c.clear(11, range(S.GATES, S.GATES + 28))
    state = {word: S.STATES + i for i, word in enumerate(S.STATE_WORDS)}
    for relation in range(7):
        c.gate(11, [state["<assistant>"], S.REQUEST_SUPPORTED, S.REQUEST + relation, S.slot(S.CONTEXT_FACTS, relation) + 14], [S.Q_MODE], targets={S.GATES: 1})
    # Each atom's own token position carries its resolved three-valued truth.
    # Later positions may carry the latest atom for decoding, but are not
    # additional evidence. Negation operates before connective reduction.
    known_atom = [S.Q_EVENT_VALID, S.Q_MODE, S.Q_FOUND]
    c.gate(11, known_atom, [S.Q_NEGATIVE, S.Q_QUERY_NEGATIVE], {S.ATOM_TRUE: 1})
    c.gate(11, known_atom + [S.Q_NEGATIVE, S.Q_QUERY_NEGATIVE], targets={S.ATOM_TRUE: 1})
    c.gate(11, known_atom + [S.Q_NEGATIVE], [S.Q_QUERY_NEGATIVE], {S.ATOM_FALSE: 1})
    c.gate(11, known_atom + [S.Q_QUERY_NEGATIVE], [S.Q_NEGATIVE], {S.ATOM_FALSE: 1})
    c.gate(11, [S.Q_EVENT_VALID, S.Q_MODE], [S.Q_FOUND], {S.ATOM_UNKNOWN: 1})
    for answer in ("yes", "no"):
        c.word(11, answer, {S.OUTPUT + i: bit for i, bit in enumerate(S.code(ids["."]))}, positive=[S.Q_MODE])
    c.gate(11, [state["<assistant>"], S.REQUEST_SUPPORTED, S.REQUEST + S.Q_HELLO], targets={S.GATES + 1: 1})
    c.emit(11, "hello", [state["<assistant>"], S.REQUEST_SUPPORTED, S.REQUEST + S.Q_HELLO])
    c.gate(11, [state["<assistant>"], S.REQUEST_SUPPORTED, S.REQUEST + S.Q_ADD, S.STATES + 26, S.STATES + 27,
                S.STATES + 28, S.STATES + 35, S.STATES + 36, S.NUM_FIRST_PRESENT, S.NUM_LAST_PRESENT], targets={S.GATES + 2: 1})
    # General bounded arithmetic: addition or |a-b|, plus a signed-result bit.
    # No table of operand pairs is constructed. Numeric values arrive through
    # attention and all operations below become matrix/ReLU tensor operations.
    c.clear(11, [S.MATH_SUM])
    c.unit(11, {S.MATH_SUM: 1}, {S.MATH_SUM: 1})
    c.unit(11, {S.MATH_SUM: -1}, {S.MATH_SUM: 1})
    c.unit(11, {S.MATH_SUM: -1}, {S.NEGATIVE_RESULT: 1})
    c.unit(11, {S.MATH_SUM: -1}, {S.NEGATIVE_RESULT: -1}, threshold=1)
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
        c.emit(11, article, [state["not"], S.STATES + 25, S.REQUEST + S.Q_CLASS, S.OUTPUT_NEGATIVE] + positive, negative)
    c.emit(11, "in", [state["not"], S.STATES + 25, S.REQUEST + S.Q_LOCATION, S.OUTPUT_NEGATIVE])
    for word in ("a", "an"):
        c.gate(11, [state[word]], targets={S.GATES + 3: 1})
    for word, relation in (("can", 1), ("needs", 2), ("has", 3), ("produces", 4), ("in", 6)):
        c.gate(11, [state[word]], targets={S.GATES + 3 + relation: 1})
    c.gate(11, [state["is"], S.REQUEST + S.Q_TRAIT], [S.OUTPUT_NEGATIVE], {S.GATES + 8: 1})
    c.gate(11, [state["not"], S.STATES + 25, S.REQUEST + S.Q_TRAIT, S.OUTPUT_NEGATIVE], targets={S.GATES + 8: 1})
    for previous in range(18, 25):
        c.emit(11, ".", [S.STATES + previous])
    c.emit(11, ".", [S.STATES + 25, S.REQUEST + S.Q_TRAIT], [S.OUTPUT_NEGATIVE])
    c.emit(11, ".", [S.STATES + 34, S.REQUEST + S.Q_TRAIT, S.OUTPUT_NEGATIVE])
    c.emit(11, ".", [S.NUMBER, S.STATES + 17])
    c.emit(11, ".", [S.NUMBER, S.STATES + 30])
    c.emit(11, "it", [state["."], S.REQUEST_SUPPORTED, S.REQUEST + S.Q_MULTI, S.slot(S.CONTEXT_FACTS, 1) + 14], [S.PHASE_STATE])

    # Block 12: copy subject/object token codes into the output representation.
    # A negative result emits the minus token, then its magnitude on the next
    # forward pass. The host only formats the spacing of these generated tokens.
    # Monotone clause IDs dominate truth flags: reductions can only select the
    # newest active clause. Within it, an existential flag wins over zero.
    for head, (source, destination) in enumerate(zip(
            (S.ATOM_TRUE, S.ATOM_FALSE, S.ATOM_UNKNOWN, S.Q_JOIN_AND, S.Q_JOIN_OR),
            (S.ANY_TRUE, S.ANY_FALSE, S.ANY_UNKNOWN, S.ANY_AND, S.ANY_OR))):
        c.attention(12, head, [{S.ONE: 8}], [{S.Q_GROUP: 8192 * 65536, source: 8192}],
                    [source], [destination])
    base = [state["<assistant>"], S.Q_MODE, S.REQUEST_SUPPORTED]
    truth_outputs = [
        ("yes", [S.ANY_TRUE], [S.ANY_AND, S.ANY_OR]),
        ("no", [S.ANY_FALSE], [S.ANY_AND, S.ANY_OR]),
        ("yes", [S.ANY_AND, S.ANY_TRUE], [S.ANY_OR, S.ANY_FALSE, S.ANY_UNKNOWN]),
        ("no", [S.ANY_AND, S.ANY_FALSE], [S.ANY_OR]),
        ("yes", [S.ANY_OR, S.ANY_TRUE], [S.ANY_AND]),
        ("no", [S.ANY_OR, S.ANY_FALSE], [S.ANY_AND, S.ANY_TRUE, S.ANY_UNKNOWN]),
    ]
    for word, positive, negative in truth_outputs:
        c.emit(12, word, base + positive, negative)
        c.gate(12, base + positive, negative, {S.COMPOSED_KNOWN: 1})
    c.emit(12, "-", [S.GATES + 2, S.NEGATIVE_RESULT])
    c.gate(12, [S.GATES + 2], [S.NEGATIVE_RESULT], {S.GATES + 21: 1})
    c.gate(12, [S.STATES + 29, S.REQUEST + S.Q_ADD], targets={S.GATES + 21: 1})
    c.emit(12, "i", [state["<assistant>"]], [S.GATES, S.GATES + 1, S.GATES + 2, S.Q_MODE])
    c.copy(12, S.SUBJECT, S.GATES, S.OUTPUT, 14)
    for relation in range(7):
        c.copy(12, S.slot(S.CONTEXT_FACTS, relation), S.GATES + 3 + relation, S.OUTPUT, 14)
    c.clip(13, range(S.OUTPUT, S.OUTPUT + 14))
    c.emit(13, "i", [state["<assistant>"], S.Q_MODE], [S.COMPOSED_KNOWN])

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
                        "Arithmetic circuit is intended for two signed operands with magnitudes no greater than 999.",
                        "Object-specific truth questions and logical negation support seven predicates under a limited inversion grammar.",
                        "Proposition composition uses three-valued reductions for homogeneous AND or OR; mixed-operator syntax and arbitrary requests remain unsupported.",
                        "Shared AND assertion objects use clause-local neural subject/predicate/polarity binding with a bounded grammar.",
                        "Coordinated objects are committed at a following boundary token; question/directive clauses are excluded from assertions.",
                        "Local lookbacks use bounded periodic address features; broader semantic accuracy still requires separate context tests.",
                        "Disjunction, conditional rules, multiple-parent traversal and unrestricted sentence generation remain unsupported.",
                        "Prompt facts use limited syntactic patterns; no weight updates occur during a conversation."],
    }
    (MODEL / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Created {file}: {count:,} parameters, {file.stat().st_size:,} bytes", flush=True)
    print(f"Nonzero parameters: {nonzero:,} ({nonzero / count:.3%}); no response bank", flush=True)
    print(f"ReLU units by block: {c.used}", flush=True)
    print(f"SHA-256: {manifest['sha256']['weights.f32']}", flush=True)


if __name__ == "__main__":
    main()
