"""Architecture and logical feature layout shared by compiler and diagnostics.

The inference runtime uses only the saved ordinary transformer tensors.
Each logical feature is encoded as the physical pair (+x, -x).
"""
WIDTH = 1024
HEADS = 16
HEAD_DIM = 64
FFN = 4096
LAYERS = 18
VOCAB = 8832
CONTEXT = 5048
ANCHOR_VALUE = 1_000_000.0
CODE_BITS = 14

ANCHOR, ONE, POS, POS_SQ = 0, 1, 2, 3
BOS_FLAG, MATH_SUM, LIVE_NUMBER = 4, 5, 6
OBJECT_ALLOWED = 7
CODE = 8
ENTITY, NUMBER, NUMBER_VALUE = 22, 23, 24
ROLE_CANDIDATE, ROLE_USER = 25, 26
PHASE_CANDIDATE, PHASE_VALUE = 27, 28
QUERY_CANDIDATE, QUERY_FLAGS = 29, 30
ARTICLE, PREPOSITION = 40, 41
REL_FLAGS = 42
ADJECTIVE, VOWEL = 49, 50
FACTS = 64  # Seven slots of [14 object-code bits, present, starts-with-vowel].
USER_STATE, PHASE_STATE = 176, 177
PREV = (180, 204, 228)  # code[14], entity, relation[7], article, preposition
LIVE_ENTITY, LIVE_QUERY = 252, 253
SUBJECT = 256
SUBJECT_PRESENT = 270
CONTEXT_FACTS = 272
REQUEST = 384
FACT_SUBJECT = 396
FACT_OBJECT = 410
FACT_RELATION, FACT_VALID, FACT_VOWEL = 424, 431, 432
RETURN = 434
OUTPUT = 450
MATH_BITS = 464
NUM_FIRST, NUM_LAST, NUM_FIRST_POS, NUM_LAST_POS, NUM_FIRST_PRESENT, NUM_LAST_PRESENT = range(478, 484)
GATES = 484
STATES = 396  # Reuses completed fact-event registers after retrieval layers.

RELATIONS = ("class", "can", "needs", "has", "produces", "trait", "location")
Q_CLASS, Q_CAN, Q_NEEDS, Q_HAS, Q_PRODUCES, Q_TRAIT, Q_LOCATION, Q_HELLO, Q_ADD, Q_MULTI = range(10)

STATE_WORDS = ("<assistant>", "i", "do", "not", "know", "hello", "is", "a", "an", "it", "can",
               "needs", "has", "produces", "exists", "in", ".")


def code(token):
    return [1.0 if token & (1 << bit) else -1.0 for bit in range(CODE_BITS)]


def slot(base, relation):
    return base + 16 * relation
