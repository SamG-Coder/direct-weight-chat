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
USER_START = 51
FACTS = 64  # Seven slots of [14 object-code bits, present, negative-polarity].
USER_STATE, PHASE_STATE = 52, 53
PREV = (176, 203, 230, 257)  # code[14], entity, relation[7], article, preposition, negation, auxiliary, number
LIVE_ENTITY, LIVE_QUERY = 54, 55
TOKEN_SUBTRACT, REQUEST_SUBTRACT, NEGATIVE_RESULT = 56, 57, 58
NEGATION, AUXILIARY, OUTPUT_NEGATIVE = 59, 60, 61
NUMBER_AFTER_FIRST, NUM_SECOND_POS = 62, 63
# Lifetimes matter: the four lookback buffers are dead after block 2 and become
# a seven-relation scratch buffer. The subject and walk cursor are independent.
RETURN = 176
CONTEXT_FACTS = 288
SUBJECT = 400
SUBJECT_PRESENT = 414
CURRENT, CLASS_VOWEL = 415, 429  # Article feature is computed after traversal.
REQUEST = 430
FACT_SUBJECT = 440
FACT_OBJECT = 454
FACT_RELATION, FACT_VALID, FACT_NEGATIVE = 468, 475, 476
OUTPUT = 256  # RETURN is dead after graph traversal.
MATH_BITS = 270
NUM_FIRST, NUM_LAST, NUM_FIRST_POS, NUM_LAST_POS, NUM_FIRST_PRESENT, NUM_LAST_PRESENT = range(478, 484)
GATES = 484
STATES = 440  # Reuses completed fact-event registers after retrieval layers.

# Inverted questions reuse lexical/role fields after their last readers.
# Block 0 writes three subject-prefix markers, block 1 compiles a query event,
# and block 2 binds the newest event (or a request-reset sentinel).
Q_PREFIX_CAN, Q_PREFIX_DOES, Q_PREFIX_IS = GATES + 25, GATES + 26, GATES + 27
Q_PREV1 = (25, 26, 27)
Q_PREV2 = (28, 29, 50)
Q_PREV3 = (42, 43, 44)
Q_EVENT_CANDIDATE, Q_EVENT_VALID, Q_EVENT_NEGATIVE = 49, 477, 52
Q_SUBJECT_TEMP = tuple(range(GATES + 10, GATES + 20)) + tuple(range(GATES + 22, GATES + 26))
Q_OBJECT, Q_MODE, Q_FOUND, Q_NEGATIVE = 30, 44, 45, 46
Q_EXPLICIT_FOUND, Q_EXPLICIT_NEGATIVE, Q_STATIC_FOUND = 47, 48, 49
Q_REL_TEMP = (25, 26, 27, 28, 29, 50, 54)
Q_QUERY_NEGATIVE = 55

# Clause boundaries and proposition composition. The reused coordinates are
# read for their old purpose before the layer that initializes their new one.
CLAUSE_BOUNDARY = 477  # Embedding -> block 0, then Q_EVENT_VALID.
CLAUSE_START = 414
LOCAL_ADJECTIVE = 414  # Clause position is finished after block 1.
CLAUSE_FIRST, LOCAL_AND, LOCAL_OR, LEXICAL_REQUEST = 284, 285, 286, 287
REQUEST_SUPPORTED = 6  # LIVE_NUMBER is finished after block-1 attention.
Q_EVENT_GROUP, Q_GROUP = 24, 3  # Numeric value and position square are dead.
Q_JOIN_AND, Q_JOIN_OR = 7, 24
Q_PREV_JOIN = (284, 287, 61)
Q_BIND_TEMP = tuple(range(GATES, GATES + 10)) + (GATES + 20, GATES + 21) + Q_REL_TEMP + (52, 56, 59, 60)
ATOM_TRUE, ATOM_FALSE, ATOM_UNKNOWN = 64, 65, 66
ANY_TRUE, ANY_FALSE, ANY_UNKNOWN, ANY_AND, ANY_OR, COMPOSED_KNOWN = range(67, 73)

RELATIONS = ("class", "can", "needs", "has", "produces", "trait", "location")
Q_CLASS, Q_CAN, Q_NEEDS, Q_HAS, Q_PRODUCES, Q_TRAIT, Q_LOCATION, Q_HELLO, Q_ADD, Q_MULTI = range(10)

STATE_WORDS = ("<assistant>", "i", "do", "not", "know", "hello", "is", "a", "an", "it", "can",
               "needs", "has", "produces", "exists", "in", ".")


def code(token):
    return [1.0 if token & (1 << bit) else -1.0 for bit in range(CODE_BITS)]


def slot(base, relation):
    return base + 16 * relation
