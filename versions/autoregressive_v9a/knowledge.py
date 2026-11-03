"""Atomic facts and a lexicon written for this experiment.

There are no completed responses, dialogue examples, sentence templates, or
corpus-derived statistics here. The compiler assigns these facts to token-vector
fields and constructs token-level attention and grammar circuits directly.
"""
FACTS = {
    "cat": dict(type="animal", can="move", needs="food", has="fur", trait="furry", location="land"),
    "dog": dict(type="animal", can="run", needs="food", has="fur", trait="furry", location="land"),
    "bird": dict(type="animal", can="fly", needs="food", has="wings", produces="eggs", trait="feathered"),
    "fish": dict(type="animal", can="swim", needs="water", has="fins", produces="eggs", trait="aquatic", location="water"),
    "horse": dict(type="animal", can="run", needs="food", has="hooves", location="land"),
    "snake": dict(type="animal", can="move", needs="food", has="scales", trait="flexible"),
    "bee": dict(type="insect", can="fly", needs="nectar", has="wings", trait="small"),
    "tree": dict(type="plant", can="grow", needs="sunlight", has="roots", produces="oxygen", trait="living", location="land"),
    "plant": dict(type="organism", can="grow", needs="water", has="cells", produces="oxygen", trait="living"),
    "flower": dict(type="structure", needs="water", has="petals", produces="pollen", trait="colorful"),
    "grass": dict(type="plant", can="grow", needs="water", has="roots", trait="green", location="land"),
    "water": dict(type="substance", can="flow", trait="liquid"),
    "ice": dict(type="solid", can="melt", trait="cold"),
    "fire": dict(type="reaction", can="spread", needs="fuel", produces="heat", trait="hot"),
    "sun": dict(type="star", can="shine", produces="light", trait="hot", location="space"),
    "moon": dict(type="satellite", can="orbit", has="craters", trait="rocky", location="space"),
    "computer": dict(type="machine", can="calculate", needs="electricity", has="processor", trait="programmable"),
    "robot": dict(type="machine", can="move", needs="energy", has="sensors", trait="programmable"),
    "cpu": dict(type="processor", can="compute", needs="electricity", produces="heat", trait="programmable"),
    "gpu": dict(type="processor", can="compute", needs="electricity", produces="heat", trait="parallel"),
    "neural network": dict(type="model", can="learn", has="weights", trait="numerical"),
    "engine": dict(type="machine", can="work", needs="energy", produces="motion"),
    "bicycle": dict(type="vehicle", can="move", has="wheels", location="land"),
    "car": dict(type="vehicle", can="move", needs="energy", has="wheels", location="land"),
    "boat": dict(type="vehicle", can="float", location="water"),
    "airplane": dict(type="vehicle", can="fly", needs="energy", has="wings"),
    "rock": dict(type="material", trait="solid"),
    "river": dict(type="waterway", can="flow", has="water", location="land"),
    "cloud": dict(type="collection", has="droplets", location="sky"),
    "rain": dict(type="precipitation", can="fall", has="water"),
    "wind": dict(type="motion", can="blow", has="air"),
    "book": dict(type="object", has="pages", trait="readable"),
    "magnet": dict(type="object", has="field", trait="magnetic"),
    "battery": dict(type="device", produces="electricity", has="chemicals"),
    "animal": dict(type="organism", needs="food", has="cells", trait="living"),
    "insect": dict(type="animal", has="legs", needs="food", trait="small"),
    "gravity": dict(type="interaction", trait="attractive"),
    "light": dict(type="radiation", can="travel", trait="electromagnetic"),
    "sound": dict(type="wave", can="travel", needs="medium"),
    "python": dict(type="language", trait="interpreted"),
}

GRAMMAR = """
i do not know hello hi hey is a an it can needs has produces exists in on . , ? !
what who where when why how which tell me about explain describe color colour
more please could would should will you your my we they he she their our its this
that these those are was were be been being and or but if then than for from to
of with without at as by also only very quite just any some all both each every
does did need have produce lives located live make made use uses using work works
add plus + = - * / calculate sum together two numbers number result answer briefly
long short detailed simple simply sentence sentences compare versus between
write original new poem story code program sort file explain again example example's
remember name called means mean definition difference different same other
thanks thank thankyou goodbye bye yes no true false first second third another
concept theory reasoning understand chat model weights training internet offline
help friend today tomorrow yesterday question earlier previous conversation
do not don't doesn't isn't aren't cannot can't won't hadn't hasn't must may might
think feel want like love important because therefore while before after during
now here there end start stop continue ignore instruction instructions repeat
given assume suppose fact facts following above below then thus enough much many
learn learned learning generate response responses compose own describe supported
""".split()


def make_vocabulary(size):
    labels = ["<eos>", "<bos>", "<user>", "<assistant>"]
    terms = set(GRAMMAR) | set(FACTS)
    for facts in FACTS.values():
        terms.update(facts.values())
    for term in sorted(terms):
        if term not in labels:
            labels.append(term)
    dynamic_start = len(labels)
    # New input words receive pointer-token IDs, not new/updated parameters.
    labels += [f"<copy_{i}>" for i in range(128)]
    number_start = len(labels)
    labels += [str(i) for i in range(size - len(labels))]
    assert len(labels) == size
    return labels, dynamic_start, number_start
