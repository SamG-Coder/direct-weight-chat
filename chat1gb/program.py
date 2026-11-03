"""Human-readable source for the neural program, not a training corpus.

The compiler translates declared lexical OR gates, rule AND gates, byte memories,
and arithmetic circuits directly into numbers. There is no fitting step.
Inference does not import this file or read its response strings.
"""
import re


def words(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\w\s]", text.lower())


def canonical(term):
    return term if term.startswith("@") else " ".join(words(term))


def program():
    groups, replies, rules = {}, [], []

    def group(name, terms):
        values = sorted({canonical(term) for term in terms})
        if name in groups and groups[name] != values:
            raise ValueError(f"Conflicting concept: {name}")
        groups[name] = values
        return name

    def reply(name, text, required=(), forbidden=(), priority=20, write_name=False, clear_name=False):
        rid = len(replies)
        replies.append({"name": name, "text": text, "write_name": write_name, "clear_name": clear_name})
        rules.append({"reply": rid, "required": list(required), "forbidden": list(forbidden), "priority": priority})
        return rid

    group("hello", ["hello", "hi", "hey", "g'day", "good morning", "good evening"])
    group("goodbye", ["goodbye", "bye", "see you", "good night"])
    group("thanks", ["thanks", "thank you", "cheers", "appreciate it"])
    group("how_are_you", ["how are you", "how's it going", "how do you feel"])
    group("identity", ["who are you", "what are you", "your name", "what model"])
    group("capabilities", ["what can you do", "your capabilities", "help me get started", "how can you help"])
    group("limits", ["limitations", "your limits", "can you think", "are you intelligent", "are you conscious"])
    group("offline", ["internet", "offline", "browse", "online", "download"])
    group("training", ["were you trained", "your training", "pretrained", "pre trained", "no training"])
    group("weights", ["your weights", "your parameters", "one gigabyte", "1gb", "1 gb", "model size"])
    group("fallback_help", ["help", "help me", "suggest a topic"])
    group("more", ["more", "elaborate", "expand", "detail", "details", "go on", "continue"])
    group("simple", ["simpler", "simply", "simple terms", "eli5", "explain like", "plain language", "easier"])
    group("another", ["another", "different one", "one more"])
    group("name_candidate", ["@name_candidate"])
    group("has_name", ["@has_name"])
    group("ask_name", ["my name", "remember me", "who am i"])
    group("forget", ["forget my name", "forget me", "clear my name"])
    group("addition", ["@addition"])
    group("unsupported_math", ["@unsupported_math"])

    # The first rule is the learned-knowledge boundary, not a fabricated answer.
    reply("unknown", "I don't have a reliable answer for that in my hand-written neural program. "
          "I can discuss basic programming, neural networks, science, and project planning; "
          "I can also add two non-negative integers. Try a direct question about one of those topics.", priority=-375)
    reply("hello", "Hello! What would you like to talk about?", ["hello"], priority=5)
    reply("hello_named", "Hello, {name}! What would you like to talk about?", ["hello", "has_name"], priority=8)
    reply("goodbye", "Goodbye! Thanks for the conversation.", ["goodbye"], priority=10)
    reply("thanks", "You're welcome. We can explore another question whenever you're ready.", ["thanks"], priority=8)
    reply("how_are_you", "I'm ready to chat. I don't have feelings or experiences, but I can work through the topics encoded in my weights. How are you doing?", ["how_are_you"], priority=15)
    reply("identity", "I'm the local 1 GB neural chat experiment. My weights were written directly, without training or pretrained checkpoints. I have a small, explicit set of conversation skills.", ["identity"], priority=25)
    reply("capabilities", "I can explain basic programming, neural networks, and science; suggest a project plan; tell a short joke or poem; remember your name in this session; and add two non-negative integers up to 12 digits each. Ask a follow-up such as 'simpler' or 'tell me more' after a supported explanation.", ["capabilities"], priority=30)
    reply("limits", "My knowledge and conversation patterns were deliberately encoded. I can combine those gates with a little session memory and a neural addition circuit, but I cannot reason or converse as broadly as a trained language model. I should say when a request is outside my program.", ["limits"], priority=30)
    reply("offline", "This model runs locally. It does not browse, call a remote model, or download anything while chatting. I cannot check current events or live information.", ["offline"], priority=28)
    reply("training", "There was no training: no dataset fitting, optimizer, gradients, or pretrained weights. A compiler converted explicit conversation rules, character memories, and arithmetic operations into neural-network parameters.", ["training"], priority=35)
    reply("weights", "My parameter file contains exactly 1,000,000,000 bytes of float32 weights: 250,000,000 parameters. Most implement an overcomplete dense projection. That size does not represent 250 million independently learned facts or general language understanding.", ["weights"], priority=36)
    reply("help", "Try 'what is a neural network?', 'explain a Python list', 'help me plan a project', 'tell me a joke', or '12345 + 67890'. You can also introduce yourself with 'my name is Alex'.", ["fallback_help"], priority=1)
    reply("set_name", "Nice to meet you, {name}. I'll remember that name for this chat session.", ["name_candidate"], priority=95, write_name=True)
    reply("recall_name", "You told me your name is {name}.", ["ask_name", "has_name"], ["name_candidate", "forget"], priority=60)
    reply("no_name", "You haven't told me your name in this session. You can say 'my name is Alex'.", ["ask_name"], ["has_name", "name_candidate", "forget"], priority=55)
    reply("forget_name", "I've cleared the name from this chat session.", ["forget"], priority=99, clear_name=True)
    reply("addition", "That adds up to {sum}.", ["addition"], priority=100)
    reply("unsupported_math", "My arithmetic circuit currently supports addition of two non-negative integers, with at most 12 digits each. Try an expression such as '123 + 456'. I should not guess an answer for other arithmetic.", ["unsupported_math"], priority=90)

    # Each topic has three explicit neural states: explanation, simpler, and more.
    topics = [
        ("neural_network", ["neural network", "neural networks", "neuron", "neurons"],
         "A neural network transforms inputs through layers of weighted connections. Each layer combines numbers and usually applies a nonlinear function. Its weights determine what it does. Training normally adjusts those weights from examples; my weights were written directly instead.",
         "Think of a neural network as many adjustable switches connected together. Input enters one end, each layer changes it, and an answer comes out. The switch settings are the weights.",
         "A dense layer computes y = activation(Wx + b). W contains connection weights and b contains biases. Nonlinear activations let several layers express more than a single linear transformation. A useful network still needs suitable weights: a large random network does not automatically know language."),
        ("machine_learning", ["machine learning", "ml"],
         "Machine learning uses examples or experience to adjust a model's behavior. A training objective measures an error or reward, and an algorithm changes parameters to improve it. Testing on separate examples helps reveal whether the model generalizes.",
         "Instead of writing every decision by hand, you show a model examples and let a procedure adjust it. Then you test it on examples it has not seen.",
         "Common settings include supervised learning from labeled examples, self-supervised learning from structure in data, and reinforcement learning from rewards. Training performance alone is insufficient: data leakage and overfitting can make a model appear better than it is."),
        ("transformer", ["transformer", "transformers", "self attention", "self-attention"],
         "A transformer processes token representations using attention and feed-forward layers. Attention lets each position combine information from other allowed positions. Language models usually learn the weights through extensive training.",
         "Attention lets a model look at relevant words elsewhere in a passage while processing the current word. Several layers build more useful representations.",
         "Attention forms queries, keys, and values from token vectors. Similarities between queries and keys determine how values are mixed. A causal mask prevents a text generator from looking at future tokens. Positional information distinguishes different orders of the same words."),
        ("gradient", ["gradient", "gradients", "backpropagation", "backprop"],
         "A gradient describes how a quantity changes when its inputs change. In neural-network training, backpropagation computes derivatives of a loss with respect to weights. An optimizer uses those derivatives to update the weights.",
         "A gradient points in the direction of increasing error. Training can take a small step the other way to try to reduce the error.",
         "Backpropagation applies the chain rule through a computation graph. Gradient descent subtracts a learning-rate-scaled gradient. The learning rate, optimizer, data, and numerical precision all affect whether those updates are useful. My construction did not use this process."),
        ("overfitting", ["overfitting", "overfit", "generalization", "generalisation"],
         "Overfitting happens when a model captures details of its training examples that do not transfer well to new examples. A separate validation or test set helps measure that gap.",
         "It is like memorizing the answers to a practice sheet without learning how to solve a new sheet.",
         "Possible controls include collecting better data, simplifying the model, regularization, and stopping training before validation performance degrades. A test set should stay separate from choices made while improving the model."),
        ("token", ["token", "tokens", "tokenizer", "tokenization"],
         "A token is a unit of text represented by an integer. It might be a byte, character, word, or word fragment. A tokenizer converts text to those units, and a decoder converts generated units back to text.",
         "Tokens are the pieces a model uses to handle text. A piece can be smaller than a word.",
         "Token boundaries affect sequence length and how unusual words are represented. This experiment uses a fixed lexical feature encoder for requests and a neural byte decoder for replies; it does not use a pretrained tokenizer."),
        ("random_weights", ["random weights", "random initialization", "random initialisation"],
         "Random weights can provide a starting point for training, but they do not contain learned language skills. Their scale matters because activations and gradients can otherwise grow or shrink excessively.",
         "A random starting position is useful when learning will follow. It is not the same thing as already knowing how to chat.",
         "Initialization schemes often depend on the number of inputs and outputs of a layer. Directly programming a network is different: its parameter values must already implement the intended computation."),
        ("python", ["python"],
         "Python is a programming language with readable syntax and a large ecosystem. Indentation groups statements. Common building blocks include functions, lists, dictionaries, loops, and classes.",
         "Python lets you write instructions such as loops and functions with relatively little punctuation. Indentation tells it which instructions belong together.",
         "A useful small program has a clear input, a few focused functions, and an observable result. Handle expected errors explicitly and use a virtual environment when a project needs its own dependencies."),
        ("variable", ["variable", "variables"],
         "A variable gives a name to a value or storage location, depending on the language. Meaningful names make it easier to understand what the value represents and where it changes.",
         "A variable is a label you use to refer to a value, such as score = 10.",
         "Distinguish the name from the object it refers to. In languages such as Python, two names can refer to the same mutable object, so changing that object through one name can be visible through the other."),
        ("function", ["function", "functions"],
         "A function groups a computation behind a name. It can accept inputs and return a result. Focused functions are easier to reuse and test than one long sequence of unrelated operations.",
         "A function is a reusable recipe: give it ingredients, and it produces a result.",
         "Try to make a function's inputs, outputs, and side effects clear. For example, separate calculating a value from writing a file so that the calculation can be tested without touching the filesystem."),
        ("loop", ["loop", "loops", "iteration"],
         "A loop repeats a block of code. A for loop commonly visits items in a sequence; a while loop repeats while a condition stays true. Make sure the loop can terminate when intended.",
         "A loop says 'do this again' for a list of items or until a condition changes.",
         "Boundary errors often come from confusing an item count with the last valid index. Also check what happens for an empty collection and whether modifying the collection during iteration changes which items are visited."),
        ("recursion", ["recursion", "recursive"],
         "Recursion is when a function solves a problem by calling itself on a smaller version. It needs a base case that stops the calls and progress toward that case.",
         "Break a problem into a smaller copy of itself, and stop when the copy is simple enough to answer directly.",
         "Tree traversal is a common example. Recursion can make the structure clear, but deep recursion uses stack space. An explicit stack or iterative approach may be more appropriate for very deep inputs."),
        ("algorithm", ["algorithm", "algorithms"],
         "An algorithm is a precise sequence of steps for solving a problem. Its usefulness depends on correctness, resource use, and how well its assumptions match the input.",
         "An algorithm is a recipe for getting from a problem to an answer.",
         "Start by defining the input, output, and edge cases. Then compare time and memory requirements as input size grows. A theoretically faster method may still be slower on small inputs because of overhead."),
        ("big_o", ["big o", "time complexity", "complexity notation"],
         "Big-O notation describes an upper bound on how resource use grows with input size. O(n) grows linearly and O(n squared) grows quadratically. It describes growth, not an exact running time.",
         "It tells you how much harder the work becomes as the input gets bigger.",
         "For nested loops, check what each loop actually visits instead of automatically multiplying their limits. Also distinguish worst-case, average-case, and amortized claims, and measure important implementations on realistic inputs."),
        ("data_structure", ["data structure", "data structures"],
         "A data structure organizes information for particular operations. Arrays are good for indexed access, hash tables for key lookup, and trees for hierarchical or ordered relationships.",
         "It is a way of arranging data so the operations you need are convenient.",
         "Choose according to the operations that dominate: lookup, insertion, deletion, traversal, or ordering. Consider memory layout as well as abstract complexity because cache behavior can matter."),
        ("debug", ["debug", "debugging", "bug", "bugs"],
         "Start with a reproducible failure and a clear expected result. Narrow the problem, inspect the relevant state, and test one explanation at a time. Fix the cause and check the original failing case again.",
         "Make the bug happen reliably, find the smallest place where things go wrong, and check that your fix changes that result.",
         "Useful evidence includes exact inputs, error messages, logs, and a minimal reproduction. Avoid changing several unrelated things together: it makes it harder to tell which change mattered."),
        ("test", ["unit test", "unit tests", "software testing", "test code"],
         "A useful test checks externally meaningful behavior. Cover normal inputs, important boundaries, and known failure cases. Tests should make a regression easier to detect, rather than merely repeat the implementation.",
         "A test gives code an input and checks whether the result is what you expected.",
         "Keep deterministic unit tests for small components, integration tests for component boundaries, and a few end-to-end checks for critical flows. A passing test suite is evidence about the cases covered, not proof that every possible behavior is correct."),
        ("git", ["git", "version control"],
         "Git records versions of files as commits. A branch names a line of development, and a diff shows changes. Before committing, inspect the working tree and verify that the intended files and changes are included.",
         "Git keeps snapshots of a project so you can see changes and return to earlier versions.",
         "Small, focused commits make review and debugging easier. A merge combines histories; a rebase rewrites the base of commits. Be careful when rewriting history that other people already use."),
        ("json", ["json"],
         "JSON is a text format for structured data. It has objects, arrays, strings, numbers, true, false, and null. Object keys and string values use double quotes.",
         "JSON is a way to write lists and named values in plain text so programs can exchange them.",
         "Use a parser rather than splitting JSON by punctuation. Validate required fields and types at a boundary, and keep in mind that the standard format does not include comments or arbitrary executable code."),
        ("database", ["database", "databases", "sql"],
         "A database stores and retrieves structured information. Relational databases organize data into tables and use relationships between rows. SQL is a language for querying and changing that data.",
         "A database is an organized store of information that a program can search and update.",
         "Useful design considerations include keys, constraints, indexes, and transactions. Parameterized queries separate values from SQL syntax. Indexes can speed up reads but add storage and update costs."),
        ("api", ["api", "apis"],
         "An API defines how one piece of software interacts with another. It specifies operations, inputs, outputs, and expected errors. A network API often exposes those operations through HTTP requests.",
         "An API is an agreed set of requests one program can make to another.",
         "A good client handles errors, timeouts, and changes in response shape deliberately. Retrying a read may be harmless; retrying a write can duplicate an action unless the API provides an idempotency mechanism."),
        ("http", ["http", "https"],
         "HTTP is a request-response protocol commonly used on the web. A request includes a method and target; a response includes a status and often a body. HTTPS protects the connection using TLS.",
         "A client asks a server for something, and the server sends a response. HTTPS adds protection to that connection.",
         "GET usually retrieves a representation, while POST commonly submits data or requests an action. Status codes describe categories such as success, client errors, and server errors. Their exact meaning still depends on the endpoint."),
        ("computer_network", ["computer network", "networking", "packet", "packets"],
         "A computer network lets devices exchange data. Data is commonly split into packets, which carry addressing information. Different protocols handle delivery, reliability, and application-level meaning.",
         "Computers send small addressed pieces of data to each other, a little like sending numbered parcels.",
         "Latency measures delay, while bandwidth measures how much data can be carried per unit time. A connection can have high bandwidth and still feel slow if each interaction requires many round trips."),
        ("cpu_gpu", ["cpu", "gpu", "graphics processor"],
         "A CPU handles a wide range of control-heavy tasks. A GPU is designed to process many similar operations in parallel. Large matrix operations can suit GPUs, while small tasks may be dominated by transfer and launch overhead.",
         "A CPU is flexible. A GPU can do lots of similar calculations at once.",
         "Performance depends on memory movement as well as arithmetic. Batching work can help a GPU, but repeatedly transferring small arrays between CPU and GPU can erase the benefit. Measure the actual workload."),
        ("memory", ["ram", "computer memory", "memory allocation"],
         "RAM holds working data that programs can access quickly. Storage keeps files persistently. A model's file size does not include all of the working memory required to run it.",
         "RAM is a program's working desk. Storage is where files are kept when they are not on the desk.",
         "Inference may need parameter storage, activations, temporary buffers, and conversation state. Numerical precision affects memory use: a float32 parameter uses four bytes, while a float16 parameter uses two."),
        ("float", ["floating point", "float32", "float16", "numerical precision"],
         "Floating-point numbers represent a finite set of values with a significand and exponent. Many decimal fractions are not exact in binary, so arithmetic can introduce rounding error.",
         "A computer has a limited number of digits for many numbers, so small rounding differences can appear.",
         "The order of additions can affect the result. Compare approximate quantities with a suitable tolerance, and watch for overflow, underflow, and cancellation when subtracting nearly equal numbers."),
        ("binary", ["binary", "bit", "bits", "byte", "bytes"],
         "A bit has two possible values, usually written 0 and 1. A byte commonly contains eight bits. Groups of bits can encode numbers, text, images, and instructions according to an agreed representation.",
         "Bits are zeros and ones. Combining them lets a computer represent many kinds of information.",
         "An unsigned byte can represent integers from 0 through 255. The same bit pattern can have a different meaning under another encoding, so file formats need to specify how their bytes are interpreted."),
        ("gravity", ["gravity", "gravitational"],
         "Gravity is the attraction associated with mass and energy. Near Earth's surface it pulls objects downward. Newton's law is useful for many everyday calculations; general relativity describes gravity through curved spacetime.",
         "Gravity is why objects fall toward Earth and why planets stay in orbit around the Sun.",
         "An orbit is continuous falling with enough sideways motion to keep missing the central body. Objects in orbit still experience gravity; apparent weightlessness comes from sharing the same free fall."),
        ("photosynthesis", ["photosynthesis", "photosynthetic"],
         "Photosynthesis lets plants, algae, and some bacteria use light energy to make energy-rich molecules. In familiar plant photosynthesis, carbon dioxide and water contribute to making sugars, and oxygen is released.",
         "Plants use sunlight to help make food from water and carbon dioxide.",
         "In plants, light-dependent reactions capture energy, and other reactions use that energy to fix carbon into organic molecules. Chlorophyll helps absorb light, but photosynthesis involves several coordinated processes."),
        ("water_cycle", ["water cycle", "evaporation", "condensation"],
         "The water cycle moves water between the surface and atmosphere through evaporation, condensation, precipitation, and flow. Sunlight supplies much of the energy that drives it.",
         "Water evaporates, forms clouds, falls as rain or snow, and moves back through rivers, soil, and oceans.",
         "Water can also move through living things, groundwater, ice, and snow. The time it stays in each place varies greatly, so the cycle is not one simple loop moving at one speed."),
        ("atom", ["atom", "atoms", "atomic structure"],
         "An atom has a small nucleus containing protons and usually neutrons, surrounded by electrons. The number of protons identifies the element. Electron behavior is described by quantum mechanics.",
         "Atoms are tiny units of matter with a nucleus and electrons around it.",
         "Atoms of the same element can have different numbers of neutrons; these are isotopes. Ions have gained or lost electrons. Chemical bonding mainly involves interactions associated with electrons."),
        ("energy", ["energy", "conservation of energy"],
         "Energy is a quantity associated with a system's ability to undergo changes and do work. It can be transferred and change form. For an isolated system, total energy is conserved.",
         "Energy can move between things and change form, but it does not simply disappear.",
         "For example, a falling object's gravitational potential energy can become kinetic energy and later thermal energy. Calling energy 'lost' often means it has spread into a less useful form, not that it has ceased to exist."),
        ("electricity", ["electricity", "electric current", "voltage", "resistance"],
         "Electric current is the flow of electric charge. Voltage describes an electric potential difference, and resistance describes opposition to current in a simple circuit. Ohm's law relates these as V = I times R for an ohmic component.",
         "Voltage provides an electrical push, current is moving charge, and resistance limits the flow in a simple circuit.",
         "The water-flow analogy can help initially, but real circuits also involve fields, capacitance, inductance, and component behavior that the analogy misses. Ohm's law is a model for appropriate components, not every electrical device."),
        ("sound", ["sound", "sound waves", "frequency"],
         "Sound is a mechanical disturbance traveling through a medium such as air or water. Frequency is related to perceived pitch, while amplitude is related to loudness, though perception is more complicated than either alone.",
         "A vibration travels through material to your ears. Faster repeating vibrations usually sound higher in pitch.",
         "Sound needs a medium and cannot propagate through an ideal vacuum. Its speed depends on the medium and conditions. A recording samples a changing signal at discrete times."),
        ("light", ["light", "photon", "photons", "electromagnetic"],
         "Light is electromagnetic radiation. Visible light is the part our eyes can detect. Light shows wave-like behavior such as interference and particle-like behavior in interactions involving photons.",
         "Visible light is one kind of electromagnetic radiation that our eyes can sense.",
         "Different frequencies correspond to different parts of the electromagnetic spectrum. Reflection changes direction at a surface, refraction changes propagation across media, and absorption transfers energy to matter."),
        ("probability", ["probability", "probabilities", "chance"],
         "Probability measures uncertainty using values from 0 to 1. Zero represents an impossible event within a model, and one represents certainty. A probability statement depends on the assumptions and information being used.",
         "Probability describes how likely something is, from impossible to certain.",
         "Conditional probability updates the description when extra information is known. Independence is a specific assumption: one event occurring does not change the probability of the other. It should not be assumed merely because events look unrelated."),
        ("average", ["average", "mean", "median"],
         "The arithmetic mean adds values and divides by their count. The median is the middle value after sorting, or the mean of the two middle values for an even count. Outliers can move the mean substantially.",
         "The mean shares the total equally. The median finds the middle of the ordered values.",
         "Choose a summary that fits the question. A median can describe a typical observation when the distribution is skewed, while a mean is useful for totals and expectations. A spread measure adds information that either center alone omits."),
        ("fraction", ["fraction", "fractions"],
         "A fraction represents a numerator divided by a nonzero denominator. Equivalent fractions represent the same value. To add fractions directly, express them with a common denominator first.",
         "A fraction tells you how many equal parts you have and how many make a whole.",
         "For example, one half and two quarters represent the same amount. Multiplying both numerator and denominator by the same nonzero number preserves the value. Division by zero is undefined in ordinary arithmetic."),
        ("percentage", ["percentage", "percent", "percentages"],
         "A percentage expresses a ratio per hundred. Ten percent means 10 out of 100. Always identify the quantity used as the base when describing a percentage change.",
         "Percent means 'out of a hundred'. The base tells you what that hundred represents.",
         "A percentage-point change and a percent change are different. Moving from 20 percent to 30 percent is an increase of 10 percentage points and a relative increase of 50 percent."),
        ("project_plan", ["project", "project plan", "planning", "plan"],
         "Start with a small, observable outcome. List the constraints, split the work into a few milestones, and identify the riskiest assumption. Build a minimal version that tests that assumption, then use the result to decide the next step.",
         "Choose what 'done' looks like, break it into small steps, and test the uncertain part early.",
         "For each milestone, name a concrete deliverable and a way to verify it. Keep a short list of dependencies and unresolved decisions. A working slice usually reveals more than a detailed plan whose assumptions have never been tested."),
        ("learning", ["study", "studying", "learn a skill", "learning strategy"],
         "Choose a small target skill, practice recalling or using it, and check the result. Revisit it over several sessions rather than relying only on one long session. Adjust the next exercise to the mistakes you actually make.",
         "Practice doing the thing, check where you got stuck, and try again later.",
         "Combine a worked example with an attempt you complete without looking. Explain the steps in your own words and vary the next problem slightly. This helps distinguish understanding from recognizing a familiar answer."),
        ("writing", ["writing", "write clearly", "clear prose"],
         "Start with the main point and give each paragraph one job. Prefer concrete subjects and active verbs. Include the detail the reader needs to understand or act, and remove repetition that does not add meaning.",
         "Say the main thing first, use ordinary words, and keep each paragraph focused.",
         "Revision is often easier after identifying the reader and purpose. Check whether each sentence follows from the previous one and whether examples explain the point better than extra abstract wording."),
    ]

    for name, terms, normal, simple, more in topics:
        group(name, terms)
        first = len(replies)
        context = "context_" + name
        group(context, [f"@prev:{first}", f"@prev:{first + 1}", f"@prev:{first + 2}"])
        reply(name, normal, [name], priority=22)
        reply(name + "_simple", simple, [context, "simple"], priority=80)
        reply(name + "_more", more, [context, "more"], priority=75)

    for name, terms, text in [
        ("python_list", ["list", "lists"], "A Python list stores an ordered collection: values = [1, 2, 3]. Use values.append(4) to add an item and values[0] to read the first item. Lists are mutable, and indexing starts at zero."),
        ("python_dict", ["dictionary", "dictionaries", "dict"], "A Python dictionary maps keys to values: person = {'name': 'Alex'}. Read person['name'] when the key must exist, or person.get('name') when a missing key is expected. Keys must be hashable."),
        ("python_function", ["function", "functions"], "A small Python function looks like:\n\ndef greet(name):\n    return 'Hello, ' + name\n\nCalling greet('Alex') returns a string. The indented body belongs to the function."),
        ("python_loop", ["loop", "loops"], "For a Python loop over values:\n\nfor item in [1, 2, 3]:\n    print(item)\n\nUse enumerate(items) when you need both the position and value. Avoid changing a list's size while iterating over that same list."),
        ("python_error", ["error", "exception", "exceptions"], "Read the final exception type and message, then follow the traceback to your code. Catch a specific expected exception with try/except; avoid suppressing every exception because that can hide unrelated bugs."),
    ]:
        group(name, terms)
        reply(name, text, ["python", name], priority=55)

    group("joke", ["joke", "funny", "make me laugh"])
    j1 = reply("joke_one", "Why did the programmer bring a ladder? To reach a higher level of abstraction.", ["joke"], priority=25)
    group("context_joke_one", [f"@prev:{j1}"])
    j2 = reply("joke_two", "I asked a binary tree for directions. It said, 'Take a left, then keep branching.'", ["context_joke_one", "another"], priority=80)
    group("context_joke_two", [f"@prev:{j2}"])
    reply("joke_three", "The computer wanted a quiet holiday, so it chose a place with no interrupts.", ["context_joke_two", "another"], priority=80)
    group("poem", ["poem", "poetry", "verse"])
    p1 = reply("poem_one", "A window holds the evening light,\nA quiet screen begins to glow.\nOne little question takes its flight,\nAnd gives the next a place to go.", ["poem"], priority=25)
    group("context_poem", [f"@prev:{p1}"])
    reply("poem_two", "The rain writes softly on the street,\nThe trees keep time without a sound.\nA thousand scattered droplets meet,\nAnd make a mirror of the ground.", ["context_poem", "another"], priority=80)
    group("story", ["story", "short story"])
    reply("story", "A robot found a seed in a drawer marked 'spare parts'. It planted the seed beside its charging station and measured the soil every morning. When the first leaf appeared, the robot added a new entry to its inventory: something that grows without an upgrade.", ["story"], priority=25)
    group("bored", ["bored", "boredom", "something to do"])
    reply("bored", "Try a small project with a visible finish: sketch something nearby, write a six-line story, or build a tiny program that answers one question. Pick the option that feels easiest to start, and give it ten minutes.", ["bored"], priority=20)
    group("overwhelmed", ["overwhelmed", "too much to do", "stressed", "stuck"])
    reply("overwhelmed", "That sounds like a lot to carry at once. Write down what is competing for your attention, choose one small action you can finish, and let the rest stay on the list for now. What is the task that feels most pressing?", ["overwhelmed"], priority=25)
    group("current_info", ["weather", "latest news", "today's news", "current price", "stock price", "exchange rate"])
    reply("current_info", "I cannot check live information while running offline. I do not have a reliable current answer to that request.", ["current_info"], priority=70)
    group("source_request", ["source", "sources", "citation", "citations", "prove that"])
    reply("source_request", "My replies come from explicitly authored content encoded into the parameter file. I cannot retrieve or verify external sources offline. Treat my explanations as limited reference material rather than a source-backed research result.", ["source_request"], priority=60)
    return {"groups": groups, "replies": replies, "rules": rules}
