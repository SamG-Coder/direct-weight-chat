# Historical selector v0 (rejected)

This prototype stores canned answers and does not satisfy the current task.
The current compact model is exactly **1,528,592 packed weight bytes**.
See the [root README](../README.md). The rejected prototype below is historical.

# Archived neural conversation prototype

An offline neural conversation program whose weights were written directly,
without training or pretrained checkpoints. The archived parameter layout and file hashes remain recorded in its original
manifest. This prototype is not the current compact checkpoint.

**Scope:** this is a bounded, explicitly programmed neural chatbot. It has 160
authored response states, topic follow-ups, a session name slot, and a neural
integer-adder circuit. It does not have the open-ended language abilities of a
trained language model. Its large dense representation is deliberately
overcomplete: most parameters are redundant projections, not independently
learned knowledge. A larger representation does not improve its understanding by itself.

## Chat now

Double-click `Chat.cmd` in this directory, or run this in PowerShell:

```powershell
python -m chat1gb.chat
```

Try:

```text
Hello
My name is Sam.
What is my name?
What is a neural network?
Explain that in simpler terms.
Tell me more.
12345 + 67890
What are your limitations?
```

`/reset` clears the previous topic and remembered name. `/quit` exits. The name
slot lives only in the running process. There is no general conversation history
or persistent personal memory.

For a single response with routing details:

```powershell
python -m chat1gb.chat --prompt '12345 + 67890' --trace
```

The launcher uses `python` from your active environment. Install dependencies
from an offline package cache before running. No package is downloaded by inference.
To select the verified NumPy CPU path explicitly:

```powershell
# Run from the repository root
python -m chat1gb.chat --backend numpy --device cpu
```

## What the network actually does

The full dense layers run for each request; inference does not substitute the
small source rule table for those matrices.

```text
Text and session state
        |
362 fixed lexical/context features
        |
516,917 dense linear hidden units
        |
119 capped-ReLU concept gates
        |
160 neural reply states
        |
572-position, 9-bit neural byte memory
        |
258-way neural byte decoder
        |
Reply text, with optional name copying or neural addition
```

The first two layers use an explicitly constructed Hadamard frame and its inverse.
Their composition implements the declared lexical OR gates. The rule layer
implements required/forbidden concept combinations using weights and biases.
The decoder reads a selected numeric byte memory and reconstructs its text using
neural logits. There is no Python dictionary of answers in inference, and the
inference module never imports the authored response source.

The ordinary host code tokenizes requests, extracts supported name/operand forms,
schedules layers, copies the name slot when instructed by a neural output token,
and displays text. This is not a transformer or an autoregressively trained LLM.
Most ordinary replies are selected authored passages, not newly composed prose.

The addition component accepts two non-negative integers of up to 12 digits each.
For each decimal column, stored neural layers compute the sum, a carry using
ReLU hinges, and the output digit. The host schedules columns; it does not add the
operands. This supports unseen numeric combinations without a stored answer table.

## Validation and limitations

Both the RTX 5080 / PyTorch CUDA path and the NumPy CPU path were tested. They
produced identical text and routing on the recorded direct and follow-up checks.

| Check | Result |
|---|---:|
| Artifact | Historical dense prototype; see its original manifest |
| Float32 parameters | 250,000,000 |
| Named tensor layout | No gaps, overlaps, or trailing padding |
| Direct chat questions: expected reply selected | 67 / 68 |
| Topic follow-ups: expected reply selected | 84 / 86 |
| Name set, recall, clear, and CLI reset | Passed |
| Exhaustive single-digit addition/carry cases | 200 / 200 |
| Multi-digit addition cases, including unseen generated operands | 260 / 260 |
| Neural routing and decoder ablations | Passed |
| Runtime imports no authored answer source | Passed |
| Python socket guard blocks a connection before network I/O | Passed |
| Original weight checksum after evaluation | Unchanged |

These are local functional checks, not an independent language-model benchmark.
The questions are chosen to exercise the intended program. Randomness was used
only to choose evaluation operands, not to create or optimize model weights.

The retained direct failure is `What does percent mean?`: the word `mean`
activates the statistics topic and wins over percentages. The next two follow-ups
therefore stay on the wrong topic. Other diagnostics demonstrate that:

- `Do not explain gravity.` still selects the gravity explanation: negation is
  not understood generally.
- A request for a new scientific theory produces the explicit unknown response.
- A request to write an arbitrary new Python program produces a general Python
  explanation rather than generating the requested program.
- Asking about a question three turns ago produces the unknown response because
  there is no general conversation-history model.

The model was kept unchanged after those results. This preserves an honest
measurement of the directly written weights rather than hiding failed examples.

Recorded median complete-reply latency was about **1.41 ms on the RTX 5080** and
**19.64 ms on the NumPy CPU path** in this run. These include both large dense
layers and byte decoding, but are not comparable to autoregressive LLM token rates:
the model selects a finite byte memory. See the evidence JSON for actual timing.

## Model and evidence files

- `model/weights.f32`: raw little-endian float32 neural parameters; no header or padding.
- `model/model.json`: tensor shapes, offsets, tokenizer features, and architecture.
- `model/manifest.json`: checksums, direct-construction provenance, and per-tensor statistics.
- `program.py`: the explicitly authored source program, used only during construction.
- `build.py`: deterministic compiler; no training loop, optimizer, or dataset fitting.
- `engine.py`: dense numeric inference, neural byte decoding, and neural addition.
- `chat.py`, `Chat.ps1`, `Chat.cmd`: console chat interfaces.
- `verify.py`: integrity, behavior, arithmetic, ablation, and offline checks.
- `evidence/verification-torch-cuda.json`: complete GPU results and retained failures.
- `evidence/verification-numpy-cpu.json`: complete CPU results and retained failures.
- `evidence/conversation.txt`: an actual recorded multi-turn demonstration.

The weight SHA-256 is:

```text
2ec4cc6befec6434207b4d7f21c2afa5f1b2bd95d03732ed595e27192ef393b7
```

## Recheck offline

```powershell
# Run from the repository root
python -m chat1gb.verify --backend numpy --device cpu
python -m chat1gb.verify --backend torch --device cuda
```

The original build command was `python -m chat1gb.build`. It refuses to overwrite
an existing `model` directory. The model can be rebuilt from these local sources
with NumPy; no pretrained file, corpus download, or network connection is used.
The offline audit hook rejects outgoing Python socket calls during construction,
chat, and verification. It is an application guard, not a system-wide firewall.

The existing deletions in the parent benchmark checkout were left untouched.
All files for this task are contained in `chat1gb`.
