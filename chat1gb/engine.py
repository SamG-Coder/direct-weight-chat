"""Inference reads numeric weights only, never program.py or authored reply text."""
from chat1gb import offline

import json
import re
import time
import warnings
from pathlib import Path

DEFAULT_MODEL = Path(__file__).resolve().parent / "model"


class NumpyBackend:
    def __init__(self, filename, count, device):
        import numpy as np
        if device == "cuda":
            raise ValueError("CUDA requires the torch backend")
        self.np = np
        self.device = "cpu"
        self.storage = np.memmap(filename, mode="r", dtype="<f4", shape=(count,))

    def tensor(self, value):
        return self.np.asarray(value, dtype=self.np.float32)

    def clip(self, value, low, high):
        return self.np.clip(value, low, high)

    def cat(self, values):
        return self.np.concatenate(values)

    def argmax(self, values, axis=-1):
        return values.argmax(axis=axis)

    def tolist(self, values):
        return values.tolist()

    def sync(self):
        pass


class TorchBackend:
    def __init__(self, filename, count, device):
        warnings.filterwarnings("ignore", message="Failed to initialize NumPy:.*")
        import torch
        self.torch = torch
        torch.set_grad_enabled(False)
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        mapped = torch.from_file(str(filename), shared=False, size=count, dtype=torch.float32)
        self.storage = mapped.to(device)

    def tensor(self, value):
        return self.torch.tensor(value, dtype=self.torch.float32, device=self.device)

    def clip(self, value, low, high):
        return value.clamp(low, high)

    def cat(self, values):
        return self.torch.cat(values)

    def argmax(self, values, axis=-1):
        return values.argmax(dim=axis)

    def tolist(self, values):
        return values.tolist()

    def sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()


class Engine:
    def __init__(self, model=DEFAULT_MODEL, backend="auto", device="auto"):
        self.directory = Path(model)
        self.config = json.loads((self.directory / "model.json").read_text(encoding="utf-8"))
        if self.config["format"] != "offline-neural-chat-f32-v1":
            raise ValueError("Unsupported model format")
        filename = self.directory / self.config["weights_file"]
        if filename.stat().st_size != self.config["weight_bytes"]:
            raise ValueError("Model weight file has the wrong size")
        if backend == "auto":
            import importlib.util
            backend = "torch" if importlib.util.find_spec("torch") else "numpy"
        self.backend_name = backend
        self.b = (TorchBackend if backend == "torch" else NumpyBackend)(filename, self.config["parameter_count"], device)
        self.w = {}
        offset = 0
        for name, spec in self.config["tensors"].items():
            if spec["offset_bytes"] != offset:
                raise ValueError("Tensor layout has a gap or overlap")
            start = offset // 4
            count = spec["nbytes"] // 4
            self.w[name] = self.b.storage[start:start + count].reshape(spec["shape"])
            offset += spec["nbytes"]
        if offset != filename.stat().st_size:
            raise ValueError("Unreferenced bytes after final tensor")
        self.features = {name: i for i, name in enumerate(self.config["features"])}
        self.last_reply = None
        self.name = None
        self.default_name = bytes(self.decode_bits(self.w["name_default.bits"])).decode("utf-8")
        self.b.sync()

    def reset(self):
        self.last_reply = None
        self.name = None

    def linear(self, prefix, value):
        result = self.w[prefix + ".weight"] @ value
        bias = self.w.get(prefix + ".bias")
        return result + bias if bias is not None else result

    def decode_bits(self, bits):
        logits = bits @ self.w["byte_decoder.weight"].T + self.w["byte_decoder.bias"]
        return self.b.tolist(self.b.argmax(logits))

    def parse(self, text):
        if len(text) > self.config["max_input_characters"]:
            raise ValueError(f"Input is limited to {self.config['max_input_characters']} characters")
        tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\w\s]", text.lower())
        present = set()
        for length in range(1, self.config["max_ngram"] + 1):
            for i in range(len(tokens) - length + 1):
                key = " ".join(tokens[i:i + length])
                if key in self.features:
                    present.add(key)
        if self.last_reply is not None:
            present.add(f"@prev:{self.last_reply}")
        if self.name:
            present.add("@has_name")
        candidate = None
        match = re.fullmatch(r"\s*(?:hi[,!]?\s+|hello[,!]?\s+)?(?:my name is|call me)\s+([A-Za-z][A-Za-z' -]{0,47}?)[.!]?\s*", text, re.I)
        if match:
            candidate = match.group(1).strip()
            present.add("@name_candidate")
        arithmetic = re.fullmatch(r"\s*(?:(?:what is|what's|calculate|compute|please calculate)\s+)?([0-9]{1,12})\s*(?:\+|plus)\s*([0-9]{1,12})\s*[?=.!]?\s*", text, re.I)
        if arithmetic is None:
            arithmetic = re.fullmatch(r"\s*(?:please\s+)?add\s+([0-9]{1,12})\s+(?:and|to)\s+([0-9]{1,12})\s*[?.!]?\s*", text, re.I)
        operands = arithmetic.groups() if arithmetic else None
        if operands:
            present.add("@addition")
        elif re.search(r"\d\s*(?:[+*/=\-]|plus|minus|times)\s*-?\d", text, re.I):
            present.add("@unsupported_math")
        x = [0.0] * len(self.features)
        for term in present:
            if term in self.features:
                x[self.features[term]] = 1.0
        return self.b.tensor(x), candidate, operands, sorted(present)

    def add_digit(self, a, b, carry):
        x = [0.0] * 21
        x[a], x[10 + b], x[20] = 1, 1, carry
        total = self.linear("adder.sum", self.b.tensor(x))
        hinge = self.b.clip(self.linear("adder.hinge", total), 0, float("inf"))
        digit = self.linear("adder.digit", self.b.cat([total, hinge]))
        carry_out = self.linear("adder.carry", hinge)
        logits = self.linear("adder.readout", digit)
        return int(self.b.argmax(logits).item()), float(carry_out.item())

    def neural_add(self, left, right):
        width = max(len(left), len(right))
        carry, result = 0.0, []
        # The host schedules columns and tokenizes digits. Numeric addition and
        # carry computation occur exclusively in the stored neural layers.
        for a, b in zip(reversed(left.zfill(width)), reversed(right.zfill(width))):
            digit, carry = self.add_digit(ord(a) - ord("0"), ord(b) - ord("0"), carry)
            result.append(chr(ord("0") + digit))
        if carry > 0.5:
            digit, carry = self.add_digit(0, 0, carry)
            result.append(chr(ord("0") + digit))
        return "".join(reversed(result)).lstrip("0") or "0"

    def reply(self, text, trace=False):
        started = time.perf_counter()
        x, candidate, operands, present = self.parse(text)
        hidden = self.linear("encoder", x)
        concepts = self.b.clip(self.linear("concept", hidden), 0, 1)
        logits = self.linear("rule", concepts)
        rid = int(self.b.argmax(logits).item())
        if self.w["name_clear.weight"][rid].item() > 0.5:
            self.name = None
        if self.w["name_write.weight"][rid].item() > 0.5 and candidate:
            self.name = candidate
        tokens = self.decode_bits(self.w["reply.bits"][rid])
        answer, pending = [], bytearray()

        def flush():
            if pending:
                answer.append(pending.decode("utf-8", errors="replace"))
                pending.clear()

        for token in tokens:
            if token == self.config["tokens"]["eos"]:
                break
            if token < 256:
                pending.append(token)
            elif token == self.config["tokens"]["copy_name"]:
                flush()
                answer.append(self.name or self.default_name)
            elif token == self.config["tokens"]["neural_sum"]:
                if operands is None:
                    raise RuntimeError("The neural program requested addition without valid operands")
                flush()
                answer.append(self.neural_add(*operands))
            else:
                raise RuntimeError("Unsupported decoder token")
        flush()
        self.last_reply = rid
        self.b.sync()
        response = "".join(answer)
        evidence = {"reply_id": rid, "reply_label": self.config["reply_labels"][rid],
                    "seconds": time.perf_counter() - started, "backend": self.backend_name, "device": self.b.device}
        if trace:
            evidence["input_features"] = present
            evidence["active_concepts"] = [name for name, value in zip(self.config["concepts"], self.b.tolist(concepts)) if value > 0.5]
        return response, evidence
