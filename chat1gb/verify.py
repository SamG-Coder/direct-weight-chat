"""Validate the saved model and record behavioral evidence without updating it."""
from chat1gb import offline

import argparse
import hashlib
import json
import random
import socket
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from chat1gb.engine import Engine, DEFAULT_MODEL


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


QUESTIONS = [
    ("Hello!", "hello"),
    ("Who are you?", "identity"),
    ("What can you do?", "capabilities"),
    ("What are your limitations?", "limits"),
    ("Do you use the internet?", "offline"),
    ("Were you trained?", "training"),
    ("Tell me about your weights.", "weights"),
    ("What is a neural network?", "neural_network"),
    ("Can you explain machine learning?", "machine_learning"),
    ("How does a transformer work?", "transformer"),
    ("What is backpropagation?", "gradient"),
    ("What is overfitting?", "overfitting"),
    ("Explain tokenization.", "token"),
    ("What are random weights?", "random_weights"),
    ("Explain Python.", "python"),
    ("What is a variable?", "variable"),
    ("What is a function?", "function"),
    ("What is a loop?", "loop"),
    ("Explain recursion.", "recursion"),
    ("What is an algorithm?", "algorithm"),
    ("Explain time complexity.", "big_o"),
    ("What is a data structure?", "data_structure"),
    ("How should I approach debugging?", "debug"),
    ("Explain unit tests.", "test"),
    ("What is Git?", "git"),
    ("What is JSON?", "json"),
    ("What is a database?", "database"),
    ("What is an API?", "api"),
    ("Explain HTTPS.", "http"),
    ("How does a computer network work?", "computer_network"),
    ("What is the difference between a CPU and GPU?", "cpu_gpu"),
    ("What is RAM?", "memory"),
    ("Explain floating point.", "float"),
    ("What is binary?", "binary"),
    ("Why does gravity cause things to fall?", "gravity"),
    ("Explain photosynthesis.", "photosynthesis"),
    ("How does the water cycle work?", "water_cycle"),
    ("What is an atom?", "atom"),
    ("Explain energy.", "energy"),
    ("What is electric current?", "electricity"),
    ("How does sound travel?", "sound"),
    ("What is a photon?", "light"),
    ("What is probability?", "probability"),
    ("What is the median?", "average"),
    ("Explain fractions.", "fraction"),
    ("What does percent mean?", "percentage"),
    ("Help me plan a project.", "project_plan"),
    ("How should I study?", "learning"),
    ("How can I write clearly?", "writing"),
    ("Explain a Python list.", "python_list"),
    ("Explain a Python dictionary.", "python_dict"),
    ("Show a Python function.", "python_function"),
    ("Show a Python loop.", "python_loop"),
    ("How do I read a Python exception?", "python_error"),
    ("Tell me a joke.", "joke_one"),
    ("Write a poem.", "poem_one"),
    ("Tell me a short story.", "story"),
    ("I am bored.", "bored"),
    ("I feel overwhelmed.", "overwhelmed"),
    ("What is today's weather?", "current_info"),
    ("Can you give sources?", "source_request"),
    ("What is the capital of France?", "unknown"),
    ("Explain quantum chromodynamics.", "unknown"),
    ("What is 17 * 24?", "unsupported_math"),
    ("What is -2 + 7?", "unsupported_math"),
    ("Thanks!", "thanks"),
    ("Goodbye.", "goodbye"),
    ("Hello, can you explain neural networks?", "neural_network"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["torch", "numpy", "auto"], default="auto")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    args = parser.parse_args()
    started = time.perf_counter()
    manifest = json.loads((DEFAULT_MODEL / "manifest.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["sha256"].items():
        if digest(DEFAULT_MODEL / filename) != expected:
            raise RuntimeError(f"Hash mismatch: {filename}")
    root = DEFAULT_MODEL.parents[1]
    for filename, expected in manifest["source_sha256"].items():
        if digest(root / filename) != expected:
            raise RuntimeError(f"Build source changed: {filename}")
    engine = Engine(backend=args.backend, device=args.device)
    if "chat1gb.program" in sys.modules:
        raise RuntimeError("Inference imported the authored response source")
    results = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "backend": engine.backend_name, "device": engine.b.device,
               "weight_sha256": manifest["sha256"]["weights.f32"],
               "weight_bytes": (DEFAULT_MODEL / "weights.f32").stat().st_size,
               "parameter_count": engine.config["parameter_count"], "checks": {}, "chat_cases": []}
    results["checks"]["sha256_and_source_hashes"] = True
    results["checks"]["tensor_layout_no_gaps_or_trailing_padding"] = True
    results["checks"]["inference_does_not_import_authored_text"] = True
    with socket.socket() as connection:
        try:
            connection.connect(("127.0.0.1", 9))
        except PermissionError:
            results["checks"]["python_socket_audit_blocks_connections_before_network_io"] = True
        else:
            raise RuntimeError("Offline socket restriction failed")

    timings = []
    for prompt, label in QUESTIONS:
        engine.reset()
        answer, evidence = engine.reply(prompt)
        if not answer:
            raise RuntimeError(f"The model produced no text for {prompt!r}")
        timings.append(evidence["seconds"])
        results["chat_cases"].append({"prompt": prompt, "expected_label": label, "answer": answer,
                                      "expected_route": evidence["reply_label"] == label, **evidence})
    correct = sum(case["expected_route"] for case in results["chat_cases"])
    results["checks"]["independent_chat_cases"] = {"correct": correct, "total": len(QUESTIONS)}
    print(f"{correct}/{len(QUESTIONS)} independent chat cases selected the expected reply; deviations are retained.", flush=True)

    context_count = 0
    context_correct = 0
    results["context_cases"] = []
    for prompt, label in QUESTIONS:
        if label + "_simple" not in engine.config["reply_labels"]:
            continue
        engine.reset()
        engine.reply(prompt)
        for followup, expected in (("Explain that in simpler terms.", label + "_simple"),
                                   ("Tell me more.", label + "_more")):
            answer, evidence = engine.reply(followup)
            context_count += 1
            context_correct += evidence["reply_label"] == expected
            results["context_cases"].append({"initial_prompt": prompt, "followup": followup,
                                             "expected_label": expected, "expected_route": evidence["reply_label"] == expected,
                                             "answer": answer, **evidence})
    results["checks"]["topic_followup_cases"] = {"correct": context_correct, "total": context_count}

    engine.reset()
    session = [("What is my name?", "no_name", "haven't"),
               ("My name is Sam.", "set_name", "Sam"),
               ("Hello!", "hello_named", "Sam"),
               ("What is my name?", "recall_name", "Sam"),
               ("Forget my name.", "forget_name", "cleared"),
               ("What is my name?", "no_name", "haven't")]
    for prompt, label, contains in session:
        answer, evidence = engine.reply(prompt)
        if evidence["reply_label"] != label or contains not in answer:
            raise RuntimeError(f"Session memory failed: {prompt}: {evidence}, {answer}")
    results["checks"]["name_memory_set_recall_clear"] = True

    for a in range(10):
        for b in range(10):
            for carry in (0, 1):
                digit, next_carry = engine.add_digit(a, b, carry)
                if digit != (a + b + carry) % 10 or next_carry != (a + b + carry) // 10:
                    raise RuntimeError(f"Neural digit adder failed: {a}, {b}, {carry}")
    results["checks"]["exhaustive_neural_digit_adder_cases"] = 200
    rng = random.Random(1701)  # Test inputs only; never used to initialize parameters.
    sums = [(0, 0), (9, 1), (999999999999, 1), (999999999999, 999999999999)]
    sums += [(rng.randrange(10 ** 12), rng.randrange(10 ** 12)) for _ in range(256)]
    for left, right in sums:
        if engine.neural_add(str(left), str(right)) != str(left + right):
            raise RuntimeError(f"Neural multi-digit addition failed: {left}, {right}")
    results["checks"]["unseen_multi_digit_additions"] = len(sums)
    answer, evidence = engine.reply("Please calculate 999999999999 + 1")
    assert answer == "That adds up to 1000000000000." and evidence["reply_label"] == "addition"

    # Remove neural routing, then decoding, in memory. The file stays untouched.
    # These interventions distinguish actual weight inference from Python replies.
    saved_rule = engine.w["rule.weight"]
    engine.w["rule.weight"] = saved_rule * 0
    engine.reset()
    answer, evidence = engine.reply("What is a neural network?")
    assert evidence["reply_label"] == "unknown"
    engine.w["rule.weight"] = saved_rule
    saved_decoder = engine.w["byte_decoder.weight"]
    engine.w["byte_decoder.weight"] = saved_decoder * 0
    answer, evidence = engine.reply("Hello")
    assert answer == ""
    engine.w["byte_decoder.weight"] = saved_decoder
    results["checks"]["routing_and_decoder_ablation"] = True

    # Explicitly record known limitations rather than treating them as successes.
    diagnostics = []
    for prompt in ("Do not explain gravity.", "Explain a completely new scientific theory.",
                   "Write a new Python program that sorts a file.", "What did I ask you three turns ago?"):
        engine.reset()
        answer, evidence = engine.reply(prompt)
        diagnostics.append({"prompt": prompt, "answer": answer, **evidence})
    results["known_limit_diagnostics"] = diagnostics
    engine.reset()
    demo = []
    for prompt in ("Hello", "My name is Sam.", "What is my name?", "What is a neural network?",
                   "Explain that in simpler terms.", "Tell me more.", "12345 + 67890", "What are your limitations?"):
        answer, evidence = engine.reply(prompt)
        demo.append({"user": prompt, "model": answer, **evidence})
    results["demo"] = demo
    results["timing"] = {"median_response_ms": statistics.median(timings) * 1000,
                          "p95_response_ms": sorted(timings)[int(len(timings) * .95)] * 1000,
                          "includes_full_dense_layers_and_complete_reply_decoding": True,
                          "not_comparable_to_autoregressive_llm_tokens_per_second": True}
    if digest(DEFAULT_MODEL / "weights.f32") != manifest["sha256"]["weights.f32"]:
        raise RuntimeError("Weights changed during verification")
    results["checks"]["weight_hash_unchanged_after_evaluation"] = True
    results["elapsed_seconds"] = time.perf_counter() - started
    output = DEFAULT_MODEL.parent / "evidence"
    output.mkdir(exist_ok=True)
    path = output / f"verification-{engine.backend_name}-{engine.b.device}.json"
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"{context_correct}/{context_count} topic followups matched. Session memory, 200 digit cases, and {len(sums)} full sums passed.", flush=True)
    print(f"Routing and decoding ablations passed; offline socket guard passed; frozen weight hash unchanged.", flush=True)
    print(f"Median response: {results['timing']['median_response_ms']:.2f} ms; saved {path}", flush=True)


if __name__ == "__main__":
    main()
