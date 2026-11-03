"""Chat with the local numeric model. No response text is imported from its source program."""
import argparse
import json

from chat1gb.engine import Engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["auto", "torch", "numpy"], default="auto")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--prompt")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    engine = Engine(backend=args.backend, device=args.device)
    if args.prompt is not None:
        answer, evidence = engine.reply(args.prompt, trace=args.trace)
        print(answer)
        if args.trace:
            print(json.dumps(evidence, indent=2))
        return
    print("1 GB Neural Chat Experiment | direct weights | no training | offline")
    print(f"Running on {engine.b.device} with {engine.backend_name}. /reset clears session state; /quit exits.")
    print("This is a bounded neural conversation program, not a general language model.\n")
    while True:
        try:
            text = input("You> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if text == "/quit":
            break
        if text == "/reset":
            engine.reset()
            print("Session reset.\n")
            continue
        if not text:
            continue
        try:
            answer, evidence = engine.reply(text, trace=args.trace)
            print(f"Model> {answer}\n")
            if args.trace:
                print(json.dumps(evidence, indent=2))
        except ValueError as error:
            print(f"Input error: {error}\n")


if __name__ == "__main__":
    main()
