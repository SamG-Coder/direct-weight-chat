import argparse
import json

from .engine import Model, DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser(description="Offline next-token generation with directly written transformer weights")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--prompt")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    model = Model(directory=args.model_dir, device=args.device)
    if args.prompt is not None:
        result = model.generate(args.prompt, args.max_new_tokens, args.trace)
        print(json.dumps(result, indent=2) if args.trace else result["text"])
        return
    print(f"Direct-weight causal transformer: 250M float32 parameters | {model.device} | offline")
    print("Experimental, untrained language ability. /reset starts a fresh context; /quit exits.\n")
    while True:
        try:
            prompt = input("You> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if prompt == "/quit":
            break
        if prompt == "/reset":
            model.reset()
            print("Context reset.\n")
            continue
        if not prompt:
            continue
        try:
            result = model.generate(prompt, args.max_new_tokens, args.trace)
            print("Model> " + result["text"] + "\n")
            if args.trace:
                print(json.dumps(result, indent=2))
        except ValueError as error:
            print(str(error))


if __name__ == "__main__":
    main()
