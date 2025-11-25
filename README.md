

##Need to ADD Rag


# Ollama local test script

This repository includes `test.py`, a small script to locally test an Ollama model (default: `llama:3.21b`) using the `ollama` CLI.

## Requirements

- Ollama installed and `ollama` available on your PATH. See https://ollama.com for installation.
- The model `llama:3.21b` pulled locally (optional) or available for automatic download by Ollama.

## Usage

Run the script from the project root:

```bash
python test.py --model llama:3.21b --prompt "Hello from my test script"
```

Flags:
- `--model`: model identifier (default: `llama:3.21b`)
- `--prompt`: the prompt to send to the model
- `--skip-list-check`: skip running `ollama list` to check model availability

## Troubleshooting

- If you see "'ollama' CLI not found in PATH", make sure Ollama is installed and the `ollama` binary is accessible from your shell.
- If the model is not listed in `ollama list`, pull it manually:

```bash
ollama pull llama:3.21b
```

- If `ollama run` fails, the script prints diagnostics including stdout/stderr for two common invocation styles. Use that output to debug.

## Example

```bash
python test.py --prompt "Write a 2-line haiku about autumn"
```

If you want, set up a small virtualenv and run `python -m venv .venv && . .venv/bin/activate` before running.

---

If you'd like, I can also add an automated test that runs a very short `ollama list` check (skips if `ollama` missing) or a CI job stub—tell me which option you prefer.
