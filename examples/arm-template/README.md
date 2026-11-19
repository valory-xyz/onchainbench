# Template: your harness's CLI as an OnchainBench arm

Every contestant runs as a CLI in its own process, so a harness enters by meeting the CLI contract
(`CONTRIBUTING.md`, "The CLI contract") and getting a session class in `onchainbench/arms.py`. This folder holds
both halves:

- `my-harness` is a stand-in CLI that meets the contract. It reverses the task text instead of calling a model,
  and prints one JSON object: the reply, the session id, an error kind, session totals and the turn's events.
- `my_harness_arm.py` is the session class around it. It builds the command line and the environment, enforces
  the wall clock and the cost cap, turns the JSON into stats and a transcript, resumes for multi-turn tasks, and
  maps the CLI's error to the bench's failure kinds.

Run its test from the bench's checkout: `.venv/bin/python -m pytest tests/test_arm_template.py`. Then replace the
command line with your CLI's. Move the class into `onchainbench/arms.py`, register the kind there (the module
docstring says where), and add a pinned install to `scripts/install-arms.sh`. Then try it:

```bash
python -m onchainbench.run --arms "my-harness:<model>" --split open --tasks a-wrap-unwrap --run try
```

The run calls the judge and the simulated user through OpenRouter, so it needs `OPENROUTER_API_KEY` in the bench's
`.env` (about a cent of judge calls a run). Without one, add `--no-judges` for a first wiring run: no judge is called,
the judged metrics read as not measured, and the run can't be published.
