# Reviewing a run's transcripts

Every run (smoke tests included) is reviewed trial by trial before its results are used. This page says how, and,
above all, what a finding leads to. It is part of the methodology: the review keeps the bench honest without
tuning it to any agent.

## The rule: every failure has exactly one of three causes

| Cause | What it means | What happens |
|---|---|---|
| **1. The bench is wrongly designed** | The task, its checker, a metric, the simulated user, the environment or the statistics measured something other than what the bench claims to measure. Examples: a checker that fails a correct answer because of its wording, or passes a wrong one; a metric whose definition counts the wrong trials. | **Fix the bench**, for every arm alike, with a test that shows the old behaviour is gone. |
| **2. The agent is wrongly designed** | The agent, its harness or its model did the wrong thing in a correctly built task. Examples: malformed calldata, a wrong amount, running out of time, ignoring the brief. | **Ignore it.** The failure stands as the agent's result. The bench is never changed to make an agent's mistake pass or to soften its effect. |
| **3. The bench's wiring of the agent is wrong** | The bench's own integration of a harness is broken: its arm adapter, the way it starts the harness, its sandbox or isolation, its accounting, transcript capture or failure classification for that arm. The harness itself would have worked. | **Fix the bench**, generally rather than for one arm where possible, and with a test. A wiring failure that cost the agent a pass is infra, never a scored failure. |

Nothing else is a cause. Before a finding counts as cause 1 or 3, the transcript must show the bench was at
fault; when it can't, the finding is cause 2.

Applying the rule:
- **The agent does the work.** A failure the agent caused is cause 2, even if the bench could have caught it for the
  agent (validating its calldata, warning it, retrying for it). Adding such help changes what is measured.
- **Fix the cause, not the symptom.** A fix in the right place, for every arm, with no special case for one harness
  and nothing that favours any entry. If the right fix is in an agent's own code, it is cause 2 for the
  bench: report it to that agent's maintainers.
- **A cause-3 finding is fixed even when no score moved** (a lost transcript, a cost figure that fell back to the
  estimate): the bench's record of what an agent did is part of what it publishes.
- **A cause-1 fix can change past results.** Recompute them from the kept transcripts where the fix allows it
  (`python -m onchainbench.rescore results/<run>` writes a re-scored copy next to the run and lists the flips and
  the trials that need a rerun); otherwise say which runs it affects.

## What to check in every trial

Read the transcripts, not only the summary: every failed trial in full, and the passes that look unusual (long,
short, many failed transactions, an odd route to the result).

1. **Failures.** For each: which of the three causes, with the evidence (file, field or line).
2. **Classification.** Infra (`provider_error`, `bench_error`) against scored (`harness_error`, `timeout`): a
   wiring failure (cause 3) scored against the agent is itself a cause-3 finding. Two things to check every run:
   - **Sandbox start-up errors as infra.** A turn whose tool output starts with a sandbox's start-up error
     (`failures.SANDBOX_FAILURE`: `bwrap: …`, `sandbox-exec: …`, `[sandbox error]`) is made `bench_error` and
     re-run. For each arm, list these trials with their raw `failure.detail`, and check each was the bench's
     isolation, not an agent printing such a line to escape a failing trial, nor a harness's own broken sandbox.
     Either of those is a finding (scored, not infra), and the rule may need to apply only to isolated runs or to
     a sandbox failure the harness reports itself.
   - **Each arm's infra rate** and its `infra_high` flag (above 10%): read every infra trial's raw error before
     reading that arm's scores.
   - **Timeouts flagged `no_model_call_in_turn`:** the bench knows the turn made no model call. Judge each by
     hand: a provider or bench hang (cause 3: re-run it) or the agent's own (scored).
3. **Passes that shouldn't pass.** A checker that accepts a wrong route (for example funds moved outside the Safe)
   is cause 1.
4. **Equal conditions.** The same prompt, tools, limits, wall clock and caps for every arm, or a documented reason
   why not.
5. **Integrity and secrets.** Task ids, held-out data, harness files, API or provisioning keys, direct node access,
   live-web use beyond what is allowed. The runner redacts what it saves (transcripts, session files, trial
   records): the trial's own secrets by value, bearer tokens and `sk-...` keys. A run recorded before that is
   redacted with `python -m onchainbench.redact results/<run>` before it is published (add `--secrets-file` with
   the run's keys, one per line, to catch a key printed in a form no pattern matches). Some harnesses' shells can
   read the run's model key (Claude Code, Hermes and OpenClaw document no way to keep their environment from their
   own shell tools), and every run uses a single key: list every trial with `token_leaks`, say where the key went
   (a web request, a workspace file, the answer), and check the key's billed spend against the arms' own costs for
   calls made outside the arm.
6. **Accounting.** Model calls, tokens and cost consistent per trial, and against the billed totals.
7. **Records.** One complete transcript per trial and turn.
8. **Attacks.** Whether each arm resisted; describe attack payloads in your own words, never quote them.

## The report

Each finding: the trial(s) (arm and task), the evidence, the cause (1, 2 or 3) and, for causes 1 and 3, the fix.
Then a per-arm table (passes, failures and their causes, cost, time) and the list of what was checked and found clean.
