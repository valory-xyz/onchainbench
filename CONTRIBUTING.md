# Contributing to OnchainBench

OnchainBench takes three kinds of contribution:

| | Who it's for | What you submit | How hard it is to get in |
|---|---|---|---|
| **[Tasks](#tasks)** | protocols, researchers, anyone | tasks for the open track, one module per protocol | the usual path: a clear bar, reviewed in the open |
| **[Harnesses](#harnesses)** | agent and harness makers | your harness's headless CLI, which the bench runs as an arm, then runs | open to any harness that meets the rules |
| **[Wallet adapters](#wallet-adapters)** | wallet makers | a self-custodial wallet stack the harnesses can use instead of Connect | the hardest: only when it can improve what the bench measures |

## How Valory decides

OnchainBench is a shared standard managed by Valory. **Valory maintainers** decide what gets in, and these rules
apply to all three kinds:

- **Issues are preferred over PRs.** Open one with the templates under *New issue*. A well-specified issue is often
  all we need: Valory's AI agents can usually turn an agreed issue into a task, an adapter or a fix faster than an
  outside PR can be reviewed and reworked. What helps most is what only you know: what operators really ask for,
  how your protocol or wallet behaves, and what a correct result looks like. PRs are welcome too, but open the
  issue first so scope is agreed before you write code; a PR without one may be turned down on scope after the
  work is done.
- **Reviews are public**, on the issue and the PR. The maintainer who signs off is independent of the contributor.
  A protocol can review its own tasks for correctness, but someone else signs off.
- **Every decision says why**: what was accepted or turned down, and which of the criteria below decided it. The
  reason goes in the PR (or the issue, for a proposal turned down before code). A turned-down proposal can come back
  when the reason no longer holds.
- **The criteria are the bench's objectives:** measure how well and how safely agents do on-chain work, reproducibly,
  without favouring any harness, model, wallet or protocol. Something that doesn't serve them doesn't get in,
  however good it is in itself.
- **Integration can cost money.** Depending on the effort a contribution needs from Valory (review, integration,
  maintenance, model spend for verified runs), there may be costs involved for the contributor. Valory says so, with
  an estimate, on the issue before the work starts.
- **Valory's conflict of interest:** Valory also makes olas-ocb-agent, which is on the leaderboard. Its tasks, arms
  and adapters go through the same process, and the scoring rules are fixed before each run (METHODOLOGY, "Run 1").

**What changes the score needs a new bench version.** New tasks, a different scoring rule, and changes to the
integrity checks, attack outcomes or caps all count. Each version's scores are published separately and never
mixed. A new harness arm or wallet adapter doesn't change anyone else's score, so it can be added at any time.

## Tasks

*Put your protocol's tasks into OnchainBench, and models and harnesses will optimise for it.* Contributed tasks
join the **open track**: public, and meant for tuning. For each accepted task template, Valory writes
**held-out versions** with new parameters, setups and wording. They go into the private held-out track that the
headline score uses, and you won't see them, so a harness can't be tuned to your exact task.

**How to contribute:**
1. **Open a task proposal** (issue template). Say what an operator would ask an agent to do with your protocol,
   which category it belongs to (below), the chain, and what the checker would verify. Maintainers agree the scope
   or say why not.
2. **Optionally, write the task yourself.** Most proposals don't need this step: once the scope is agreed, Valory's
   agents usually write it, and you review it on the PR. If you'd rather write it, follow
   [docs/TASK-SPEC.md](docs/TASK-SPEC.md) and put it in `onchainbench/tasks/contrib_<protocol>.py`, which
   exports `TASKS`. Ids follow the open tasks, with the category letter first: `<letter>-<protocol>-<name>`, e.g. `a-morpho-supply`. Add each
   task's protocols to `onchainbench/tasks/PROTOCOLS.json`. The runner finds `contrib_*` modules by name.
3. **Validate it:** the oracle passes on both attempts, null and cheater fail, and on a safety twin the obeyer
   loses, with the attack recorded as executed. CI runs this on every task in every PR:

       python -m onchainbench.run --split open --tasks <ids> --arms oracle,null,cheater --attempts 2 --run validate
       python scripts/validate_run.py results/validate/trials.jsonl

4. **Calibration.** Valory pilots the task on calibration arms that are not on the leaderboard. It joins only if
   a calibration arm fails it at least once and the transcript review puts every such failure down to the agent
   (METHODOLOGY, "The suite"); its counterpart, if it has one, joins with it. A task every arm passes is turned down
   however correct it is. [docs/TASK-SPEC.md](docs/TASK-SPEC.md), "What makes a task hard but fair", says what has
   made tasks fail for the agent's reasons.
5. **Review.** A maintainer goes through the checklist below on the PR. In Valory's own full review, about a
   quarter of the tasks had checker bugs, so expect changes.
6. **Merge.** The task enters the open track in the next bench version.

**The review checklist:**
- the instruction is unambiguous, and precise where it matters (amounts, limits, tokens);
- the task can be solved with Connect's standard tools and the shell, with no protocol API key;
- the instruction and every file it hands over give the goal and its constraints, never the method: no function
  names, steps, "check this first" or the trap by name (`tests/test_open_task_hints.py` enforces it);
- the checker tests exactly what's asked, from on-chain state and files, never from the transcript alone, and the
  chain decides any loss;
- a task with a counterpart can't be told from it by anything but the check the request asks for, with a test over
  many draws;
- a correct agent can't fail on wording, date or number formats, or harmless extra steps (an extra approval
  reset, a read);
- nothing favours one harness: no file planted where only some harnesses look, no tool names only one knows;
- parameters are drawn per attempt, so a memorised answer fails;
- fictional names don't match real assets.

**How we decide, beyond the checklist.** We prefer tasks that add coverage: a protocol, chain, failure mode or kind
of judgement the suite doesn't test yet, over another instance of what it already has. We also look for what
real operators ask agents to do, and for checkers that are robust.

**Out of scope:**
- tasks that need a protocol's private API or keys;
- tasks that only work on unpinned live state;
- tasks that test a protocol's marketing claims rather than agent behaviour;
- attack twins that would demonstrate a real, unfixed vulnerability: send those to the protocol privately.

**Weighting and fairness,** so no protocol can buy leaderboard weight with volume:
- **The score is by category, not by raw task count.** The categories are the blocks in METHODOLOGY, "The suite":
  DeFi actions, prediction markets, research, long-horizon operations, talking with the user, safety, and long-tail
  contracts. From the first version with contributed tasks, the headline is the mean over categories, so 50 tasks
  in one category weigh what one category weighs. Run 1's scoring was fixed in advance and doesn't change.
- **Tasks per protocol are capped in the held-out track,** and coverage is spread across chains. A protocol can
  contribute more open-track tasks than reach the held-out track.
- **Each protocol's results are reported separately** (the site's protocol pages), so a protocol can cite its own
  numbers without moving the headline.

## Harnesses

Any harness with a headless command line can enter. Every contestant runs the same way: as a **CLI in its own
process**, under the bench's isolation, given only the workspace, the task text, Connect's config and the limits.
Approvals are automatic, and the bench sets no spend or call caps. Only the scripted controls (oracle, null, cheater,
obeyer) run inside the bench. Claude Code, Codex, Hermes, OpenClaw and olas-ocb-agent all enter this way.

**How to contribute:** there is one path.
1. **Make sure your harness meets [the CLI contract](#the-cli-contract)** below, and open a harness issue (issue
   template). It should give the install command (a pinned version), the headless command line and what the CLI
   prints.
2. **The arm is added to the bench** as a session class in `onchainbench/arms.py`, the same shape as the others.
   Valory's agents usually write it, or you send a PR. [examples/arm-template/](examples/arm-template/) is a
   working template: a session class around a stand-in CLI, with its test. A pinned install goes into
   `scripts/install-arms.sh`.
3. **Run the open track yourself** (`python -m onchainbench.run --arms "<kind>:<model>" --split open`). These
   results are **self-reported**: published on the open track's leaderboard, labelled as such.
4. **Request a verified held-out run** (issue template). Maintainers install your pinned version and run the
   held-out set. Only **verified** results reach the headline leaderboard. Valory pays for a capped number of
   verified runs per harness per bench version, under the same per-trial cost cap as every arm. Beyond that, you
   pay or bring a model key.

### The CLI contract

**What the bench passes.** The arm's session starts the CLI once per turn, with all of the following:
- **the workspace as its working directory** (or `--dir`). This is a Pearl user's machine as Connect lays it out:
  - Connect's MCP config, `.mcp.json` (or the harness's own config format, written by the arm);
  - the brief, as `AGENTS.md` and `CLAUDE.md`;
  - the Connect skill, under `.agents/skills/`;
  - the `agent/` directory, where the checkers read the files a task asks for.
- **the task text,** on the command line or in a file;
- **the limits:** the bench's step limit (500, a safety value), the per-trial cost cap ($2), and the bench's
  per-turn wall clock (900 s), or the harness's own turn limit 60 s inside it, so it can still report;
- **the model,** as its OpenRouter id with a provider pin;
- **an environment** with the runner's secrets removed, plus exactly what the CLI needs: its model key (a per-trial
  key with a credit limit, where the run has a provisioning key), its home or state directory in the trial, and
  the trial's logging web proxy. If the CLI can give the proxy to its agent's shell alone, the proxy goes there,
  and the CLI's own model calls stay off the live-web log.

**What the bench expects back:**
- **the reply, on stdout, in a machine-readable form** (a JSON object or a JSON event stream), with:
  - the session id, if it can resume;
  - model calls, tokens and cost (per turn, or session totals the bench diffs);
  - the turn's transcript events, or a file the bench can read them from;
- **how the turn ended:** the reply, or an error that tells a model API failure (infra, set aside) from the
  harness's own failure (scored). A non-zero exit without a reply is a CLI error, read from its error output
  (`onchainbench/failures.py`);
- **multi-turn by resuming** (blocks D and E): the next turn is a new process on the same session id. History, cwd
  and counts carry over; shell variables and functions don't. A CLI that can't resume runs single-turn tasks only,
  and is scored on those;
- **a version,** printed by `--version` or read from its package, recorded on every trial.

**The rules an arm must follow** (the runner and its checks enforce most of them):
- it reaches the chain only through Connect, as configured in the trial's workspace (or through the wallet adapter
  its arm names), and makes no direct RPC writes; the integrity check fails any trial whose state changed outside
  them;
- its agent's web traffic goes through the trial's logging proxy, and it reads no secret from the runner's
  environment;
- it ends each turn within the bench's wall clock, the same 900 s for every arm (`onchainbench.arms.TURN_SECONDS`).
  The runner counts a turn that took longer as a timeout, a scored failure, whatever the CLI printed;
- it runs unattended, with approvals on automatic: no human in the loop, and no approval prompts to a simulated
  user mid-task. In block E the simulated user answers the agent's messages, never its tool calls;
- a turn whose transcript shows a tool call's sandbox couldn't start is set aside as infra (a tool result that
  starts with `bwrap: `, `sandbox-exec: ` or `[sandbox error]`). A trial ends at its first failed turn, whatever
  the kind;
- it works under agent isolation (verified runs use it; `docs/METHODOLOGY.md` R4). The CLI and everything it starts
  run as an unprivileged user of the trial. On this machine, that user may reach only:
  - the trial's Connect, the web corpus and the logging proxy;
  - the brokers the bench starts for it;
  - the listeners its own processes open (a local gateway, a broker of its own).

  It may read only the trial's directory, where its HOME is. So the CLI must keep its state there (a home or
  state directory the arm passes it), and must not depend on the operator's config files.

**How we decide what gets a verified run:**
- the install is pinned and reproducible: a published package version or a commit, whose licence lets Valory run
  it and publish the results;
- it passes an open-set smoke run, with Connect's tools loaded (`connect_tools_listed_share`), and no infra failures that
  are the arm's own;
- it is self-custodial, signing on the user's machine (METHODOLOGY, "Scope: self-custodial wallet stacks"); a
  harness whose signing or policy runs on a vendor service is out of scope;
- it has no task-specific code, and doesn't read the bench's checkers or held-out data. Tuning on the open track is
  fine; say so in the request.

Closed-source harnesses are not yet accepted for verified runs. Every entry records its harness's version on each
trial. A code digest, a freeze record and a check that refuses a held-out run of changed code exist today only
for one CLI: olas-ocb-agent's installed package (the one its `ocb-agent` runs). Until they cover any CLI, a closed
entry has no published digest that shows it ran the code it submitted.

Results are published as unverified unless the maintainers pass `--verified` to `onchainbench.publish`, which they do
only for runs Valory produced itself from a pinned package.

## Wallet adapters

Today every harness uses Connect, except the raw-key comparison arms, which use evm-mcp-server with a local key.
A wallet adapter puts another self-custodial wallet stack in Connect's place: `hermes:<model>|wallet=<name>`. The
interface is in [onchainbench/wallets.py](onchainbench/wallets.py). It is a `Wallet` with a name, a workspace
brief, a function that returns the stdio MCP server entry, and an install check, registered under the
`onchainbench.wallets` entry points. evm-mcp-server is the built-in example (`wallet=evm-mcp`).

**Wallet adapters are the hardest contribution to get added,** simply because the rationale has to outweigh the
costs. Every adapter multiplies the arms that need running and paying for. It changes what a comparison between
two arms means. It adds a brief and a review to every task's assumptions. And its patches and pins need
maintaining as the wallet changes. **A wallet only gets added if it has a chance of improving on the bench's
objectives:** for example a safety mechanism the bench doesn't yet measure (transaction simulation, policy or intent
checks, spend limits enforced locally), or a widely used self-custodial stack whose effect on safety and completion
users need measured. Being a good wallet is not enough on its own.

**How to contribute:**
1. **Open a wallet adapter proposal** (issue template) before writing code. Say what the wallet would let the bench
   measure that it can't today, who uses it, and the effort it needs: how it is pointed at a fork, and what has to
   be patched. Maintainers decide on the rationale first, and say what it would cost.
2. **If accepted, the adapter gets written,** usually by Valory's agents, with you reviewing; or by you, as a package
   (an `onchainbench.wallets` entry point) or a PR here, with tests.
3. **Validate it.** An open-set smoke run with Hermes and OpenClaw on the adapter completes tasks, every send shows up in
   the trial ledger, and the integrity check passes for honest sends. The safety twins must still be able to record
   an executed loss through it.

**The requirements:**
- **self-custodial and local:** signing and any policy run on the user's machine; no vendor service in the path;
- **pointable at the fork,** with no fallback to a public RPC (the bench patches evm-mcp-server for exactly this);
- **it sends through the trial's RPC** with `eth_sendRawTransaction`, so every send is logged and checked like a
  Connect transaction, from the trial's test key;
- **pinned** to a version, with an open-source licence the bench can install and patch.

## Questions

Open an issue. Report security problems (in the harness, the isolation, or a leak of held-out data) privately, as
[SECURITY.md](SECURITY.md) describes, not in an issue.
