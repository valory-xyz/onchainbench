# Security policy

## Reporting a vulnerability

Report vulnerabilities privately, through GitHub's private vulnerability reporting on this repository (the
*Security* tab, "Report a vulnerability"), once the repository is public. Please don't open a public issue, pull
request or discussion for one, and don't publish details until a fix is out.

Include what you found, the bench version or commit, how to reproduce it, and what an attacker gains.

## In scope

- **The harness:** the runner, fork Connect and its guardrail, intent gate, RPC allowlist and integrity checks, the
  brokers, the checkers and judges, and the publish and verify steps. For example, a way for an agent to pass a
  task it didn't do, to hide an executed loss, or to reach a chain other than its trial's fork.
- **The isolation:** a way for an agent under test to escape its trial's sandbox or unprivileged user, read the
  operator's files or keys, reach the network beyond what its trial allows, or affect another trial.
- **Leaks of held-out data:** any held-out task id, instruction text, parameters or checker logic that can be
  recovered from this repository, a published run, the site or the harness's behaviour.

Out of scope: vulnerabilities in the protocols the tasks use, or in a contestant's harness (report those to their
maintainers), and attack twins that would demonstrate a real, unfixed vulnerability (send those to the protocol
privately, as CONTRIBUTING.md says).

## What to expect

We acknowledge a report within 5 working days and keep you updated until it is resolved. We aim to fix a confirmed
issue in the harness or the isolation before the next scored run, and to say in the run's notes which published
results it affected; for a confirmed held-out leak we assess which tasks it exposes. With your permission, we
credit you in the fix's release notes.
