<!-- CONTRIBUTING.md has the process. Issues are preferred over PRs: Valory's agents can often do the work faster. Link the issue where the scope was agreed. -->

Issue: #

**For tasks**
- [ ] `onchainbench/tasks/contrib_<protocol>.py`, ids `<category letter>-<protocol>-<name>`, protocols in `PROTOCOLS.json`
- [ ] Validated: oracle passes on both attempts, null and cheater fail, the obeyer loses on twins (`scripts/validate_run.py`)
- [ ] The review checklist in CONTRIBUTING.md, "Tasks"

**For code (arms, wallet adapters, bench changes)**
- [ ] Tests, with 100% coverage (`scripts/check.sh`)
- [ ] Does it change what a score measures? If so, say how: it needs a new bench version
