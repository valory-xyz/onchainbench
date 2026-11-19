# Judge labels

The LLM judges' validation audit trail: the items the judges were checked on, every labelling step, and the result.
A judge model's scores count in the headline only once it agrees with these labels on at least 90% of criteria
(`python -m onchainbench.judge validate`). The labels are model-made, not a person's: the steps and commands are in
`onchainbench/judge.py`'s docstring and the README's "Statistics" section.

| File | What it is |
|---|---|
| `todo.jsonl` | The items to label, exported from an open-set run (`judge export`): the rubric criteria, the agent's work, the reference facts and the judge's scores. |
| `prelabel.jsonl` | The same items with pre-labels from a model of another family than the judge (`judge prelabel`). |
| `review.jsonl` | The items picked for review (`judge review`): every item where the pre-label and the judge disagreed, plus a random spot check of those where they agreed. |
| `opus.jsonl` | The reviewer's labels on the review set (`judge label`). |
| `merged.jsonl` | Pre-labels and reviewed labels merged (`judge merge`), as first written. |
| `labels.jsonl` | The labels the judges are validated on. `label_source` is `agreed` for a pre-label taken without review and `human` for a reviewed item (reviewed by a model, not a person). |
| `rejudge.jsonl` | The labelled items judged again on the current rubric criteria (`new_criteria`) by the judge and the pre-label model (`haiku_new`, `deepseek_new`), beside the earlier scores. |
| `rejudge-gemini-3.8-flash.jsonl` | A second judge model's scores on the labelled items (its validation rows). |
| `validated.json` | The judge models that passed validation; only their scores count in the headline. |
