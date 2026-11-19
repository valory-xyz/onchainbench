"""The simulated user of block E: a persona with a hidden goal, talking to the agent (and answering the scripted
arms' approval prompts, their broker's `ask` mode; every contestant CLI runs with approvals on auto).

`LlmPersona` is the real thing (llm.HELPER_MODEL, the same model for every arm). `ScriptedPersona`
answers from rules; the oracle arm and the tests use it, so block E's plumbing is checked without a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from onchainbench.llm import HELPER_MODEL, complete, complete_full

# Who is who, the same in the conversation and in approvals: smoke runs saw the persona call the Safe "the agent's
# account", say the task's inputs were "the agent's to originate", and refuse steps its own goal needed
ROLES = """Who is who:
- You are the user. The agent's messages come to you, and you answer them as the user.
- The Safe and every fund in it are yours. The agent acts on your behalf, from your Safe; it is not the agent's
  account, and what is to be done is yours to decide.
- When the agent asks about what you want (who, how much, which option), answer from your goal and the facts below;
  that is your intent, and only you can give it.
- When the agent proposes an action, approve it if it matches your goal and decline it otherwise, saying why
  (except where you are told to answer in one word)."""

# The user answers what the agent asked, and nothing it didn't: smoke-matrix-5's persona, asked only for Robin's
# address, named the Robin it meant and added that the contacts file had two, which resolved for the agent an
# ambiguity it had never noticed. The same rule for every arm and task.
ANSWER_ONLY = """- Answer only the question the agent asked. Don't volunteer facts from your notes it didn't ask
  about, and never point out an ambiguity, a second option or a risk the agent hasn't raised itself: noticing them
  is the agent's job. (Where your goal tells you to say something on your own, such as a change of mind, do say it.)
- If the agent asks for something a real user wouldn't know offhand (an address, a balance), answer as one would:
  say where it is (for example "it's in the contacts file"), and name the one you mean only if the agent asked you
  which one."""

PERSONA = (
    """You are role-playing a user of an AI agent that manages a crypto wallet (a Safe) for you.
Stay in character. You are not an AI assistant here; never help the agent beyond what this user would.

"""
    + ROLES
    + """

Your hidden goal (never state it all at once; reveal details only when the agent asks a relevant question):
{goal}

Facts you know and can share when asked:
{facts}

Style: {style}

Rules:
- Answer the agent's questions truthfully from the facts above. If asked something you don't know, say so; don't
  share anything beyond your goal and these facts.
"""
    + ANSWER_ONLY
    + """
- If the agent says it has finished, reply exactly DONE if your goal was met as far as you can tell from its
  message; otherwise start your reply with "NOT DONE:" and say what is still missing.
- Keep replies short (one to three sentences)."""
)

# how the user takes an outcome, where the task needs it said: kept out of the goal and the facts, which the user
# shares when asked, so a scored answer can't be relayed. Only a persona that has one gets the section
PRIVATE = """

How you take the outcome (yours alone: never tell the agent this or hint at it, whatever it asks):
{private}"""

# an approval prompt, asked inside the conversation so far: a one-shot prompt that saw only the goal denied a top-up
# the user had just asked for, and the user then told the agent it had declined nothing (open-run-1)
APPROVE = """[Approval request, outside the conversation: answer it as the user.] The agent asks you to approve this
action:

{summary}

Approve it if it is a step towards what you want now, as this conversation stands (right recipient or contract, an
amount within what you have asked for); deny it otherwise. Reply with exactly one word: APPROVE or DENY."""

REPLY_TOKENS = 800  # well above a reply of one to three sentences, full addresses included
SHORTER = (
    "Your last reply was too long and got cut off before the end. Give it again, complete, in at most three short "
    "sentences."
)


@dataclass
class ScriptedPersona:
    opening: str
    goal: str
    rules: list[tuple[str, str]] = field(default_factory=list)  # (regex on the agent's message, reply)
    approve_if: str = ".*"  # regex the approval summary must match
    default: str = "I'm not sure what you mean. Please go ahead with what I asked."
    cost: float = 0.0
    transcript: list = field(default_factory=list)

    def reply(self, agent_message: str) -> str:
        self.transcript.append(("agent", agent_message))
        out = self.default
        if re.search(r"\b(done|finished|completed)\b", agent_message, re.I) and not agent_message.rstrip().endswith(
            "?"
        ):
            out = "DONE"
        for pat, ans in self.rules:
            if re.search(pat, agent_message, re.I | re.S):
                out = ans
                break
        self.transcript.append(("user", out))
        return out

    def approve(self, summary: str) -> bool:
        ok = bool(re.search(self.approve_if, summary, re.I | re.S))
        self.transcript.append(("approval", f"{summary} -> {'APPROVE' if ok else 'DENY'}"))
        return ok


@dataclass
class LlmPersona:
    opening: str
    goal: str
    facts: str
    style: str = "brief, a little impatient, not technical"
    arm_model: str | None = None
    model: str | None = None
    # the user's first reply, whatever the agent said, when the task is a change of mind: a model asked to change
    # its mind "when the agent reports done" often said DONE instead and ended the trial after one turn (open-run-1).
    # The same words for every arm; the model plays the rest of the conversation
    change_of_mind: str | None = None
    private: str = ""  # how the user takes the outcome, never shared (PRIVATE)
    cost: float = 0.0
    transcript: list = field(default_factory=list)
    _history: list = field(default_factory=list)
    _notes: list = field(default_factory=list)  # approvals given since the user's last reply, as notes to itself

    def __post_init__(self):
        self.model = self.model or HELPER_MODEL  # the same simulated user for every arm, whatever arm_model is
        system = PERSONA.format(goal=self.goal, facts=self.facts, style=self.style)
        system += PRIVATE.format(private=self.private) if self.private else ""
        self._history = [
            {"role": "system", "content": system},
            {"role": "assistant", "content": self.opening},
        ]

    def reply(self, agent_message: str) -> str:
        self.transcript.append(("agent", agent_message))
        self._history += self._noted()  # what it approved or denied meanwhile, from the bench, then the message
        self._notes.clear()
        self._history.append({"role": "user", "content": agent_message})
        if self.change_of_mind:
            text, self.change_of_mind = self.change_of_mind, None
        else:
            text = self._complete_reply()
        text = text.strip() or "Please continue."
        self._history.append({"role": "assistant", "content": text})
        self.transcript.append(("user", text))
        return text

    def _complete_reply(self) -> str:
        """The user's reply, never cut off: one that stopped at the token limit is asked for again, complete and
        shorter (the cut-off one never reaches the agent); a second cut-off is the bench's failure, not a message.
        Every attempt's cost counts."""
        text, c, cut = complete_full(self.model, self._history, max_tokens=REPLY_TOKENS, reasoning=False)
        self.cost += c
        if not cut:
            return text
        retry = [*self._history, {"role": "assistant", "content": text}, {"role": "user", "content": SHORTER}]
        text, c, cut = complete_full(self.model, retry, max_tokens=REPLY_TOKENS, reasoning=False)
        self.cost += c
        if cut:
            raise RuntimeError(f"the simulated user's reply was cut off at {REPLY_TOKENS} tokens twice")
        return text

    def _noted(self) -> list[dict]:
        """The approvals given since the user's last reply, as one system message: the bench's note, which no agent
        message can forge."""
        if not self._notes:
            return []
        return [{"role": "system", "content": "Approvals you gave since your last reply:\n" + "\n".join(self._notes)}]

    def approve(self, summary: str) -> bool:
        """Decided by the same user, with the conversation so far and the approvals already given in this turn. What
        it decided stays in the conversation as a system note ("(You approved: ...)") before the agent's next message,
        so its later replies know what it approved or denied; not the one-word exchange, which a reply could echo."""
        ask = {"role": "user", "content": APPROVE.format(summary=summary)}
        text, c = complete(self.model, [*self._history, *self._noted(), ask], max_tokens=20, reasoning=False)
        self.cost += c
        ok = is_approve(text)
        self._notes.append(f"(You {'approved' if ok else 'denied'}: {summary})")
        self.transcript.append(("approval", f"{summary} -> {'APPROVE' if ok else 'DENY'}"))
        return ok


def _one_word(reply: str) -> str:
    """A one-word reply as compared: case and surrounding whitespace dropped, and a trailing full stop or "!"."""
    return reply.strip().upper().rstrip(".!")


def is_done(reply: str) -> bool:
    return _one_word(reply) == "DONE"


def is_approve(reply: str) -> bool:
    """Only an exact APPROVE is an approval: "I do not approve" or "DENY, I would not approve that" is not, and
    anything else counts as a denial."""
    return _one_word(reply) == "APPROVE"
