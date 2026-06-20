# ochat — Quick Use Guide

Terminal chat with a local Ollama model that remembers conversations across
sessions and recalls facts about you over time.

## Prerequisites

Ollama must be running, with both models pulled:

```bash
ollama serve                       # if not already running
ollama pull gemma4:12b             # the chat model
ollama pull nomic-embed-text       # the embedding model (for memory search)
```

## Installation

```bash
mkdir -p ~/.local/bin
ln -sf /Users/developer/ochat/ochat.py ~/.local/bin/ochat
```

Make sure `~/.local/bin` is on your `PATH`. No virtualenv or `pip install`
needed — the script declares its own dependencies (`numpy`, `requests`) and
`uv run --script` resolves them automatically the first time it runs.

## Running it

```bash
ochat
```

You'll see a `you>` prompt. Type a message and press Enter; the reply streams
back under an `ochat>` prefix. Press **Ctrl-D** (EOF) to quit — your
conversation is saved automatically after every reply, so resuming later just
means running `ochat` again.

## Named threads

Each thread is a separate, independently-resumable conversation:

```bash
ochat --thread work
ochat --thread personal
```

Omitting `--thread` uses a thread named `default`.

## Controlling response speed vs. reasoning depth

```bash
ochat --think off      # fastest — default, no hidden reasoning step
ochat --think low
ochat --think medium
ochat --think high
```

Lower thinking levels answer faster; higher levels let the model reason more
before responding, at the cost of latency. `off` is the default because it's
roughly 9x faster for typical questions with no loss of correctness.

## Listing your threads

```bash
ochat threads
```

Shows every thread's name, message count, and last-updated time.

## Managing long-term memory

After each reply, `ochat` automatically (and invisibly) asks the model what's
worth remembering long-term, and stores any new facts. You don't have to do
anything for this to happen — but you can inspect and curate what it knows:

```bash
ochat memory list              # see every fact it has stored, with its id
ochat memory forget <id>       # delete a specific fact by id
```

Facts are shared across all threads — something it learns in one
conversation can be recalled in a completely different one, days later.

## Calendar awareness

`ochat` now always knows the current date and time, and mentions it to the
model automatically on every turn — so it can correctly figure out what
"tomorrow" or "next Thursday" actually means, instead of guessing.

On macOS, it goes a step further:

- Your upcoming events from Calendar.app (the next 7 days, across every
  calendar) are quietly pulled in as context, so you can ask things like
  "what do I have on Thursday?" and get a real answer.
- If you ask it to add something — "schedule a dentist appointment next
  Tuesday at 2pm" — it figures out the actual date/time you mean, and asks
  you to confirm before writing anything:

  ```
  Add to calendar? "Dentist appointment" -- Tue, Jun 23 2026, 02:00 PM-03:00 PM [y/N]
  ```

  Nothing is added to your real calendar unless you type `y` or `yes`.

- You can also just list what's coming up directly, without chatting at all:

  ```bash
  ochat calendar list
  ```

None of this works off macOS — `ochat` just quietly skips it and keeps
working normally (date/time awareness in chat still works everywhere).

**One-time setup note:** the very first time `ochat` tries to read or write
your calendar, macOS will pop up a permission dialog (Automation and/or
Calendars, under **System Settings > Privacy & Security**). Only you can
approve that dialog — `ochat` can't click through it for you. If calendar
features seem to silently do nothing, check that you actually approved the
prompt (or that it isn't sitting unanswered behind another window), and
that ochat/Terminal has access under **Privacy & Security > Automation**
and **Privacy & Security > Calendars**.

## Where your data lives

```
~/.local/share/ochat/
  threads/<name>.json     # full conversation history for each thread
  memory.db               # long-term facts (plain SQLite database)
  extraction.log          # background fact-extraction errors, if any occur
```

Everything is a plain, human-readable file (or a standard SQLite database)
you're free to open, back up, or edit directly — there's no hidden state and
no background service running when you're not using `ochat`.

## Troubleshooting

| Symptom | What to do |
|---|---|
| `error: Ollama isn't reachable at http://127.0.0.1:11434` | Run `ollama serve` |
| `error: required model '<name>' is not installed` | Run the `ollama pull <name>` command it prints |
| `error: chat request failed (...); message not saved, try again` | Just resend your message — nothing was lost, the failed turn writes nothing to disk |
| It doesn't seem to remember something you just said | Background fact-extraction takes a few seconds after the reply — check `ochat memory list` a little later, or `~/.local/share/ochat/extraction.log` for errors |
| A thread file got corrupted | `ochat` detects this automatically, renames it to `<name>.json.corrupt-<timestamp>` in `~/.local/share/ochat/threads/`, and starts that thread fresh — your old data isn't deleted, just set aside |

## Uninstalling

```bash
rm ~/.local/bin/ochat
rm -rf ~/.local/share/ochat      # deletes all threads and remembered facts
```
