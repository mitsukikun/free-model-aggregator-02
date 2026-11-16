# free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/mitsukikun/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/mitsukikun/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server that serves an OpenAI- and an Anthropic-compatible API over the fourteen free
OpenRouter models, routing each request to the best free model that can serve it, failing over
when a model refuses, and reporting the catalog it read. Quality-ranked routing over the
brief's published ratings, failover with per-model cooldowns and a recovered-refusals header,
both client wires with the Anthropic side translated to and from Chat Completions, in JSON and
server-sent events, with a `--list` report of the catalog it found.

## Run it

```sh
MAKEFUN_ENV_URL=http://localhost:8080 MAKEFUN_PROMPT_FILE=prompt.txt node main.ts
node main.ts --list
```

## How it routes

- Candidates are the brief's free models that meet the prompt's requirements, ranked by
  `rating_milli` desc, then by fewest calls made to that model, then by model id.
- A model that refuses (429/5xx/timeout), or answers without the prompt's token, without a
  stop reason, or without usable usage, is cooled down for the next prompt and the request is
  retried on the next candidate (at most 4 attempts, timeouts 9/7/5/3 s).
- Every recovered refusal is reported on the answer in the `x-makefun-upstream-failures`
  header (JSON array of `{status, code, model}`).
- A 401/403/402 stops the trial's gateway calls (the key is missing or spent); a 400/404 or
  a "budget is spent" refusal marks that model dead for the trial.
- Every answer names the model actually dialled (`openrouter/<vendor>/<model>:free`); the
  provider's bare id is put back to the dialled id on every answer and streamed chunk.

---

<sub>Entered on [make.fun](https://make.fun/?ref=3fk7ms5j&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
