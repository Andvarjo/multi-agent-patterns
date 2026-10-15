# Multi-Agent Patterns

Multi-agent architecture patterns, built from scratch with LangGraph and **measured against a single-agent baseline**.

Most multi-agent demos show that a pattern *can* be built. This repo asks a harder question: **when is it actually worth it?** Every pattern is implemented end to end, evaluated on a dataset with an LLM judge, and compared against one well-written prompt on quality, tokens and latency. The findings, including the negative ones, are documented in each pattern's README.

## Why this repo exists

I built this repository with three goals:

1. **Experiment.** Implement the main multi-agent architectures one by one, from a simple sequential pipeline to a full multi-agent system, and see how they behave in practice rather than on paper.
2. **Share what I learn.** Each pattern documents its design decisions, results, failure modes and open questions, so other engineers can reuse the conclusions, not just the code.
3. **Show how I build.** The repo is a working portfolio: it shows how I design agent systems, structure shared infrastructure, set up evaluations, and reason about trade-offs with data. The commit history shows how each piece was built incrementally.

## Key result so far

> **Pattern 01 (sequential pipeline):** on a 16-article summarization benchmark, a three-agent pipeline used **~2.4x the tokens and latency** of a single call and scored **lower** on fidelity, coverage and concision. Each handoff between agents lost information, such as hedges, attributions and intentions, that later agents could not recover.
>
> Lesson: decomposition is not free. A multi-agent design has to earn its cost against a strong single-call baseline.

Full analysis: [`patterns/p01_pipeline/README.md`](patterns/p01_pipeline/README.md)

## Patterns

| # | Pattern | Project | What it explores | Status |
|---|---------|---------|------------------|--------|
| 01 | [Sequential pipeline](patterns/p01_pipeline/) | Technical article summarizer | Shared state, reducers, prompt specialization, handoff loss | ✅ Done |
| 02 | Router | Support assistant | Conditional routing, structured outputs, router accuracy | ⏳ Next |
| 03 | Parallel fan-out / fan-in | Product review analyzer | Concurrency, aggregation, latency vs. tokens | 📋 Planned |
| 04 | Generator + critic | Unit test generator | Loops, stop conditions, deterministic feedback | 📋 Planned |
| 05 | Supervisor with tools | Research assistant | Dynamic orchestration, tool calling, tracing | 📋 Planned |
| 06 | Handoffs + human-in-the-loop | Trip planner | Control transfer, persistence, human approval | 📋 Planned |
| 07 | Capstone | Multi-agent PR reviewer | Combining the patterns above in a real workflow | 📋 Planned |

## How every pattern is evaluated

Each pattern follows the same methodology, so results are comparable across the repo:

- **A single-agent baseline** with the same instructions and the same interface as the multi-agent system.
- **A dataset of cases** designed to include easy, long and deliberately tricky inputs.
- **An LLM judge** with an explicit rubric, using a stronger model than the agents under test, and structured output so scores are typed and validated.
- **Multiple runs per case**, because model outputs vary between runs.
- **Tokens and end-to-end latency** tracked per agent. Tokens are reported instead of dollars because prices change.
- **Failures recorded as data**, with system errors and judge errors tracked separately.
- **Raw results committed** next to each pattern, so every number in a README can be recomputed.

## Repository structure

```
multi-agent-patterns/
├── core/                    # shared infrastructure, grown as patterns need it
│   ├── llm.py               # model factory by role + LLM call with token/latency metrics
│   ├── data.py              # JSONL dataset loader
│   └── harness.py           # runs systems over a dataset, judges, saves and summarizes results
├── patterns/
│   └── p01_pipeline/        # one folder per pattern
│       ├── state.py         # graph state
│       ├── agents.py        # agents (graph nodes) and their prompts
│       ├── graph.py         # LangGraph wiring and entry point
│       ├── baseline.py      # single-agent baseline
│       ├── judge.py         # evaluation rubric
│       ├── run.py           # evaluation entry point
│       ├── data/            # evaluation dataset
│       ├── results/         # raw results cited in the README
│       └── README.md        # design, results and findings
└── results/                 # local evaluation runs (git-ignored)
```

Shared code lives in `core/` only once a second use for it is clear; everything specific to a pattern stays in its folder.

## Getting started

Requirements: [uv](https://docs.astral.sh/uv/) and an Anthropic API key.

```bash
git clone https://github.com/<your-user>/multi-agent-patterns.git
cd multi-agent-patterns
uv sync
cp .env.example .env   # add ANTHROPIC_API_KEY, WORKER_MODEL and JUDGE_MODEL
```

Run a pattern and its evaluation:

```bash
uv run python -m patterns.p01_pipeline.graph              # run the pipeline on one case
uv run python -m patterns.p01_pipeline.run --limit 2 --runs 1   # quick evaluation smoke test
uv run python -m patterns.p01_pipeline.run                # full evaluation
```

> **Python version.** The project targets Python 3.14. Python 3.15 was released in October 2026, but at the time of writing some dependencies (PyYAML, via `langchain-core`) did not yet ship wheels for it, and building from source failed. Moving to 3.15 should only require changing `.python-version` and `requires-python` once wheels are available.

## Tech stack

- **[LangGraph](https://docs.langchain.com/oss/python/langgraph/graph-api)** for agent orchestration
- **Claude models** via `langchain-anthropic`, selected by role (`worker` for agents, `judge` for evaluation) through environment variables
- **Pydantic** for structured judge output
- **uv** for dependency management and **Ruff** for formatting and linting

## About

I'm Andrés Felipe Varela, a Senior AI Engineer working on LLM systems in production. I built this repo to go deeper into multi-agent architectures with an engineering mindset: build each pattern, measure it honestly, and document when it helps and when it doesn't.

- LinkedIn: [Andrés Felipe Varela](https://www.linkedin.com/in/andresvj/)
- Email: [andresvarela92@hotmail.com](mailto:andresvarela92@hotmail.com)

Feedback, questions and discussion are welcome through GitHub issues.

## License

[MIT](LICENSE)