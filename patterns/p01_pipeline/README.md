# Pattern 01: Sequential Pipeline

A fixed chain of specialized agents where each agent's output becomes the next agent's input. There is no routing and no loops: the order of execution is decided in code, not by the model.

```mermaid
graph LR
    START((start)) --> extractor
    extractor -->|key_points| writer
    writer -->|draft| editor
    editor -->|summary| END((end))
```

## Use case

Summarize a technical article into one paragraph for a senior engineering audience, split into three steps:

| Agent | Input | Output | Responsibility |
|-------|-------|--------|----------------|
| `extractor` | `article` | `key_points` | Pull 4–7 concrete, fact-bearing points from the article |
| `writer` | `key_points` | `draft` | Turn the points into a connected paragraph (≤120 words) |
| `editor` | `draft` | `summary` | Tighten wording without changing any facts |

Each agent only sees the previous step's output, wrapped in XML tags (`<article>`, `<key_points>`, `<draft>`). Its system prompt says where it sits in the pipeline and who consumes its output.

## State

All agents share a single `PipelineState` (`state.py`):

```python
class PipelineState(TypedDict, total=False):
    article: str
    key_points: str
    draft: str
    summary: str
    usage: Annotated[list[LLMUsage], operator.add]
```

- `total=False` because the graph starts with only `article`; each node fills in its own field.
- `usage` uses an `operator.add` reducer, so every node appends its token and latency record instead of overwriting the list.

## Files

| File | Contents |
|------|----------|
| `state.py` | `PipelineState` definition |
| `agents.py` | System prompts and the three node functions |
| `graph.py` | LangGraph wiring, `run_pipeline()` entry point |
| `baseline.py` | Single-call baseline used for comparison |

## Running

Requires `ANTHROPIC_API_KEY` and `WORKER_MODEL` in `.env` (see `.env.example`).

```bash
uv run python -m patterns.p01_pipeline.graph
```

This prints the final summary, the per-agent usage records, and the graph as Mermaid.

## Preliminary experiment: pipeline vs. single-call baseline

`baseline.py` runs the same task with a single model call. Its prompt asks the model to first identify the key points and then write the summary, under the same output constraints. Both systems were run once on the sample REST → gRPC article.

| System | Calls | Input tokens | Output tokens | Latency |
|--------|------:|-------------:|--------------:|--------:|
| Pipeline | 3 | 1,001 | 265 | ~4.9 s |
| Baseline | 1 | 325 | 99 | ~1.9 s |

The baseline cost roughly a third as much and finished in less than half the time.

**Quality.** Both summaries kept all four figures, but the baseline added content that is not in the article:

- *"delivering substantial performance gains"*: the article never calls the improvement "substantial". That word is the model's interpretation.
- *"to support gRPC"*: also absent from the text, even if it is a reasonable inference.

The pipeline stuck to the facts. A likely reason is that the writer never saw the article, only the extracted key points, and the editor was explicitly forbidden from adding anything.

**Hypothesis.** Splitting the task into isolated steps may improve faithfulness, at roughly 3x the cost.

**Limitations.** This is one article and one run. Model outputs vary between runs, and on a different article the result could flip. Turning this hypothesis into a finding requires:

1. **More articles** of different lengths and information densities (a JSONL dataset).
2. **A consistent quality criterion** instead of manual reading (an LLM judge).
3. **Multiple runs per case**, to account for run-to-run variability.

## When to use it

**Good fit**
- The task breaks down into stable, ordered steps.
- Each step benefits from a narrow, focused prompt.
- You want predictable cost and latency (exactly N model calls).

**Trade-offs**
- Latency adds up: steps run strictly one after another.
- Errors propagate. If the extractor misses a point, later agents cannot recover it.
- No feedback loop: the editor cannot send work back to the writer.

Later patterns in this repo address these limits with routing, parallelism, and review loops.
