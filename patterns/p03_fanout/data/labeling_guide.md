# Labeling guide

Every label in `reviews.jsonl` follows these rules. The extraction prompts of both systems
state the same rules, so the models are evaluated against the criteria they were given.

## Overall sentiment

- `positive`: the review contains only positive evaluations.
- `negative`: the review contains only negative evaluations.
- `mixed`: the review contains both positive and negative evaluations.
- `neutral`: the review evaluates nothing (a question, a pure description).

Sentiment considers everything the reviewer evaluates, including things outside the aspect
list (microphone, app, design, customer support). Sarcasm is labeled by its intended meaning.

## Aspects

An aspect is labeled only when the reviewer **evaluates** it, not when it is merely mentioned.
Each aspect appears at most once per review, with polarity `positive` or `negative`.
Mild satisfaction ("fine", "decent", "as advertised") counts as `positive`.

| Aspect | Covers | Does not cover |
|--------|--------|----------------|
| `battery` | How long a charge lasts, charging time, battery indicator | A battery or port that stops working over time (`durability`) |
| `sound` | Playback quality, bass, clarity, volume and volume limits, noise cancelling | Microphone and call quality |
| `comfort` | Fit, pressure, weight, heat, whether it stays in or on | Looks and design |
| `connectivity` | Pairing, dropouts, range, latency, multipoint, device compatibility | Companion apps and software |
| `price` | Value for money, price compared with alternatives | |
| `durability` | Breakage, wear and failures that appear after use, or robustness that survives it | Damage present on arrival (`shipping`) |
| `shipping` | Delivery time, delivery problems, condition and packaging on arrival | Warranty, returns and customer support |

Not aspects (never labeled, but they do count toward sentiment): microphone, apps and software,
design and colors, customer support, warranty.

Comparisons with other products are labeled from the point of view of the reviewed product:
"sounds fuller than my old JBL" is `sound: positive`.