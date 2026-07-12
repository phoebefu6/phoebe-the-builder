# Streaming Window Aggregator

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-engineering-pro/streaming-aggregator/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-engineering-pro/streaming-aggregator/demo.ipynb)

> We can't compute rolling metrics live — event-time windows with watermarks in ~100 lines, with the same semantics Flink and Spark charge a cluster for.

## Business Impact
- **Before:** "Rolling 5-minute conversion rate" means a batch query that's 20 minutes stale, and out-of-order events silently land in the wrong bucket or vanish.
- **After:** Windows finalize live as the watermark advances; every event is either windowed or counted as explicitly late — a conservation guarantee most pipelines never state.
- **Estimated ROI:** Live operational metrics without standing up a streaming cluster; the mental model transfers 1:1 when the team does adopt Flink/Spark.

## Tech Stack
Python 3.10+ (pure stdlib core), Streamlit, pandas, matplotlib. Tumbling + sliding windows, per-key aggregates (count/sum/mean/min/max), configurable allowed lateness. Runs fully offline.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works
1. **Event time, not arrival time** — events carry when they happened; the simulator delivers them out of order (mild disorder + very-late stragglers).
2. **Watermark** — max event time seen minus `allowed_lateness`: the stream's claim that nothing older is still coming.
3. **Finalization** — a window is emitted (immutable) once the watermark passes its end; open windows keep accumulating.
4. **Late handling** — events arriving after their window closed increment `late_dropped`, never silently merge or vanish; the notebook asserts `windowed + late == total`.
5. **Tumbling vs sliding** — slide == window gives one bucket per event; slide < window gives overlapping windows (3x points, 3x state at 60s/20s).

Demo: 2,000 events, 10 minutes, 3 keys → 30 tumbling windows finalized live, 35 late drops, conservation proven; lateness sweep shows the correctness-vs-delay trade-off.

## Learning Connection
Built while studying Kafka/streaming concepts (Month 7: Data Engineering Pro).
Applies: event-time vs processing-time, watermark semantics, window finalization, and late-data accounting.

## Impact Note
- **Who benefits:** Teams needing live rolling metrics before (or instead of) a streaming cluster; anyone learning what `withWatermark()` actually does.
- **Potential risks:** In-memory state only — a process restart loses open windows; production use needs checkpointing (which is exactly what the big engines add).
