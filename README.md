# A reliability-evaluation framework for Chinese connective-sense annotation

> **What this repository contains.** Code only. Raw corpora, the processed files
> derived from them, and the model outputs are **not** redistributed here: the
> corpora carry their own licences, and the processed files embed the corpus
> sentences. `data/` and `outputs/` are therefore git-ignored; see §2 for how to
> rebuild them from the public releases.

---

## 1. Data status (read this first)

| Resource | Status | Notes |
|---|---|---|
| **TED-CDB** (Long et al., EMNLP 2020) | ✅ **downloaded** | PDTB 3.0 three-level senses, 72 Chinese TED talks, 12,983 relations. `data/raw/ted-cdb/` |
| **Chinese-DiMLex** | ✅ downloaded | CC BY-NC-SA 4.0, 226 connectives / 260 sense exemplars. `data/raw/chinese-dimlex/` |
| Chinese TED 16 talks (LREC 2020) | not downloaded | Explicit CC BY 4.0; a subset of TED-CDB. Use it when you need unambiguous redistribution rights. |
| **HIT-CDTB** (Harbin Institute of Technology) | ✅ **50-document sample parsed** (3,120 relations, p2+p3) | The full 525 documents require a signed data agreement; the sample does not. `data/raw/hit-cdtb/` |
| Others (SciDTB / GCDT / SCTB) | not downloaded | Different frameworks (dependency / RST); their sense inventories are incompatible with PDTB, so they can only supplement. |

### 1.1 Why TED-CDB for now

CDTB and TED-CDB are both Chinese corpora in the PDTB framework, but they **cannot be compared in a single table**:

| | CDTB 0.5 | TED-CDB |
|---|---|---|
| Genre | Xinhua newswire (written) | TED talks (planned speech) |
| Size | ~5,500 relations | **12,983 relations** |
| Sense inventory | PDTB 2.0/3.0 style | **PDTB 3.0, three levels** (`Contingency.Cause.Reason`) |
| Published 4-class F1 | ~95 | ~60 |
| Access | LDC, paid | public repository |

### 1.2 HIT-CDTB: loaders, offsets and mapping

The 50-document public sample is wired in (`data/raw/hit-cdtb/extracted/HIT_Discourse_Example/`, 200 files).

**File format.** Each document comes as `.txt` (GB18030, space-tokenised) plus annotation files at three granularities:

| Suffix | Granularity | Relations in the sample |
|---|---|---|
| `.p1` | sentence group | 162 |
| `.p2` | sentence | 1,963 |
| `.p3` | clause | 1,157 |

`load_hitcdtb` reads `p2,p3` only by default: PDTB has no sentence-group layer, and including `.p1` would count the same stretch of text three times. Override with `--hit-levels p1,p2,p3`.

**Three pitfalls you need to know about:**

1. **Line endings.** Offsets are computed on "universal newline" text, where `\r\n` counts as **one** character. `Path.read_text()` performs that translation; `read_bytes().decode()` does not, and every offset then shifts by the number of line breaks before it. Measured on `mz/chtb_1010#.txt`: 5,182 characters after GB18030 decoding, 5,116 after normalisation — the difference of 66 is exactly 66 `\r\n` pairs. The error **varies per file and is not a constant**, so it cannot be compensated downstream; `_read()` normalises instead.
2. **Span format.** HIT writes `a...b`, and **`b` is exclusive** (the opposite of TED-CDB's inclusive `a..b`). Paired connectives are written `504...507;525...527`, with `;` separating Content (`不仅;同时`). 162 of the sample's 1,050 explicit blocks take this form.
3. **Implicit connectives.** For implicit relations the `Span` is an insertion point (zero width) and `Content` is the connective the annotator **supplied**. It goes into `meta["implicit_connective"]` and `Item.connective` is left empty — otherwise the label leaks.

**Mapping table.** `HIT_TO_PDTB` lives in `src/discourse_reliability/data/hitcdtb.py`; the four top-level classes map onto PDTB 3.0 one to one:

| HIT | PDTB L1 | Notes |
|---|---|---|
| 1 时序 (temporal) | `TEMPORAL` | synchronous / precedence / succession |
| 2 因果 (causal) | `CONTINGENCY` | cause, inference, purpose; **conditions (2-4/2-5/2-6) also sit here** |
| 3 比较 (comparison) | `COMPARISON` | contrast / concession |
| 4 扩展 (expansion) | `EXPANSION` | conjunction, progression, specification, generalisation, parallel, alternative, list |

So **only L1 can be compared directly with TED-CDB**; L2/L3 are a lossy project-level mapping, and the parts flagged `mapping_lossy=True` (e.g. HIT's "formal condition" / "arbitrary condition" folded into `CONTINGENCY:Condition`) survive only in `meta["hit_label"]`.

```bash
python run.py build-data --source hitcdtb \
  --input data/raw/hit-cdtb/extracted/HIT_Discourse_Example \
  --out data/processed/hitcdtb.jsonl        # -> 3,120 relations
```

**Correctness check.** For all 3,282 blocks (p1+p2+p3) we verified `detokenize(tokens[Arg.Span]) == Arg.Content` block by block: Arg1/Arg2 match with **zero errors across all 3,282 blocks**; of the 1,050 explicit-connective blocks, 1,048 match exactly and the remaining two are genuine content differences in the corpus itself (the span for `其他观点？` carries an extra question mark; the source has `其＋原因` while Content says `其原因`). Those two are left uncorrected: `Item.connective` keeps Content and `meta["conn_span_text"]` keeps the span text, so the discrepancy stays inspectable instead of being silently swallowed.

---

## 2. Quick start (no API key required)

```bash
cd discourse-reliability

# 0) build the main dataset from TED-CDB (the recommended primary corpus)
python run.py build-data --source tedcdb \
  --input data/raw/ted-cdb/extracted/TED-CDB-main/TED_CDB_Data \
  --dimlex-xml data/raw/chinese-dimlex/chinese-dimlex-main/chinese_dimlex.xml \
  --out data/processed/tedcdb.jsonl

python run.py sample --input data/processed/tedcdb.jsonl \
  --out data/processed/tedcdb_pilot.jsonl --n 600 \
  --explicit-fraction 0.45 --min-per-stratum 5 --assign-splits

# 1) build the dataset from Chinese-DiMLex
python run.py build-data --source dimlex \
  --input data/raw/chinese-dimlex/chinese-dimlex-main/chinese_dimlex.xml \
  --out data/processed/dimlex.jsonl

# 2) stratified sample of 200 items (DiMLex only has 260)
python run.py sample --input data/processed/dimlex.jsonl \
  --out data/processed/dimlex_pilot.jsonl --n 200 \
  --explicit-fraction 1.0 --min-per-stratum 3 --assign-splits

# 3) generate the human-annotation material
python run.py prep-annotation --input data/processed/dimlex_pilot.jsonl \
  --out-dir data/interim/r1

# 4) smoke-test the pipeline with a mock model (testing only)
python run.py run --input data/processed/dimlex_pilot.jsonl \
  --out outputs/mock.jsonl --model mock --samples 5 --temperature 0.7

# 5) evaluation + triage
python run.py evaluate --gold data/processed/dimlex_pilot.jsonl --preds outputs/mock.jsonl --out outputs/eval.json
python run.py triage  --gold data/processed/dimlex_pilot.jsonl --preds outputs/mock.jsonl --out outputs/triage.json

# 6) counterfactual variants + human-readability worksheet
python run.py counterfactual --input data/processed/dimlex_pilot.jsonl \
  --out outputs/variants.jsonl --validate-csv outputs/variants_review.csv
```

---

## 3. Connecting real models

```bash
export OPENAI_API_KEY=...
export DEEPSEEK_API_KEY=...
export DASHSCOPE_API_KEY=...      # Qwen
export ZHIPU_API_KEY=...          # GLM
export ANTHROPIC_API_KEY=...

python run.py run --input data/processed/cdtb_600.jsonl \
  --out outputs/gpt52_zero.jsonl --model openai:gpt-5.2 --regime zero_shot \
  --samples 5 --temperature 0.7 --workers 8
```

Supported specs: `openai:<model>`, `deepseek:<model>`, `qwen:<model>`, `glm:<model>`, `anthropic:<model>`, `chatanywhere:<model>` (multi-vendor gateway), `compat:<base_url>|<model>`.

The gateway key is read from `CHATANYWHERE_API_KEY`:

```bash
export CHATANYWHERE_API_KEY=...
python run.py run --input data/processed/l1/hit_balanced.jsonl \
  --out outputs/l1_hit_balanced_gpt6luna_zs.jsonl \
  --model chatanywhere:gpt-6-luna --level l1 --max-tokens 1024
```

**Parameter adaptation.** Models behind a gateway do not agree on parameter names (`gpt-6-luna` accepts only `max_completion_tokens` and rejects `temperature`). When the adapter sees a 400 whose `error.param` points at such a field, it drops the field and retries — each instance pays for one failed request only. Truncated responses (`finish_reason=length`) are reported as `truncated` rather than disguised as "unparseable".

**A failure is not a wrong answer.** Gateway timeouts and placeholder strings are relay problems, not model answers. Let failures return fast during the batch run and fill the gaps separately; the main table counts only answers actually received:

```bash
# batch: fail fast, do not burn three retries here
python run.py run ... --timeout 120 --max-retries 1 --max-tokens 2048

# backfill: re-ask only the failed items, round after round until zero; each
# entry carries meta["retried"] and the first error
python scripts/retry_unparseable.py \
  --preds outputs/l1_x.jsonl --items data/processed/l1/hit_balanced.jsonl \
  --out outputs/l1_x_filled.jsonl --rounds 4 --max-tokens 2048 --timeout 240
```

Report the first-pass failure count, the failure types and the number of backfill rounds in the results table — an item that needed a backfill cannot be mixed in with one that succeeded first time.

**Reproducibility.** Archive the model version string, the call date, the prompt template (`src/discourse_reliability/prompts.py`) and the raw outputs alongside the predictions. APIs drift; a paper has to state the snapshot its results belong to.

```bash
# freeze a baseline: sha256 of data and prediction files + prompt version + git commit
python scripts/freeze_baseline.py --out outputs/v0/manifest.json

# aggregate several runs into one table
python scripts/summarize_runs.py --gold data/processed/l1/hit_balanced.jsonl \
  --preds "outputs/l1_hit_balanced_*_zs.jsonl" --level l1 \
  --out outputs/l1_model_table.json
```

---

## 4. Human annotation workflow

```bash
# two annotators fill one sheet each (column annotator_sense_l3)
python run.py agreement --a sheet_a.csv --b sheet_b.csv \
  --items items.jsonl --out-dir data/interim/r1
# -> agreement.json (κ at L1/L2/L3) + adjudication.csv (disagreements only)

# a third person fills the adjudicated column of adjudication.csv
python run.py merge-adjudication --items items.jsonl \
  --adjudication adjudication.csv --out gold.jsonl
```

---

## 5. Modules and experiment blocks

| Block | Content | Code |
|---|---|---|
| B1 gold standard and human ceiling | sampling, annotation, IAA, adjudication | `data/sampling.py`, `annotation.py` |
| B2 reliability matrix | model adapters, three prompt regimes, stratified metrics, bootstrap CIs | `models/`, `prompts.py`, `runner.py`, `metrics/` |
| B3 shortcut control | connective deletion / synonym substitution / reordering; lexeme prior | `counterfactual.py`, `baselines/lexeme_prior.py` |
| B4 triage protocol | confidence, cross-model consensus, hybrid scorer, cost–quality curve | `triage.py` |
| B5 bias and validity | disagreement items, label distribution, error typology | `metrics/evaluate.py` + human codebooks |

![alt text](image.png)

Design choices:

- **The bootstrap resamples items, not API calls** — repeated samples of the same item are not independent.
- **The lexeme prior is estimated out of fold**, so a test item's own label never enters its own prior.
- **The hybrid triage scorer uses 5-fold cross-fitting**, so no item is both trained on and scored.
- **Sense labels are normalised at all three levels** (first letter upper, rest lower); otherwise DiMLex's `Negative-Condition` and `Negative-condition` become two classes.
- **Items with no usable transformation skip the counterfactual** rather than producing an empty string or an ungrammatical sentence (DiMLex single-sentence exemplars are the obvious case).

### 5.1 The L1 main experiment block (current line)

L1 (four top-level classes) is the only granularity directly comparable with TED-CDB. The experiment plan is in [`../refine-logs/L1_EXPERIMENT_PLAN.md`](../refine-logs/L1_EXPERIMENT_PLAN.md).

```bash
# 0) materialise every block (folds are built at the document level, so no
#    document straddles a split)
python scripts/build_l1_blocks.py \
  --hit data/processed/hitcdtb.jsonl \
  --ted data/processed/tedcdb.jsonl \
  --out-dir data/processed/l1

# 1) L1 prompting and inference (a four-class task in its own right, not the L3
#    template truncated)
python run.py run --input data/processed/l1/hit_balanced.jsonl \
  --out outputs/l1_model.jsonl --model mock --level l1 --samples 3

# 1b) real models: few-shot exemplars must be drawn per fold, otherwise items
#     from the same document in the evaluation file leak in
python scripts/run_l1_cv.py --eval data/processed/l1/hit_balanced.jsonl \
  --train-template data/processed/l1/hit_fold{k}_train.jsonl \
  --out outputs/l1_fs.jsonl --model deepseek:deepseek-chat \
  --regime few_shot --k-shots 4 --level l1

# 1c) reasoning models need an explicit budget: the default 256 tokens
#     truncates the JSON into something unparseable
python run.py run --input data/processed/l1/hit_balanced.jsonl \
  --out outputs/l1_reasoner.jsonl --model deepseek:deepseek-reasoner \
  --level l1 --max-tokens 1024

# 2) L1 metrics / prior baseline / triage
python run.py evaluate --gold data/processed/l1/hit_balanced.jsonl \
  --preds outputs/l1_model.jsonl --out outputs/l1_eval.json --levels l1
python run.py prior --input data/processed/l1/hit_balanced.jsonl \
  --out outputs/l1_prior.jsonl --level l1
#    triage needs the predictions of several models in one file: --base-model
#    picks the model being scored, the others only contribute the consensus
#    signal and are no longer filtered out
python run.py triage --gold data/processed/l1/hit_balanced.jsonl \
  --preds outputs/l1_pool.jsonl --out outputs/l1_triage.json --level l1 \
  --base-model deepseek:deepseek-chat

# 3) are two runs really different: paired Δ + bootstrap CI + McNemar
python scripts/compare_runs.py --gold data/processed/l1/hit_balanced.jsonl \
  --base outputs/l1_zs.jsonl --alt outputs/l1_fs.jsonl --out outputs/l1_compare.json
```

Note that `--model` is an alias of `--base-model` in `triage`. Earlier versions filtered the prediction file down to a single model first, which left the consensus policy with an empty pool; the filter now applies only to the model being scored, and the consensus pool keeps every model.

| Block | File | Notes |
|---|---|---|
| B1 human ceiling | `l1/b1_human.jsonl` | 600 items, class quotas plus proportional fill |
| B2 main matrix (balanced) | `l1/hit_balanced.jsonl` | 400 = 4×100, so 72% EXPANSION cannot dominate |
| B2 main matrix (distributed) | `l1/hit_distributed.jsonl` | 406, the corpus's real distribution |
| B2 cross-validation | `l1/hit_fold{k}_{train,test}.jsonl` | 5 folds, split by document |
| B3 cross-genre | `l1/ted_balanced.jsonl` + `l1/ted_fold{k}_train.jsonl` | HIT↔TED |
| B4 shortcuts | `l1/hit_fold{k}_test.jsonl` (explicit items) | connective deletion / synonym substitution |
| B5 triage | as above, with `--level l1` | five policies plus an oracle |

Key convention: **`--level l1` switches to a separate four-class prompt template**. Truncating the L3 template to L1 is not the same task and makes the output harder to parse. Folds, splits and class counts are written to `data/processed/l1/manifest.json`.

### 5.2 The annotation site (B1 double annotation)

```bash
# 1) build the blinded packet: sense_l1/l2/l3 and the HIT relation code are
#    dropped here, so the page cannot leak them
python scripts/build_annotation_site.py \
  --packet data/processed/l1/b1_packet.jsonl --out annotation/items.js

# 2a) open directly (works from file://; progress lives in localStorage, export
#     a CSV when done) — double-click annotation/index.html

# 2b) or run the server so annotators work in a browser and submissions land on
#     disk
python annotation/serve.py --port 8765
#     annotators open http://<host-ip>:8765/?a=A

# 3) agreement (the page only collects L1, so name the column)
python run.py agreement --a data/interim/b1_round1/submissions/A_latest.csv \
  --b data/interim/b1_round1/submissions/B_latest.csv \
  --items data/interim/b1_round1/items.jsonl --out-dir data/interim/b1_round1 \
  --column annotator_sense_l1 --n-boot 1000
```

| File | Purpose |
|---|---|
| `annotation/index.html` | single-page app: argument pairs, four class buttons, keys 1–4 / U / ←→ |
| `annotation/guideline.md` | the full annotation guideline (decision procedure, boundaries, worked examples, self-check list) |
| `annotation/serve.py` | optional server; `POST /submit` writes into `submissions/` |
| `scripts/smoke_annotation_app.js` | browserless smoke test: `node scripts/smoke_annotation_app.js` |

The exported CSV has the same columns as `prep-annotation` but fills only `annotator_sense_l1`. `agreement` reads the L3 column by default and falls back to L1 with a warning. `serve.py` has no authentication — `?a=` is a self-declared identity — so use it only on a trusted network.

---

## 6. Known limitations

1. **No CDTB.** The main data are TED-CDB; the column mapping in `data/cdtb.py` has never been validated against a real file, and that is the first thing M0 must do when CDTB arrives.
2. **TED-CDB is spoken.** Its numbers are not directly comparable with CDTB's; every table in the paper has to name the corpus and the genre.
3. **The TED-CDB repository ships no LICENSE file.** The paper says the corpus "will be released for free"; only the 16-talk LREC 2020 subset is explicitly CC BY 4.0. Use that subset if you need redistribution rights.
4. **DiMLex exemplars are biased.** Dictionary editors chose them, so they are easier than natural text. They are fine as smoke tests and few-shot examples, not for estimating model reliability.
5. **The synonym table is small** (18 pairs). Coverage sets the sample size for B3; it needs expanding and manual sense-preservation checks before a real run.
6. **Cross-corpus replication is not built.** SciDTB / GCDT / SCTB need their own loaders and their sense inventories are incompatible with PDTB.
7. **No API cost accounting.** `run` records latency faithfully but does not aggregate token billing.
8. **HIT-CDTB is 50 documents, with a mapped sense inventory.** 3,120 relations are enough to validate the parser and the pipeline, not to carry the main results; the full 525 documents await the data agreement. `HIT_TO_PDTB` is a **lossy mapping of our own**, so L2/L3 comparisons stay exploratory until a human review, and formal conclusions use L1 only.
9. **The HIT-CDTB p1 layer is excluded.** PDTB has no matching granularity and including it would count the same text repeatedly; the cost is that sentence-group-level phenomena are invisible.

---

## 7. Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

24 tests cover: label normalisation, Cohen/Fleiss κ, determinism of stratified sampling, DiMLex parsing, TED-CDB parsing (including that implicit connectives must not leak), HIT-CDTB parsing (CRLF offset normalisation, paired connectives, longest-prefix matching in the mapping table), the L1 protocol (four-class prompts, document-level folds without leakage, triage reading the right level), the multi-model triage pool (equal weight per model when computing consensus entropy, `--base-model` selecting the scored model), double-annotation agreement (UNKNOWN excluded from κ, document-clustered CIs), counterfactual variants, the mock end-to-end run, and the prior baseline.

---

## 8. Repository layout

```
discourse-reliability/
├── run.py                     # entry point (no installation needed)
├── src/discourse_reliability/
│   ├── schema.py              # Item / Prediction / JSONL
│   ├── labels.py              # PDTB sense normalisation and three-level projection
│   ├── prompts.py             # three prompt templates + parsing
│   ├── runner.py              # concurrent calls
│   ├── annotation.py          # two-annotator agreement + adjudication merge
│   ├── counterfactual.py      # counterfactual variants
│   ├── triage.py              # triage scoring and cost–quality curves
│   ├── data/{tedcdb,hitcdtb,dimlex,cdtb,sampling}.py
│   ├── models/{mock,openai_compat,anthropic}.py
│   ├── metrics/{agreement,bootstrap,evaluate}.py
│   └── baselines/lexeme_prior.py
├── scripts/fetch_cdtb.py      # safe extraction and structure check for the licensed archive
├── configs/
├── tests/
└── data/{raw,interim,processed}/   # git-ignored; rebuild from the public releases
```
