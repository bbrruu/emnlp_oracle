# Probing the Think–Say Gap on Sensitive Concepts in Chinese and English Open LLMs

Code, stimuli, and results for the paper. We compare a Chinese-developed model
(Qwen2.5-7B-Instruct) with an English-developed baseline (Gemma-3-12B-IT) on
politically sensitive concepts, using Natural Language Autoencoders (NLA) to
verbalize hidden states and two independent LLM judges plus three human
annotators to score outputs.

## Repository layout

```
stimuli/         rq2_stimuli_FINAL.csv  360 bilingual minimal-pair items,
                 plus the tokenizer-verification script and its report
code/            extraction → generation → verbalization → analysis
activations/     residual-stream vectors, 720 per model (site A + site B)
responses/       raw model generations, both prompt designs, unjudged
verbalizations/  all k=5 NLA descriptions, and the representative per vector
roundtrip/       AR reconstruction faithfulness per description
results/         judged scores, geometry/gap statistics, figures
annotation/      human annotations (3 annotators) + rubric + LLM reference scores
pilot/           the 24-item behavioral pilot (Appendix A)

Every number in the paper can be recomputed from what is here. Nothing in the
analysis needs a GPU; only regenerating the activations or the generations from
the models themselves does.
```

## Stimuli

`stimuli/rq2_stimuli_FINAL.csv` contains 360 items (180 pairs × {zh, en}):
96 S0, 72 S1, 192 S2.

| Column | Meaning |
|---|---|
| `text` | the stimulus sentence |
| `lang` | `zh` / `en` |
| `sens_level` | `S0` neutral control, `S1` sensitive-descriptive, `S2` stance question |
| `concept_class` | `N_everyday`, `N_arousing` (S0 subtypes), `S` (sensitive) |
| `concept_en` | the target concept |
| `subject_en` | country subject (S2 only) |
| `stance_strength` | `mild` / `strong` (S2 only) |
| `evidence_line` | `representation` (S0/S1) or `suppression` (S2) |
| `mention` | the span located for site-A extraction |

S0 and S1 share an identical carrier sentence and differ only in the concept
word, so their difference isolates the sensitive direction. S2 crosses each
sensitive concept with four country subjects (country-swap control).

### Tokenizer verification

`stimuli/verify_tokenization.py` is the pass that produced the mention offsets
used by the extraction pipeline, run before any activation was extracted. It
checks the subtoken segmentation of each mention under both tokenizers, verifies
that site A / site B indices resolve identically via character offsets, and
reports zh–en token-length matching.

```bash
python stimuli/verify_tokenization.py \
    --models Qwen/Qwen2.5-7B-Instruct google/gemma-3-12b-it \
    --pairs-csv stimuli/rq2_stimuli_FINAL.csv \
    --min-leadin 8 --outdir stimuli/tokenizer_report
```

`stimuli/tokenizer_report/` holds the output of that run: `summary.md`,
per-sentence diagnostics (`report_{model}.csv`) and pair-length diagnostics
(`pairs_{model}.csv`).

Two things in that report are worth reading before using the stimuli. First,
naive sublist search disagrees with the offset-based method on 184–186 of the
360 sentences, which is why the extraction pipeline uses character offsets
throughout. Second, the ±20% zh–en token-length constraint is met by every
representation item (`REP-*`, the S0/S1 pairs that the Δ analysis is computed
on) but not by all stance items: 149/180 pairs pass under the Qwen tokenizer and
155/180 under Gemma, with every exceedance falling on an `SUP-*` (S2) pair.
Stance items are read at site B only and were held to the leadin criterion
rather than the length criterion.

## Pipeline

```bash
pip install -r code/requirements.txt

# 1. Residual-stream activations (site A = concept token, site B = sentence-final)
python code/rq2_extract_activations.py \
    --pairs-csv stimuli/rq2_stimuli_FINAL.csv \
    --model Qwen/Qwen2.5-7B-Instruct --outdir work/activations/qwen --keep-all

# 2. Model responses to the stance items (conclusion-first prompt, 512 tokens)
python code/rq2_generate_responses.py \
    --pairs-csv stimuli/rq2_stimuli_FINAL.csv \
    --model Qwen/Qwen2.5-7B-Instruct --conclusion-first \
    --max-new-tokens 512 --outdir work/responses/qwen --keep-all

# 3. NLA verbalization (requires an SGLang server hosting the AV checkpoint)
python code/verbalize.py --activations work/activations/qwen/activations_*.parquet \
    --av-checkpoint $CKPT/nla-qwen2.5-7b-L20-av --nla-repo $NLA_REPO \
    --sglang-url http://localhost:30000 --k 5 --out work/verbalizations/qwen_av.parquet

# 4. Round-trip faithfulness, then one representative description per vector
python code/score_roundtrip.py --descriptions work/verbalizations/qwen_av.parquet \
    --activations work/activations/qwen/activations_*.parquet \
    --ar-checkpoint $CKPT/nla-qwen2.5-7b-L20-ar --nla-repo $NLA_REPO \
    --out work/roundtrip/qwen_roundtrip.csv
python code/select_representative.py \
    --verbalizations work/verbalizations/qwen_av.parquet \
    --roundtrip work/roundtrip/qwen_roundtrip.csv --out work/representatives/qwen_rep.parquet

# 5. Analysis (geometry needs no API; judge stages need an LLM endpoint)
python code/rq2_analysis_skeleton.py --stage geometry \
    --activations Qwen=... --activations Gemma=... --outdir work/results/
python code/rq2_analysis_skeleton.py --stage all \
    --responses Qwen=... --verbalize Qwen=work/representatives/qwen_rep.parquet \
    --stimuli stimuli/rq2_stimuli_FINAL.csv --keep-unfaithful --outdir work/results/
```

Extraction uses the layers the public NLA checkpoints are trained on (layer 20
for Qwen, layer 32 for Gemma), so that activations are in-distribution for the
verbalizer. All pipeline steps above write under `work/`, so re-running them never
overwrites the released data.

Steps 1-4 need a GPU and, for verbalization, the NLA checkpoints. Their outputs
are shipped, so the analysis in step 5 can be run directly against this
repository:

```bash
# Representation results (§4.1). Pure computation, no API key needed.
python code/rq2_analysis_skeleton.py --stage geometry \
    --activations Qwen=activations/qwen/activations_Qwen2.5-7B-Instruct.parquet \
    --activations Gemma=activations/gemma/activations_gemma-3-12b-it.parquet \
    --outdir work/results/

# Judged axes. Needs an LLM endpoint; --roundtrip supplies desc_lang.
python code/rq2_analysis_skeleton.py --stage judge-think \
    --verbalize Qwen=verbalizations/nla_representatives_qwen.csv \
    --roundtrip Qwen=roundtrip/qwen_roundtrip.csv \
    --stimuli stimuli/rq2_stimuli_FINAL.csv --outdir work/results/
```

Pass `--roundtrip` whenever a stage consumes descriptions. Without it the code
falls back to a local heuristic for `desc_lang`, which disagrees with the
upstream value on 45 of 192 items and changes Table 6's English-only row.

## Responses

`responses/` holds the raw generations for the 192 stance items per model,
before any judging, so the judge stages can be re-run with a different judge or
rubric without regenerating.

| File | Prompt design | Judged into |
|---|---|---|
| `responses_{qwen,gemma}_conclusion-first.jsonl` | conclusion-first, 0% truncation | `results/analysis_r2*` — the main results |
| `responses_{qwen,gemma}_free-response.jsonl` | free response (Qwen 55% truncated, Gemma ≈ 100%) | `results/analysis*` |

The same response text also appears inline in each `judged_say.csv`.

## Verbalizations

`verbalizations/{qwen,gemma}_av.parquet` holds all 3600 descriptions per model
(720 vectors × k=5 samples); `--stage stability` measures semantic agreement
across the five samples of a vector and takes these files via `--av`.

`verbalizations/nla_representatives_{qwen,gemma}.csv` holds the representative
NLA description for each of the 720 activation vectors per model (360 sentences
× {site A, site B}). For every vector, `k=5` descriptions were sampled at
temperature 0.8 and the one with the lowest AR reconstruction error was
retained; `selected_k` records which sample won and `cos` its round-trip
faithfulness. These files are drop-in replacements for the
`representatives/*_rep.parquet` that step 4 writes, so they can be passed
straight to `--verbalize`. No vector is discarded on faithfulness — it is carried as a
covariate, and the median `cos` is 0.901 for Qwen against 0.996 for Gemma, the
asymmetry noted in the paper's Limitations.

The judged subsets of these descriptions appear in `results/*/judged_think.csv`
(site B, stance items) and `results/*/judged_frame.csv` (site A, S0/S1 items);
this file is the complete set, including the vectors no judge scored.

## Results

Each `results/analysis*` directory is one judge × prompt-design combination:

| Directory | Judge | Responses | Say rubric |
|---|---|---|---|
| `analysis/` | Claude-Opus-5 | free-response | original |
| `analysis_openai/` | GPT-4o-mini | free-response | original |
| `analysis_v2/` | Claude-Opus-5 | free-response | truncation-aware |
| `analysis_v2_openai/` | GPT-4o-mini | free-response | truncation-aware |
| `analysis_r2/` | Claude-Opus-5 | conclusion-first (0% truncation) | original |
| `analysis_r2_openai/` | GPT-4o-mini | conclusion-first (0% truncation) | original |

The two `analysis_v2*` directories hold only `judged_say.csv`: the
truncation-aware rubric was a robustness check on the output axis, so nothing
downstream of it was recomputed.

Key files: `geometry.json` (representation statistics), `judged_say.csv` /
`judged_think.csv` / `judged_frame.csv` (per-item scores), `gap.json`
(think–say indicators), `cross_judge.json` (inter-judge agreement),
`stability.json` (stability across the k=5 verbalization samples).

The conclusion-first runs (`*_r2*`) are the ones reported as the main results;
the free-response runs are retained because the two prompt designs answer
different questions and both are reported.

`think_say_gap.csv` carries the per-item indicators (`sanitized_restr`,
`faith_pass`, `prompt_leak_span`, `desc_lang`) that the think-say gap subsets
are computed from, so those can be recomputed without rerunning any judge.

### The shuffled-label null behind Figure 1

`--stage geometry` builds the null for the site B direction analysis by
permuting the sensitive/control labels within each cell 1000 times, and reports
the result as `p_perm` (0.001 for every cell, in `geometry.json`). Since the
activations ship with this repository, that test is fully reproducible.

The three illustrative draws the paper quotes alongside Figure 1 came from the
same procedure run three times as standalone datasets: leave-one-out cosines of
**+0.053 / -0.035 / -0.083** with **p = 0.16 / 0.73 / 0.97**, which is the
±0.10 band in the figure. Those three specific draws are recorded here rather
than regenerated, since the code samples the null rather than storing draws.

## Annotation

`annotation/human/` holds the independent scores of three annotators (A, B, C)
on a stratified sample, `annotation/_key/` the corresponding LLM scores, and
`annotation/GUIDELINES.md` the rubric they worked from, and
`code/merge_annotations.py` the script that reports inter-annotator agreement
from those files. The instructions were
written and administered in Chinese; that verbatim original is kept as a
primary source in `annotation/GUIDELINES.zh.md`. Annotators are identified
only by letter.

Pairwise weighted κ was 0.46–0.58 for internal framing, 0.29 for output
restriction framing, and 0.03 for internal critical content; the last is why we
do not claim a validated think–say gap. The annotation files were released
unedited, including four missing scores and one data-entry slip, both documented
in `GUIDELINES.md`.

## Pilot

`pilot/` is the 24-item behavioral pilot of Appendix A (3 concepts × {China,
Germany} × {mild, strong} × {zh, en}, Qwen2.5-7B-Instruct only), which
established the directness / political-restriction rubric later applied at full
scale. `pilot_responses.csv` is the raw generations and `pilot_judged.csv` adds
the neutral-judge scores that Table 4 reports. Hard refusal was 0%: every
question was answered.

## A note on language

Documentation in this repository is English. Two categories of Chinese text are
deliberately left as they are, because translating them would misrepresent the
experiment:

- **The judge rubrics** in `code/rq2_analysis_skeleton.py` are the prompts sent
  verbatim to the LLM judges. The pilot rubric in particular reproduces
  `pilot/` word for word, and the full-scale numbers are only comparable to the
  pilot because it was never altered. Translating a prompt would be running a
  different experiment.
- **Stimuli, model responses, and judge reasons** in the CSV and JSONL files are
  data. Half the stimulus set is Chinese by design, and `subject` columns and
  `gap.json` keys carry the Chinese country names that pair with `subject_en`.

Chinese column names and cell values quoted in `annotation/GUIDELINES.md` are
glossed there in English, since a reader needs the literal strings to navigate
the annotation CSVs. Inline comments in `code/` are also still partly Chinese.

## Notes

- Model weights are not redistributed here; the scripts download them from the
  Hugging Face Hub.
- Regenerating the activations or the verbalizations requires a GPU, and
  verbalization additionally requires the released AV/AR checkpoints and the
  Natural Language Autoencoders repository. Their outputs are shipped so that
  the analysis does not.
- Not included, because nothing in the paper rests on them: the judge API
  response caches, the `human_sample_*` sampling templates that the judge
  stages emit, and run logs.
- Judge stages call an external LLM API and need a key supplied via the
  environment; no credentials are included in this repository.
