# Probing the Think–Say Gap on Sensitive Concepts in Chinese and English Open LLMs

Code, stimuli, and results for the paper. We compare a Chinese-developed model
(Qwen2.5-7B-Instruct) with an English-developed baseline (Gemma-3-12B-IT) on
politically sensitive concepts, using Natural Language Autoencoders (NLA) to
verbalize hidden states and two independent LLM judges plus three human
annotators to score outputs.

## Repository layout

```
stimuli/     rq2_stimuli_FINAL.csv   360 bilingual minimal-pair items
code/        extraction → generation → verbalization → analysis
results/     judged scores, geometry/gap statistics, figures
annotation/  human annotations (3 annotators) + LLM reference scores
```

## Stimuli

`stimuli/rq2_stimuli_FINAL.csv` contains 360 items (180 pairs × {zh, en}).

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

## Pipeline

```bash
pip install -r code/requirements.txt

# 1. Residual-stream activations (site A = concept token, site B = sentence-final)
python code/rq2_extract_activations.py \
    --pairs-csv stimuli/rq2_stimuli_FINAL.csv \
    --model Qwen/Qwen2.5-7B-Instruct --outdir activations/qwen --keep-all

# 2. Model responses to the stance items (conclusion-first prompt, 512 tokens)
python code/rq2_generate_responses.py \
    --pairs-csv stimuli/rq2_stimuli_FINAL.csv \
    --model Qwen/Qwen2.5-7B-Instruct --conclusion-first \
    --max-new-tokens 512 --outdir responses/qwen --keep-all

# 3. NLA verbalization (requires an SGLang server hosting the AV checkpoint)
python code/verbalize.py --activations activations/qwen/activations_*.parquet \
    --av-checkpoint $CKPT/nla-qwen2.5-7b-L20-av --nla-repo $NLA_REPO \
    --sglang-url http://localhost:30000 --k 5 --out verbalizations/qwen_av.parquet

# 4. Round-trip faithfulness, then one representative description per vector
python code/score_roundtrip.py --descriptions verbalizations/qwen_av.parquet \
    --activations activations/qwen/activations_*.parquet \
    --ar-checkpoint $CKPT/nla-qwen2.5-7b-L20-ar --nla-repo $NLA_REPO \
    --out roundtrip/qwen_roundtrip.csv
python code/select_representative.py \
    --verbalizations verbalizations/qwen_av.parquet \
    --roundtrip roundtrip/qwen_roundtrip.csv --out representatives/qwen_rep.parquet

# 5. Analysis (geometry needs no API; judge stages need an LLM endpoint)
python code/rq2_analysis_skeleton.py --stage geometry \
    --activations Qwen=... --activations Gemma=... --outdir results/
python code/rq2_analysis_skeleton.py --stage all \
    --responses Qwen=... --verbalize Qwen=representatives/qwen_rep.parquet \
    --stimuli stimuli/rq2_stimuli_FINAL.csv --keep-unfaithful --outdir results/
```

Extraction uses the layers the public NLA checkpoints are trained on (layer 20
for Qwen, layer 32 for Gemma), so that activations are in-distribution for the
verbalizer. `--stage geometry` is pure computation and reproduces the
representation results without any API access.

## Results

Each `results/analysis*` directory is one judge × prompt-design combination:

| Directory | Judge | Responses |
|---|---|---|
| `analysis/` | Claude-Opus-5 | free-response |
| `analysis_openai/` | GPT-4o-mini | free-response |
| `analysis_r2/` | Claude-Opus-5 | conclusion-first (0% truncation) |
| `analysis_r2_openai/` | GPT-4o-mini | conclusion-first (0% truncation) |

Key files: `geometry.json` (representation statistics), `judged_say.csv` /
`judged_think.csv` / `judged_frame.csv` (per-item scores), `gap.json`
(think–say indicators), `cross_judge.json` (inter-judge agreement),
`stability.json` (stability across the k=5 verbalization samples).

The conclusion-first runs (`*_r2*`) are the ones reported as the main results;
the free-response runs are retained because the two prompt designs answer
different questions and both are reported.

## Annotation

`annotation/human/` holds the independent scores of three annotators (A, B, C)
on a stratified sample, and `annotation/_key/` the corresponding LLM scores.
Annotators are identified only by letter. Pairwise weighted κ was 0.46–0.58 for
internal framing, 0.29 for output restriction framing, and 0.03 for internal
critical content; the last is why we do not claim a validated think–say gap.

## Notes

- Model weights are not redistributed here; the scripts download them from the
  Hugging Face Hub.
- NLA verbalization additionally requires the released AV/AR checkpoints and the
  Natural Language Autoencoders repository.
- Judge stages call an external LLM API and need a key supplied via the
  environment; no credentials are included in this repository.
