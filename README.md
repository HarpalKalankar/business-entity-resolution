# Business Entity Resolution (ML Challenge 2026)

Our team's pipeline for the Unstop ML Challenge (72-hour hackathon): for every Source 1
business, find all matching records in Source 2 and Source 3. The score is macro F0.5,
which weighs precision 2x.

**Status (26 Sep 2026)**

| Run | Held-out macro F0.5 (train) | Leaderboard |
| --- | --- | --- |
| Run 1 | 0.9792 | **0.967** |
| Run 2 (code in this repo) | pending | pending |

Leader at the time of writing: 0.9884. See [`docs/methodology.md`](docs/methodology.md)
for the approach, the loss breakdown and the next steps.

## Pipeline at a glance

```
normalize (Stage 0) -> blocking + pre-ranker (Stage 1) -> LightGBM pair matcher (Stage 2)
                    -> decision layer: one-owner + expected F0.5 (Stage 3) -> output/*.tsv
```

## Setup

1. **Clone** this repo.
2. **Add the challenge files** (not in git, too large or not ours) from the organizers'
   `student_resource` zip into the repo root:

   ```
   business-entity-resolution/
   ├── dataset/          <- train/ and test/ folders from student_resource
   ├── utils/            <- validate_submission.py from student_resource
   ├── code/business_entity_resolution/   (this repo)
   ├── docs/
   └── .vscode/
   ```

3. **Python 3.10–3.12** (any environment: venv, conda):

   ```bash
   cd code/business_entity_resolution
   python -m venv .venv
   .venv\Scripts\activate          # Windows   (Linux/Mac: source .venv/bin/activate)
   pip install -r requirements.txt
   ```

   If a pinned version will not install on your OS, drop the `==version` part for that line.

## Run

From `code/business_entity_resolution`:

```bash
# quick check on a 2% sample (about 15-30 min): writes output/*_s2.tsv
python src/run_pipeline.py --sample-pct 2

# full data (about 9 h on a 16 GB laptop): writes output/candidate_pairs.tsv,
# output/matching_results.tsv and runs the official validator
python src/run_pipeline.py --train-frac 0.3

# resume from a stage (finished test shards are skipped)
python src/run_pipeline.py --from stage2 --train-frac 0.3
```

In VS Code, open the repo root. Pick your Python interpreter, then use the Run and Debug
configurations in `.vscode/launch.json`: `FULL PIPELINE`, `Full pipeline on 2% sample`,
and one per stage.

- Everything the run creates goes to `work/` (resources, parquet files, models) and
  `output/`. Both are git-ignored.
- Upload `output/matching_results.tsv` to the Portal only after the validator prints
  **PASS**. Do not open or re-save it in Excel, which truncates rows beyond 1,048,576.
- The Portal allows 5 submissions a day, so agree in the team before uploading.

Detailed commands for each stage, the features and the outputs are in
[`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md).

## Rules we must keep

- **No external data** (APIs, databases, geocoding). Transliteration tables, gazetteer and
  vocabulary are learned from the provided files only.
- **Licences:** every dependency is MIT/BSD/Apache/ISC. Any model we add must be MIT or
  Apache 2.0 and at most 8B parameters.
- **Country is an open set:** never hard-code or filter to {US, India}. France appears only
  in test.
- `candidate_pairs.tsv` is part of the final ranking; a smaller candidate set per entity ranks higher.

## Working together

- Work on a branch per idea (`feat/india-blocking`, `feat/brand-names`), then open a PR.
- Put the **held-out macro F0.5 from the Stage 3 "chosen" line** in every PR description,
  so changes are compared on the same number.
- Never commit `dataset/`, `work/`, `output/`, or credentials.
