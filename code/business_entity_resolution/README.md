# Business Entity Resolution: pipeline

Match every Source 1 business to its records in Source 2 / Source 3.
Design: normalize → block → LightGBM pair model → expected-F0.5 set selection.

## Reproduce end to end (one command)

```bash
cd code/business_entity_resolution
python src/run_pipeline.py --train-frac 0.3        # full data -> ../../output/*.tsv + validator
python src/run_pipeline.py --sample-pct 2          # dev run on a 2% sample (*_s2.tsv)
python src/run_pipeline.py --from stage2           # resume from a stage
```

Training uses a `--train-frac` share of train S1 entities (salted hash, independent of any
sample), each blocked against the FULL train S2/S3 pools so name collisions are realistic.

## Setup

```bash
# from code/business_entity_resolution, in any Python 3.10-3.12 environment
pip install -r requirements.txt
```

## Layout

```
student_resource/
├── dataset/                     (given)
├── work/                        (created) resources + normalized parquet
└── code/business_entity_resolution/
    ├── requirements.txt
    └── src/
        ├── lexicon.py           abbreviations, legal forms, state codes (linguistic only)
        ├── text_utils.py        unicode, homoglyphs, transliteration fallback, edit distance
        ├── translit_dict.py     LEARNS native-script -> Latin tables from train ground truth
        ├── gazetteer.py         MINES cities / regions / name vocab from Source 1
        ├── name_norm.py         business-name normalizer
        ├── address_norm.py      address normalizer + part classifier
        ├── stage0_normalize.py  Stage 0 entry point
        ├── stage0_report.py     quality report for Stage 0
        ├── block_features.py    hashed blocking signatures (name x address conjunctions)
        ├── stage1_blocking.py   Stage 1 candidate generation (forward top-K + reverse top-k)
        ├── stage1_eval.py       blocking recall / oracle F0.5 on train
        ├── stage1_prerank.py    Stage 1b LightGBM pre-ranker -> candidate_pairs.tsv
        ├── pair_features.py     Stage 2 pair features (62 + competition features)
        ├── stage2_matcher.py    Stage 2 LightGBM matcher (2 passes, grouped 5-fold)
        ├── stage3_decide.py     Stage 3 decision layer -> matching_results.tsv
        ├── sampling.py          deterministic, salted entity subsets
        └── run_pipeline.py      end-to-end runner + official validator
```

## Stage 0: normalization

Run from `code/business_entity_resolution`:

```bash
# dev subset: 2% of S1 entities + all their train matches + 2% of other records (~10 min)
python src/stage0_normalize.py --data ../../dataset --work ../../work --sample-pct 2
python src/stage0_report.py --work ../../work --suffix _s2

# full data (all 6 files, about 22M rows; roughly 1-2 h depending on cores)
python src/stage0_normalize.py --data ../../dataset --work ../../work
```

Useful flags: `--workers N`, `--only train_source1 test_source2`, `--overwrite`, `--rebuild`
(re-learn resources), `--resources-only`. Finished files are skipped, so an interrupted
run resumes where it stopped.

### Outputs (`work/`)

| File | Content |
| --- | --- |
| `resources/translit.json` | Learned tables: 1,347 name tokens across 9 Indic scripts, 16 state names |
| `resources/gazetteer.json` | Cities per country, mined regions (e.g. France), name-token vocab |
| `norm/{split}_source{k}.parquet` | One row per record (columns below) |

| Column | Meaning |
| --- | --- |
| `name_clean` | Normalized full name, legal words as canonical tags (`pvt`, `ltd`, `llc`, `sarl` …) |
| `name_core` | Name without legal-form words |
| `name_key` | Sorted core tokens minus generic words/honorifics, used as the blocking key |
| `name_alt` | Alias cores from `dba` / `formerly known as` |
| `legal` | Canonical legal tags |
| `name_translit` | 0 Latin · 1 fallback transliteration · 2 learned dictionary |
| `name_is_domain` | Name was a web domain (segmented into words) |
| `addr_clean` | Normalized address text (street, locality, city, state) |
| `house_num`, `all_nums` | First house number / all numbers (ordinals excluded) |
| `street`, `locality`, `city`, `state`, `postcode`, `unit`, `landmark` | Parsed parts |
| `addr_missing`, `addr_translit` | Flags |

### Stage 0 quality (2% train sample, 152k true pairs)

| Agreement on true pairs | Raw | Normalized |
| --- | --- | --- |
| Name exact / `name_key` equal | 10.8% | 74.9% |
| Address exact / state equal | 7.3% | 99.2% |
| House number equal | | 84.9% |
| City equal | | 88.8% |

## Stage 1: blocking + pre-ranker

```bash
# dev sample (after stage 0 with --sample-pct 2)
python src/stage1_blocking.py --work ../../work --split train --suffix _s2
python src/stage1_eval.py     --work ../../work --suffix _s2
python src/stage1_prerank.py  --work ../../work --mode train --suffix _s2
python src/stage1_blocking.py --work ../../work --split test  --suffix _s2
python src/stage1_prerank.py  --work ../../work --mode apply --split test --suffix _s2 --k 25

# full data: same commands without --suffix
```

**How it works**

1. Each record gets ~30 hashed features: name tokens, bigrams and phonetic keys, plus
   conjunctions of a name token with street / city / state / house number, address-only keys
   (whole address, street bigram x city, house x locality, unit), and initials x house/city
   for acronyms (`MC` = Mohindra & Co).
2. Features seen in more than `--df-cap` target records (2000 at full scale, scaled down in
   sample mode) are dropped; score = sum of IDF of shared features (sparse product, `sparse_dot_topn`).
3. Per country (open set) and per source: forward top-K=30 per S1 plus reverse top-3 S1 per
   S2/S3 record. The reverse search uses the one-owner property.
4. Pre-ranker (LightGBM, 18 cheap features) keeps the top `--k` per S1 across S2+S3. This set is
   what Stage 2 scores and what `output/candidate_pairs.tsv` contains.

**Results on the 2% train sample (152k true pairs)**

| Candidate set | Pairs per S1 | Pair recall | Oracle F0.5 ceiling |
| --- | --- | --- | --- |
| Blocking union | 55.3 | 99.41% | 0.9982 |
| Blocking, forward top-10 + reverse top-1 | 21.0 | 99.15% | 0.9974 |
| Pre-ranker top-10 (held-out 20%) | 9.9 | 99.44%* | 0.9983 |
| Pre-ranker top-25 (default) | 24.2 | 99.45%* | 0.9983 |

*Recall relative to all true pairs of the held-out entities. The sample holds about 2% of
the distractors, so ranks are less crowded than on the full data; that is why the default
is K=25 rather than 10. Confirm on a larger sample (e.g. `--sample-pct 10`) on a bigger machine.

## Stage 2: pair matcher

```bash
python src/stage1_prerank.py  --work ../../work --mode apply --split train --suffix _s2 --k 25
python src/stage2_matcher.py  --work ../../work --suffix _s2 --step features --split train
python src/stage2_matcher.py  --work ../../work --suffix _s2 --step train      # resumable per fold
python src/stage2_matcher.py  --work ../../work --suffix _s2 --step features --split test
python src/stage2_matcher.py  --work ../../work --suffix _s2 --step predict
# full data: no --suffix; add --train-frac 0.3 to the train-split feature step to fit in RAM
```

**Features (no country column, so the model transfers to France)**

| Group | Features |
| --- | --- |
| Name | ratio, token-set, token-sort, partial (name_core); Jaro-Winkler and Levenshtein on name_key; token-set on full name with legal words; key/core/first-token equal; alias best score; IDF-weighted soft overlap both ways; **IDF of unmatched tokens** (what makes a look-alike a different business); phonetic Jaccard; token counts; legal-form agreement; transliteration and domain flags |
| Address | house number equal / suffix (07 vs 407) / log numeric distance / found in the other's numbers; all-numbers Jaccard; street token-set, Jaro-Winkler, Jaccard; city equal and Jaro-Winkler; state, postcode, unit equal; locality; whole-address token-set, ratio, Jaccard; missing-address and landmark flags |
| Commonness | log count of S1 and S2/S3 records sharing the name key; count sharing house + street |
| Context | blocking score, forward/reverse rank, pre-ranker probability, source |
| Competition | for the S1 entity and for the S2/S3 record: rank, gap to best, margin over the runner-up, number of competitors (pass 1 from the pre-ranker, pass 2 from pass-1 out-of-fold probabilities) |

**2% train sample, out-of-fold (44,139 entities; pairs lost in blocking count as misses)**

| Decision (Stage 3 will refine) | Macro F0.5 | Singletons | Entities with matches |
| --- | --- | --- | --- |
| Pass 1, one-owner, threshold 0.5 | 0.9888 | 0.979 | 0.9894 |
| Pass 1, one-owner, threshold 0.7 | **0.9898** | 0.986 | 0.9901 |
| Pass 2, one-owner, threshold 0.7 | 0.9893 | 0.983 | 0.9896 |
| Oracle ceiling (perfect matcher on these candidates) | 0.9983 | | |

Main remaining error: S2/S3 records with an **empty address**, matched on name only (both
false merges and misses). The full data has ~50x more same-name records than the sample,
which is why the full run must train on candidates blocked against the full pools.

## Stage 3: decision layer

```bash
python src/stage3_decide.py --work ../../work --suffix _s2 --mode tune    # on train OOF
python src/stage3_decide.py --work ../../work --suffix _s2 --mode apply   # -> output/matching_results_s2.tsv
```

1. **Calibration:** isotonic regression on out-of-fold train probabilities.
2. **One owner:** each S2/S3 record goes only to its best S1 entity, and only if it beats the
   runner-up by `margin` (ground truth never links one record to two entities).
3. **Set selection per S1 entity.** Two rules are tuned and the better one on the tuning half wins:
   - *expected F0.5*: sort candidates by p; predicting the top k has TP = sum p, FP = k - TP,
     FN = remaining p; pick the k with the highest 1.25TP / (1.25TP + 0.25FN + FP); predict
     nothing when alpha x prod(1 - p) is higher (the chance the entity is a singleton);
   - *threshold*: keep pairs with p >= t.
4. Knobs (`p_min`, `alpha`, `margin`, `t`) are tuned on half the train entities and reported on
   the other, untouched half. Chosen settings are stored in `work/models/stage3.json`.

**2% sample, untouched half (22,003 entities)**

| Rule | Macro F0.5 | Singletons | Entities with matches |
| --- | --- | --- | --- |
| Expected F0.5, pass 1 (chosen: p_min 0.3, alpha 1.0, margin 0.1) | 0.9896 | 0.984 | 0.9899 |
| One-owner + threshold 0.7, pass 1 | 0.9897 | 0.987 | 0.9899 |
| Oracle ceiling | 0.9983 | | |

The two rules tie on the sample because the model's probabilities are nearly all close to 0 or 1.
On the full data, with more look-alikes, the rules may differ, and tuning picks the winner.

## Fair play

No external data or APIs. Transliteration tables, gazetteer and vocab are learned only
from the provided files; `lexicon.py` holds generic linguistic abbreviations.
