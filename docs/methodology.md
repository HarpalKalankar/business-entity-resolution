# Methodology, results and next steps

## Data facts that drive the design

| Fact | Value | Consequence |
| --- | --- | --- |
| Train S1 / S2 / S3 | 2.21M / 5.03M / 5.29M | Block per country and source, sparse maths, shard the test set |
| Test S1 / S2 / S3 | 1.73M / 4.89M / 5.08M | 259k French S1 entities with no labels |
| Singletons (S1 with no match) | 5.6% | An empty list scores 1.0, so the singleton decision matters |
| Matches per S1 | mean 3.46 | Most entities need several matches |
| S2/S3 records linked to 2+ S1 | **0** | One-owner rule: each record goes to at most one S1 |
| Records with non-Latin names | 7% (9 Indic scripts) | Learned transliteration |
| Hard negatives | same name, different house number / city | Distinguishing tokens and the house number matter |

## Stage 0: normalization (`stage0_normalize.py`)

- **Learned transliteration** (`translit_dict.py`): native-script tokens of S2/S3 records
  are aligned with their S1 twin in train. This gives 1,347 name tokens and 16 native state
  names; `anyascii` is the fallback for unseen tokens.
- **Names**: homoglyph repair confirmed by the vocabulary (`8ig` -> `big`), ID and PIN
  noise removed, aliases (`dba`, `formerly known as`), domains segmented
  (`energyvrtextile.com`), legal forms mapped to canonical tags (including fuzzy typos
  like `Prfivete Limited`).
- **Addresses**: comma parts are classified (postcode / state / city / street / landmark /
  unit), not read by position. Abbreviation maps use a generic default, with a French
  override triggered by French street words. Ordinal words (`fourteenth` -> `14th`) and
  India `c/o` handling are included.
- **Gazetteer** (`gazetteer.py`): cities and regions mined from S1 (train + test).

## Stage 1: blocking + pre-ranker

- ~30 hashed features per record: name tokens, bigrams and phonetic keys, plus
  **name-token x street / city / state / house conjunctions**. The dataset reuses a small
  name vocabulary, so single tokens are too common to block on. There are also
  address-only keys and initials keys for acronyms.
- Features above a document-frequency cap are dropped. Score = sum of the IDF of shared
  features (`sparse_dot_topn`). Forward top-30 per S1 plus reverse top-3 S1 per record.
- LightGBM pre-ranker on 18 cheap features keeps the top **K** candidates per S1:
  K = 25 in run 1, K = 15 in run 2 (smaller candidate sets rank higher).
- Test is processed in 8 hash shards of S1. `test_global.py` recomputes every
  record-side quantity across all shards, so sharded output = unsharded output.

## Stage 2: pair matcher (`pair_features.py`, `stage2_matcher.py`)

- 70+ features, with **no country feature** (France is unseen):
  - Name: fuzzy scores, IDF-weighted soft overlap, IDF of *unmatched* tokens, phonetic,
    legal form, aliases.
  - Address: house number exact / suffix / distance, street, city, state, unit, whole address.
  - Commonness: how many records share this name.
  - Competition: rank, gap and margin among an S1's candidates and among a record's
    competing S1 entities.
  - Run 2 adds brand-name share (`b_oov`), `addr_eq` and **sibling features**
    (similarity to the S1's other confident candidates).
- LightGBM, grouped 5-fold by S1, 2 passes. Pass 2 adds competition computed from the
  pass-1 out-of-fold probabilities.

## Stage 3: decision layer (`stage3_decide.py`)

1. Isotonic calibration on out-of-fold probabilities.
2. One owner: each record goes to its best S1 only if it beats the runner-up by a margin.
3. Per S1, choose the top-k that maximises expected F0.5; predict empty when
   alpha x prod(1 - p) is higher. A threshold rule competes, and settings are tuned on half
   the entities and reported on the other half.

## Results

**Run 1** (train-frac 0.3, 662k held-out entities, K = 25)

| Metric | Value |
| --- | --- |
| Blocking pair recall | 98.35% (India 97.5–97.9%, US 98.7–98.8%) |
| Held-out macro F0.5 | 0.9792 |
| Pair precision / recall | 99.46% / 95.12% |
| Leaderboard | **0.967** |

**Where run 1 loses its 0.0208** (share of macro F0.5):

| Error type | Loss |
| --- | --- |
| True match in candidates but model too strict | 0.0068 |
| True match lost in blocking | 0.0044 |
| Predicted empty but the entity has matches | 0.0042 |
| One false match on a non-singleton | 0.0042 |
| False match on a singleton | 0.0013 |

The loss is mostly **recall**. Typical low-probability true matches:

- Invented DBA/brand names at the S1's exact address (`Korectodelta`, `Zetahalo`).
- No-address records with an exact name (`Peak Obic`).
- Typos combined with house-number drift.

The test score (0.967) is below held-out (0.979). Run 1 used shard-local competition,
with 20,193 cross-shard conflicts; this is fixed for run 2.

## Next ideas (ranked)

1. Check run 2 (sibling and brand features, global competition, more trees) on held-out
   F0.5 against 0.9792.
2. **Blocking recall for India** (97.5–97.9%): extra keys for no-address records
   (phonetic name key x state) and brand names (address-only keys with looser caps).
3. Tune K (10–15) against the candidate-size ranking rule.
4. Optional cross-encoder re-ranker (`xlm-roberta-base`, MIT) on borderline pairs, which
   helps transliterated and French names. The laptop RTX 4060 is enough.
