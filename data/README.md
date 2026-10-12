# Data card

| Folder | Content | In Git? |
|---|---|---|
| `sample/` | `train_sample.csv`, `test_sample.csv`: 8 rows each, written by the authors for pipeline testing. **Synthetic sentences, not real user posts.** | yes |
| `raw/tweeteval_hate/` | TweetEval *hate* task (SemEval-2019 HatEval, English tweets): train 9,000 / val 1,000 / test 2,970. Created by `python scripts/download_data.py`. | **no** |
| `processed/` | reserved for derived files | **no** |

**Schema** (all CSV files): `id` (string), `text` (string), `label` (1 = harmful/hateful, 0 = benign).

**Ethics.** The raw dataset contains real offensive tweets with user mentions
anonymised as `@user`. It is used only for research evaluation, never redistributed
in this repository, and no personal data is collected by the authors.
