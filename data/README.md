# Data card

| Folder | Content | In Git? |
|---|---|---|
| `sample/` | `train_sample.csv`, `test_sample.csv`: 8 rows each, written by the authors for pipeline testing. **Synthetic sentences, not real user posts.** | yes |
| `raw/tweeteval_hate/` | TweetEval *hate* task (SemEval-2019 HatEval, English tweets): train 9,000 / val 1,000 / test 2,970. Created by `python scripts/download_data.py`. | **no** |
| `raw/NoisyHate/` | NoisyHate (Ye et al., 2023): 1,339 toxic Jigsaw comments, each with a version obfuscated by real people. Created by `git clone https://github.com/YiranYe/NoisyHate data/raw/NoisyHate`. Used in Task 3 as an external check. | **no** |
| `processed/` | Task 3 aggregated tables (`metrics_summary.csv`, `latency_summary.csv`, `hypothesis_tests.csv`) | yes |

**Schema** (all CSV files): `id` (string), `text` (string), `label` (1 = harmful/hateful, 0 = benign).

**Ethics.** The raw dataset contains real offensive tweets with user mentions
anonymised as `@user`. It is used only for research evaluation, never redistributed
in this repository, and no personal data is collected by the authors.
