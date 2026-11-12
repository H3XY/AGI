# Dataset checksums

SHA-256 of the data files used, as downloaded (data/ is not committed; the loaders in core/ re-download them). Upstream sources can change over time (NVD rescoring), so a mismatch means the inputs differ from the paper's.

| File | Source | SHA-256 |
|---|---|---|
| rotten_tomatoes_train.parquet | Hugging Face cornell-movie-review-data/rotten_tomatoes | `f899e3cb8124a12b7d82c30ba5cd35d27eb6575d1b10d65f731a60348454a959` |
| rotten_tomatoes_test.parquet | Hugging Face cornell-movie-review-data/rotten_tomatoes | `5427a2b59d1b9bed1ea5cc3f963843bd13ea7443f32d27e74a957a3d181cc545` |
| rotten_tomatoes_validation.parquet | Hugging Face cornell-movie-review-data/rotten_tomatoes | `3ea894e394cd24413b781790683924a2598507d146da9b4ad0a2a01830c77b00` |
| nvd_cves_2024.parquet | NVD API 2.0, published 2024-01-01 to 2024-04-29, downloaded 2026-11-08 | `aaddea6b189b6daf038eb9b2e85f7ef8c91da3440500b4eb38fe69211fcb595c` |
| nvd_cves_2024_holdout.parquet | NVD API 2.0, published 2024-05-01 to 2024-08-28, downloaded 2026-11-09 (confirmatory set) | `83b152c60ea59f0b857be36ab09457b8a6b332d6a52a264cca8e011f8f1fd2f6` |
