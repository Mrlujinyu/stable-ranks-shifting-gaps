# Replication package: Stable Ranks, Shifting Gaps

## Environment and frozen inputs

Python 3.12.14; install `requirements.txt` (NumPy 2.3.5, pandas 3.0.1, SciPy 1.18.1, statsmodels 0.14.6, patsy 1.0.3, matplotlib 3.11.2). The CSV entry points need no parquet files. `environment.json` is the archived environment; `analysis_outputs/environment_current.json` records the earlier reproduction environment. Software prerequisites must already be installed or installed explicitly.

`data/` is a frozen snapshot, not a download of the latest leaderboard. The Test-directory selection is dated 2026-09-19 and includes 24 candidates; `configuration_selection.csv` gives exclusions, while `table2_configurations.csv` gives stable submission identities, the 14 primary mappings, two expanded-only entries, and recorded upstream result hashes. Those result hashes are source provenance fields; package-file hashes are separately listed in `SHA256SUMS.txt`. Original per-submission result files are not bundled. Main data are `main_panel.csv`, `agent_matrix.csv`, and `task_quality_labels.csv`. Pro data are `pro_observed_matrix.csv`, `pro_labels_analysis.csv`, `pro_task_quality_labels.csv`, and `pro_submission_metadata.csv`.

Upstream sources are SWE-bench's public experiments archive and the released Verified human annotations, Agent Psychometrics task outcomes, and Pro Verified v2 correction identities. Exact references and retrieval URLs are in `SOURCE_REFERENCES.bib` (keys `swebench2026experiments`, `openai2024verified`, `ge2026psychometrics`, `zheng2026proverified`). Main submission IDs contain their version labels; Pro source labels/timestamps do not provide uniform harness pins. Use the frozen tables for reproduction, not current online files. Do not treat unavailable execution metadata as known.

## Quick regression and point-estimate checks

From this package root:

```sh
python scripts/run_fast_checks.py
```

This executes inclusive-tail unit tests (including ties and nonfinite inputs), identity checks, stable-ID matrix alignment and reorder checks on the existing 10,000 subset indices, all primary/Pro pair point estimates and paired-test values, observed-composition and influence point checks. It does **not** generate bootstrap or permutation samples or refit a GEE. Reading all existing indices is not a new full inference run. Reports are written under `analysis_outputs/final_checks/`.

## Frozen single-task influence reconstruction

```sh
python scripts/recompute_influence_permutation.py
```

This reconstructs the exact original `scripts/original_research/controls.py` RNG consumption: initialize PCG64 with seed 20260919 once; generate exact matches; interleave 250-row multinomial bootstrap and paired-sign batches for 10,000 draws; then construct metadata matches and continue the same generator through 10,000 more draws. Each sign swaps the fixed pair's flagged/comparison labels. The statistic is mean flagged-minus-comparison **single-task influence** in pp. It is distinct from mean absolute deletion-vector contrast. New arrays include swap signs, permutation statistics, bootstrap statistics, pair identities and paired influence differences. Original permutation arrays were not supplied; these are deterministic reconstructions, not recovered original files. The script checks identities and the full-precision summary/intervals against `matched_summary.json` without modifying it.

## Full inference commands

```sh
python scripts/audit_results.py --resample --gee
python scripts/check_matched_resampling.py
python scripts/comparison_scope_analysis.py
python scripts/additional_diagnostics.py
```

The first command generates 10,000 shared-row bootstraps and 10,000 equal-size subsets for the primary and Pro panels and refits the primary GEE; it is not a smoke test. The matching command reconstructs the fixed-match intervals under the two documented streams. Full-reference matching restarts the seed after loading matches; the matched-union/influence stream includes match construction. These streams are distinct. Scope analysis fixes the original top-five/adjacent sets and uses the same primary draws and full 91-pair BH families. The cached `shared_random_subsets.npz` includes task order and indices. Outputs go to this package's `analysis_outputs/`, never implicitly to a sibling source package. Run full inference in a fresh extraction to retain the distributed output snapshot.

Subtype/omission/Bayesian models and the complete alternative-definition/denominator pipeline are archived results. Their original scripts under `scripts/original_research/` need the original parquet directory layout and external resources and are not invoked by the CSV commands. No command reruns coding agents or evaluates repaired tasks.

## Figure and table generation

```sh
python scripts/make_figures_tables.py
python scripts/make_additional_figure.py
python scripts/make_scope_tables.py
python scripts/make_supplement_tables.py
```

These create `figures/`, `tables/`, and `supplement/` **inside this extracted package**. They do not copy files into `paper_source.zip`. Data and stored numerical results are inputs. Generated formatting can differ from the final manuscript's local float placement and captions; the numeric rows and identities correspond. The finalized TeX objects are separately provided in `paper_source.zip`.

| Object | Evidence / generator |
|---|---|
| Main Table 1, sample | main_panel, task_quality_labels, configuration_selection |
| Main Table 2; S1-S3, identities | table2_configurations, configuration_selection; make_figures_tables / make_supplement_tables; candidate_selection.tex supplied with source |
| Main Table 3; S12, scopes | scope_analysis.json; make_scope_tables |
| Main Table 4, transitions | all91_pairs; make_figures_tables |
| Main Table 5; S6, thresholds | core_summary; sensitivity; make_supplement_tables (S6); main sensitivity table supplied with source |
| Main Table 6, affine components | pairwise_affine_components; scope_analysis; make_scope_tables |
| Main Table 7, common support | support_comparisons; make_scope_tables |
| Main Table 8, standardized reversed pairs | standardized_reversal_pairs; archived S13; make_scope_tables |
| Main Table 9, matching | matched_full_reference_summary; matching_balance; influence_summary; make_figures_tables |
| Main Table 11; archived S4-S5, Pro | pro_summary; pro_submission_metadata; make_figures_tables / make_supplement_tables |
| Main Table 10; archived S7, GEE contrasts | primary_profiles joined by full agent identifier; make_supplement_tables |
| S8, omnibus | interaction_summary; make_supplement_tables |
| S9, omissions | interaction_robustness with explicit internal-to-submission-to-manuscript mapping; make_supplement_tables |
| S10, all 91 pairs | all91_pairs; make_supplement_tables |
| S11, all 36 Pro pairs | pro_all36_pairs + pro_statistical_transitions; make_supplement_tables |
| Main Figure 1 | TikZ source in paper_source.zip, using frozen sample counts |
| Main Figures 2-3; Figure S1 | random_removal, all14_scores/ranks, all91_pairs; make_figures_tables |
| Main Figure 4 | affine_scores; make_additional_figure |

Internal A00-A13 codes are not manuscript A01-A14 minus one. Main, expanded-only and Pro identities remain separate. A reversed signed comparison negates the gap and reverses/negates interval endpoints; it leaves two-sided p/q unchanged. Source identities and hashes are never inferred from scores.

## Availability and terms

This repository provides the replication code, frozen analysis tables and existing outputs for the paper. It does not imply conference acceptance or completion of independent author reproduction. An anonymized review snapshot is available at https://anonymous.4open.science/r/FSE2027-Replication-E5F8/. The license for author-developed code remains unspecified; no new license grant is made. Third-party attribution and applicable upstream terms remain in `THIRD_PARTY_NOTICE.md`. No agent logs, model weights, task text, copied article PDFs or system fonts are bundled.

Frozen inputs are the distributed files in `data/` and `analysis_outputs/`. Commands write generated outputs to the locations described above; run them in a fresh extraction to preserve the snapshot. `SHA256SUMS.txt` covers all distributed files except itself. The table map refers to the main manuscript with core results integrated from the archived supplement. Generators retain their archived layouts; finalized main-manuscript TeX layouts are provided separately in `paper_source.zip`.
