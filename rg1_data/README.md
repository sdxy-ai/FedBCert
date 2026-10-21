# RG-1.0 saved records

This directory exports the completed frozen RG-1.0 experiment. It contains no
training program, model re-evaluation, new bootstrap, machine/user paths, or
changed experimental measurements. All 32 cases and negative outcomes remain.

Frozen implementation commit: `e99878f48ef441346011f27e5bdd63df3e568f63`.

## Files

- `case_metrics.csv` / `.json`: 32 cases; local P/I/MLP utility and all/exposed
  graph/test class/probability differences and S/D/N coverage.
- `unit_workload_FN.csv`: 64 paired F/N comparisons, absolute byte differences,
  case-relative reductions, and within-case median CPU ratios.
- `unit_workload_strategy_medians.csv`: all 512 case×workload×strategy rows;
  P uses L/F/E/S/D/N and controls I/MLP use L. Numeric values are three-formal-
  repeat medians; no warmup is included. Count/byte invariants remain unchanged.
- `group_workload_strategy_means.csv`: four dataset/partition groups plus
  Overall, two workloads and eight model/strategy entries. Equal case means of
  the preceding medians; descriptive only, with no new interval or p-value.
- `group_workload_FN.csv`: mean absolute bytes, paired reductions, the separately
  labeled ratio-of-mean-byte reductions, paired CPU ratios, adverse-case counts.
- `frozen_statistics_all_scopes.csv` / `analysis.json`: unchanged frozen
  descriptive analysis and all overall/dataset/dataset×partition summaries.
  `analysis.json` also retains each seed's flattened metrics.
- `all_records.json`: all 1,536 formal repeated measurements (P 1,152,
  controls 384), not 1,536 independent statistical units.
- `case_summaries/`: all 32 original summary contents, including warmup timing
  rows, routed utility and certified accuracy. Warmups are not in median CSVs.

## Units and field meanings

Rates and reductions in CSV/JSON are fractions; multiply by 100 for percent or
percentage points as appropriate. Times are seconds; displayed PDF times may
be milliseconds. `cpu_seconds` is the primary outer `perf_counter` wall clock;
`process_seconds` is auxiliary process CPU time, not the primary timing. Stage
times need not sum to the outer total. `fallback_accounting/inference_seconds`
is internal fallback service time, not end-to-end inference time.

Bytes count serialized application requests and responses including required
summary/header/ID fields. `request_response_packets` is an application-message
count, not physical network packets. `remote_vectors` and
`fallback_accounting/propagated_rows` are work/transmission counts, not unique
nodes. Encoded/reused rows and remote stages are likewise cumulative counts.
No link-security, network latency or approval costs are measured.

`NA` CSV cells correspond to absent/undefined values, never imputed zero. For
example, local/control paths do not define remote-service stage timings.
Both signed negative and positive results are retained. `certified_accuracy`
has all assessed group nodes as denominator; it is not accuracy conditional on
certification. F/N Macro-F1 may coincide in class while their logits differ.

The two workloads share frozen models but are not interchangeable. In the
32-node workload, N decreases bytes in 24/32 cases and increases them in eight,
all CiteSeer/METIS. The paired reduction and ratio-of-means reduction differ;
neither is a substitute selected after observing its sign. Statistical units
are dataset/partition/base-seed cases, not targets, clients or timing repeats.

Only the already frozen 10,000-draw descriptive paired bootstrap is reproduced
here (analysis RNG 42028616). These fixed graphs are not independent draws from
a graph population. No strict floating-point soundness follows from zero
observed violations or the numerical guard.

## Source-content checksums

- `all_records.json`: `d461beaa360c16b20a8ff72fe0a3722114bbf4d006a64ce60bfa857f01b3e2c6`
- `case_metrics.json`: `2375b48eec4f513267dc1129d5dc8ec9d51575978d19b0e4c7a6eddeaf1def70`
- `analysis.json`: `83bbf3fa50623fa844b313989d0f08c798c06e69a30879fc9232c386bc02a34f`

Machine manifests, approval records and private environment paths are not
included. Run `python scripts/verify_rg1_reference.py` from the repository root
to check the three pinned JSON files and recompute the frozen descriptive
analysis. This needs no model checkpoints or benchmark downloads.
