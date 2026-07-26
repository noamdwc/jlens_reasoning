# Paper-Gap Reporting Design

## Goal

Restore the paper comparisons that were lost when the J-Lens sanity report
moved to typed experiment results.

## Report behavior

The capability table reports the spider J-Lens rank beside the configured paper
target rank. The intervention table reports the gap between each case's
`alpha=1` target rank and the configured paper target rank.

Rank gaps use:

```text
observed rank - paper target rank
```

A matching rank is displayed as `0 ranks`; a worse rank is displayed with a
leading plus sign, such as `+3 ranks`. Rows without a directly comparable paper
target display `N/A`.

Paper comparisons are diagnostic only. They do not change any experiment
check, failure, or overall PASS/FAIL result.

## Implementation

Reporting reads the existing `paper_target_rank` and `paper_primary_alpha`
values from `ExperimentResult.policy`; no new experiment fields or calculations
are added to execution code. A small formatting helper owns rank-gap rendering.

Focused reporter tests cover both the spider gap and per-intervention `alpha=1`
gap, including an exact match with the paper target.
