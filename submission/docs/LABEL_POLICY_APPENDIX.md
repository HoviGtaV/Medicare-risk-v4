# Label Policy Appendix

## Objective
This project predicts provider-level Medicare risk using exclusion-based proxy labels. The model produces a risk score, not a legal fraud judgment.

## Unit of analysis
- one row = one provider-year
- provider identifier = `npi_clean`
- year field = `summary_year`

## Target
- `target_excluded_24m`

Definition:
- `target_excluded_24m = 1` if provider exclusion happens strictly after the provider-year end and within 24 months after that year end
- `target_excluded_24m = 0` otherwise, after applying exclusion and censoring rules

## Time alignment
For each provider-year:
- `year_end = summary_year-12-31`
- `horizon_end = (summary_year + 2)-12-31`

A positive label is assigned only when:
- exclusion date > year_end
- and exclusion date <= horizon_end

## Exclusion rules
Rows are excluded from the modeling dataset if:
- the provider was already excluded on or before the summary year end
- follow-up is incomplete at the censor date

## Censoring rule
Only rows with complete follow-up are kept for modeling. This avoids incorrectly treating incomplete future observation as a true negative.

## Negative class
A negative row is a provider-year where:
- follow-up is complete
- exclusion did not occur within 24 months after year end
- the provider was not already excluded by year end

## Duplicate handling
Before final v4 train/validation/test splitting:
- duplicates on (`npi_clean`, `summary_year`) are removed
- keep rule: `keep="first"`

## Final split
- Train: 2018-2021
- Validation: 2022
- Final train for official model: 2018-2022
- Final holdout test: 2023

## Interpretation
The target is a proxy based on exclusion information. It does not prove fraud, guilt, or intent. The model output should be interpreted only as a risk score for ranking and review prioritization.

## Limitations
- exclusions are imperfect proxy labels
- exclusion may happen after provider behavior reflected in claims summaries
- some high-risk providers may never be excluded
- the current deployment story is strongest for batch scoring of feature-ready provider-year tables


