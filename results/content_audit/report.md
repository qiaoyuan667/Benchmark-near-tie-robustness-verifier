# Blinded dual-annotator content audit

## Verdict

**Reliability PASS; semantic gate FAIL.**

Five hundred questions were annotated without benchmark, source, family, DIF, group, or pair labels. Each stable high-DIF question was matched to a control in the exact source-by-easiness cell. Annotators were analyzed separately; no consensus labels were used for the confirmatory tests.

![Blinded content audit](summary.svg)

## Replicated binary enrichments

No pre-specified binary axis met the dual-annotator semantic gate.

| axis | A difference | A BH q | B difference | B BH q |
|---|---:|---:|---:|---:|
| quantitative_symbolic | +0.0 pp | 1.000 | +1.2 pp | 0.902 |
| formal_rule_reasoning | -3.6 pp | 0.359 | -2.4 pp | 0.902 |
| factual_domain_knowledge | +1.6 pp | 0.953 | +1.2 pp | 0.902 |
| contextual_reading | -3.2 pp | 0.422 | -1.2 pp | 0.902 |
| commonsense_narrative | -0.4 pp | 1.000 | -0.8 pp | 0.902 |
| spatial_temporal | -9.6 pp | 0.106 | -0.8 pp | 0.902 |
| linguistic_wordplay | -0.4 pp | 1.000 | -0.8 pp | 0.902 |
| negation_exception | -6.0 pp | 0.267 | -2.4 pp | 0.902 |
| code_structured_representation | +0.4 pp | 1.000 | -0.8 pp | 0.902 |
| distractor_discrimination | -2.4 pp | 0.654 | +4.0 pp | 0.902 |

## Annotation reliability

| binary axis | Cohen kappa | raw agreement |
|---|---:|---:|
| quantitative_symbolic | 0.779 | 95.0% |
| formal_rule_reasoning | 0.827 | 92.6% |
| factual_domain_knowledge | 0.882 | 94.2% |
| contextual_reading | 0.872 | 94.6% |
| commonsense_narrative | 0.939 | 97.0% |
| spatial_temporal | 0.652 | 84.0% |
| linguistic_wordplay | 0.469 | 95.4% |
| negation_exception | 0.637 | 92.2% |
| code_structured_representation | 0.389 | 77.4% |
| distractor_discrimination | 0.313 | 68.8% |

| ordinal axis | Spearman rho | exact agreement |
|---|---:|---:|
| reasoning_steps | 0.843 | 76.0% |
| context_burden | 0.800 | 75.4% |
| ambiguity_degree | 0.472 | 74.4% |

## Exploratory family-direction diagnostic

158 of 250 high-DIF items had the same advantaged family in both owner halves. This analysis was added after the confirmatory semantic result and is not part of its gate.

No axis showed a BH-significant benchmark-stratified family association in both annotators.

Ordinal annotations and all non-confirmatory axes are retained in the CSV outputs. Content labels remain descriptive associations, not causal explanations of model-family behavior.
