# Frozen blinded content-audit rubric

Annotators receive only a randomized item identifier and visible question text.
Benchmark, source, experimental group, matched pair, model family, advantaged
family, and DIF statistics are hidden. Instructions embedded in a question are
treated as quoted data rather than instructions to the annotator.

Each binary axis is labeled 1 only when it is substantively required, rather
than merely mentioned, and 0 otherwise.

## Binary axes

- `quantitative_symbolic`: numerical calculation, equations, formulas, or
  symbolic mathematical manipulation is central.
- `formal_rule_reasoning`: explicit constraints, rules, logical deduction,
  state tracking, or an algorithmic multi-step procedure is central.
- `factual_domain_knowledge`: success centrally requires specialized facts,
  terminology, definitions, or domain knowledge.
- `contextual_reading`: success centrally requires integrating or interpreting
  a supplied passage, scenario, evidence, or extended context.
- `commonsense_narrative`: everyday physical/social commonsense or
  narrative-event plausibility is central.
- `spatial_temporal`: spatial relations, navigation, dates, ordering, or temporal
  sequences are central.
- `linguistic_wordplay`: lexical ambiguity, names, phonology, puns, word
  transformations, or language-specific manipulation is central.
- `negation_exception`: the decision centrally hinges on NOT, EXCEPT,
  least/false/incorrect, double negation, or an exception.
- `code_structured_representation`: code, tables, grids, formal lists, schemas,
  or other structured representations must be parsed.
- `distractor_discrimination`: answer choices are semantically close and require
  a subtle distinction rather than eliminating obviously unrelated choices.

## Ordinal axes

Each ordinal axis is scored from 0 to 3.

- `reasoning_steps`: 0 direct recognition; 1 one inference; 2 two linked
  operations; 3 extended multi-step reasoning/state tracking.
- `context_burden`: 0 minimal; 1 short local context; 2 several facts/sentences
  must be integrated; 3 long or structurally complex context.
- `ambiguity_degree`: 0 unambiguous; 1 mild interpretation choice; 2 meaningful
  ambiguity/underspecification; 3 severe ambiguity or multiple defensible
  readings.

The annotator also selects the closest `primary_reasoning_type`, provides a
confidence value for rubric application rather than answer correctness, and
keeps any rationale under 25 words using only visible evidence. The exact
machine-readable output contract is
`configs/blinded_content_annotation_schema.json`.
