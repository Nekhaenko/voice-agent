# Spec Delta

## Purpose

Measures whether the agent's "is a web search needed?" decision is correct. Provides a labeled question set, an offline evaluation run, and a pass/fail report for the routing behavior.

## ADDED Requirements

### Requirement: Labeled question set drives routing evaluation
The system SHALL maintain a labeled set of Russian questions where each item states the expected routing decision: search required or search not required. The set SHALL include both categories.

#### Scenario: Dataset covers both decisions
- **WHEN** the evaluation dataset is loaded
- **THEN** it contains at least one item expecting a search and one item expecting direct answer, in Russian

#### Scenario: Unlabeled item is rejected
- **WHEN** a dataset item lacks an expected routing decision or an expected decision value is unrecognized
- **THEN** the evaluation run fails fast with an item-level error before executing

### Requirement: Evaluation reuses the real routing path without web requests
The eval runner SHALL produce each item's routing decision through the same routing path the live agent uses, but SHALL NOT reach the internet during an eval run: no DuckDuckGo requests are made and no final answer is generated.

#### Scenario: Decision matches and no search executes
- **WHEN** an item expecting "search required" is routed for evaluation
- **THEN** the runner records a hit only if the live routing path shows the search step would be entered, and no search request is sent

#### Scenario: No-answer guarantee during eval
- **WHEN** an evaluation run completes
- **THEN** it contains decisions only; no user-facing answer was produced and no internet search was contacted

### Requirement: Eval report is per-item and aggregate with a threshold
After a run, the system SHALL report each item's expected decision, actual decision, and hit/miss, plus aggregate accuracy. The run SHALL be marked passing only if aggregate accuracy meets a configurable minimum.

#### Scenario: Report content
- **WHEN** an evaluation run over the dataset completes
- **THEN** the report lists every item with expected and actual decision and hit or miss, shows overall accuracy, and states whether it meets the configured threshold

#### Scenario: Threshold failure is visible
- **WHEN** aggregate accuracy falls below the configured minimum
- **THEN** the run is reported as failed and the process signals failure so it can gate checks
