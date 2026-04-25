# ktools Refactor Review

Date: 2026-04-25

## Scope

This note is based on a review of the current `ktools` package layout, representative files in each package, repo-wide imports, and the existing test layout.

What I reviewed:

- All non-cache files under `ktools`.
- Representative files from every top-level package.
- Repo-wide imports to see which namespaces are still active.
- The existing top-level `tests/` tree versus tests embedded inside `ktools`.
- Current worktree signals, including the in-progress move from `feature_engineering` to `fe`.

At the time of review there are about 157 non-cache files under `ktools`.

## Executive Summary

`ktools` is messy mainly because it contains two different library designs at the same time:

1. A newer, cleaner stack built around:
   - `BaseKtoolsModel`
   - `DatasetConfig`
   - `BasePreprocessor`
   - `ktools.models.*`
   - `ModelPipeline`

2. An older stack built around:
   - `IKtoolsModel`
   - `DataSciencePipelineSettings`
   - `basic_feature_transformers`
   - `modelling/*`
   - the older cross-validation executors

Those two stacks are not isolated from each other. They cross-import in enough places that the package boundaries no longer map cleanly to responsibilities.

The biggest structural problems are:

- Two competing model APIs.
- Two competing preprocessing/data APIs.
- Namespace overlap between `modelling` and `models`.
- Namespace overlap between `fe`, `feature_selection`, and parts of `preprocessing`.
- Tests and fixtures living both inside `ktools` and in the top-level `tests/` tree.
- Inconsistent naming and spelling: `modelling` vs `models`, `visualisation` vs `visualization`, `Tests` vs `tests`, `Automl_models`, `Interfaces`, and several typos.
- A large `utils` bucket containing unrelated concerns.
- Side effects at package import time.

My recommendation is to choose the newer typed stack as the canonical architecture, quarantine the legacy stack behind compatibility shims, and then collapse duplicate namespaces in phases.

## Current Top-Level Package Shape

Approximate non-cache file counts by top-level package:

| Package | File count | Notes |
| --- | ---: | --- |
| `base` | 4 | Small and coherent; useful base abstractions. |
| `config` | 1 | Small and coherent; useful typed config. |
| `experiment` | 1 | Too small to justify a top-level package. |
| `fe` | 7 | Feature engineering, but abbreviated and overlapping with other areas. |
| `feature_selection` | 5 | Reasonable concern, but should be grouped with feature work. |
| `fitting` | 8 | Mixed old and new orchestration styles. |
| `hyperopt` | 11 | Good responsibility, but should be renamed and tightened. |
| `metrics` | 3 | Coherent and small. |
| `model_selection` | 10 | Mixed concerns; not all files belong here. |
| `modelling` | 62 | Main concentration of legacy design and duplication. |
| `models` | 9 | Cleaner canonical surface for model implementations. |
| `preprocessing` | 16 | Mixed old and new preprocessing styles. |
| `utils` | 7 | Grab bag; should be reduced aggressively. |
| `visualisation` | 10 | Useful concern, but naming and structure need cleanup. |

The `modelling` package is the clearest signal that the current hierarchy reflects project history rather than a clean design.

## Current Architecture Split

### Newer stack

These files look like the intended long-term direction:

- `ktools/base/model.py`
- `ktools/base/preprocessor.py`
- `ktools/base/joblib_mixin.py`
- `ktools/config/dataset.py`
- `ktools/models/*`
- `ktools/preprocessing/categorical.py`
- `ktools/preprocessing/numerical.py`
- `ktools/preprocessing/core.py`
- `ktools/preprocessing/pipe.py`
- `ktools/fitting/pipe.py`
- `ktools/experiment/validation_selector.py`

Characteristics of this stack:

- Typed, explicit interfaces.
- Schema is carried via `DatasetConfig`.
- Preprocessors are object-oriented and fit/transform compatible.
- Models implement a consistent `BaseKtoolsModel` contract.
- Pipelines read like standard train/inference flow.

### Legacy stack

These files reflect the older design:

- `ktools/modelling/Interfaces/*`
- `ktools/modelling/*`
- `ktools/utils/data_science_pipeline_settings.py`
- `ktools/preprocessing/basic_feature_transformers.py`
- `ktools/preprocessing/basic_preprocessor.py`
- `ktools/preprocessing/i_feature_transformer.py`
- `ktools/preprocessing/i_preprocessing_utility.py`
- `ktools/fitting/cross_validation_executor.py`
- `ktools/fitting/memory_efficient_cross_validate_test_executor.py`
- `ktools/fitting/safe_cross_validation_executor.py`

Characteristics of this stack:

- Heavier use of mutable settings objects.
- Less clear package boundaries.
- Older interface names and wrapper layers.
- Tests embedded inside the source tree.
- More broad imports and more hidden coupling.

## Main Structural Problems

### 1. Two competing model interfaces

There are two model contracts:

- `BaseKtoolsModel` in `ktools/base/model.py`
- `IKtoolsModel` in `ktools/modelling/Interfaces/i_ktools_model.py`

This is the main architectural split.

Examples:

- `ktools/models/gbdt/lightgbm.py` uses `BaseKtoolsModel`.
- `ktools/fitting/pipe.py` expects `BaseKtoolsModel`.
- `ktools/fitting/cross_validation_executor.py` expects `IKtoolsModel`.
- Much of `ktools/modelling/*` still implements `IKtoolsModel`.

Result: the training/orchestration layer cannot be reasoned about cleanly, because different entry points assume different model bases.

### 2. Two competing preprocessing/data abstractions

There are again two styles:

- Newer: `DatasetConfig` plus `BasePreprocessor` and `PreprocessingPipeline`
- Older: `DataSciencePipelineSettings` plus `basic_feature_transformers`

`DatasetConfig` is a schema object.
`DataSciencePipelineSettings` is a much larger bundle of paths, dataframes, mutation state, and transformation state.

Those are different levels of abstraction and should not coexist without a very explicit boundary.

If `DataSciencePipelineSettings` is still needed, it should become a dedicated data bundle object under a clearly named package. It should not live in `utils`.

### 3. `modelling` and `models` overlap badly

`models` is the cleaner public-facing namespace.
`modelling` is a large older namespace containing:

- interfaces
- automl wrappers
- bayesian models
- ensembles
- wrappers
- pytorch utilities
- multiple concrete model implementations
- embedded tests and fixtures

There is no good long-term reason to keep both `modelling` and `models` as first-class top-level packages.

### 4. Feature work is split across multiple packages

Feature-related logic is spread across:

- `fe`
- `feature_selection`
- parts of `preprocessing`
- some orchestration-style utilities such as `robust_feature_importance_checker`

This makes it hard to answer basic questions like:

- What is feature engineering?
- What is preprocessing?
- What is feature selection?
- What is inspection or experimentation?

These should be separated by responsibility, not by project history.

### 5. Tests and fixtures are embedded inside the package

There is already a top-level `tests/` tree, but there are also embedded tests and test assets inside `ktools`, including:

- `ktools/feature_selection/tests/*`
- `ktools/fitting/Tests/*`
- `ktools/model_selection/tests/*`
- `ktools/modelling/Tests/*`
- `ktools/modelling/ktools_models/pytorch_nns/Tests/*`
- `ktools/preprocessing/Tests/*`

This is a major source of clutter.

Tests and test data should live under top-level `tests/`, not inside the installable package.

### 6. Naming is inconsistent

Examples:

- `modelling` vs `models`
- `visualisation` vs likely expected `visualization`
- `Tests` vs `tests`
- `Interfaces` vs `interfaces`
- `Automl_models` vs `automl`
- `create_plots_for_binary_classication_problem.py` has a typo
- `categorical_denoiser_prepreprocesser.py` has a typo
- `VariantionalKernelSelector` has a typo

This makes the tree feel unreliable even before reading the code.

### 7. `utils` is doing too much

Current utility modules include:

- `helpers.py`
- `loader.py`
- `load_dataframes.py`
- `reduce_dataframe_usage.py`
- `tools.py`
- `data_science_pipeline_settings.py`

These are not one category.

They belong to different concerns:

- data loading
- optimization/grid loading
- dataframe memory management
- task inference
- generic transforms
- dataset bundle state

`utils` should be the smallest package in a clean design, not a fallback drawer.

### 8. Package import side effects

`ktools/__init__.py` currently prints a pyfiglet banner and creates a logger on import.

Package imports should not print banners.
That belongs in a CLI or explicitly called startup path, not in the library root.

## Proposed Direction

Use the newer stack as the canonical architecture.

Specifically:

- Canonical model base: `BaseKtoolsModel`
- Canonical schema/config object: `DatasetConfig`
- Canonical preprocessing style: `BasePreprocessor` + `PreprocessingPipeline`
- Canonical model namespace: `ktools.models`
- Canonical orchestration surface: training/pipeline classes that operate on the new model and preprocessor abstractions

Treat the following as legacy and phase them out behind shims:

- `ktools.modelling`
- `IKtoolsModel`
- `DataSciencePipelineSettings`
- settings-driven feature transformer chains
- embedded tests inside `ktools`

## Proposed Target Hierarchy

I would aim for something close to this:

```text
ktools/
  __init__.py
  core/
    constants.py
    logging.py
    persistence.py
    model.py
    preprocessor.py
  config/
    dataset.py
  data/
    bundle.py
    kaggle.py
    loading.py
  preprocessing/
    categorical.py
    numerical.py
    memory.py
    pipeline.py
  features/
    engineering/
      basic.py
      deep.py
      pca.py
    selection/
      shap.py
    inspection/
      robust_importance.py
  models/
    automl/
    bayesian/
    ensemble/
    gbdt/
    nn/
    wrappers/
  training/
    cv.py
    pipelines.py
    splitters.py
    validation.py
  optimization/
    optuna.py
    search_spaces/
    feature_elimination.py
  metrics/
  visualization/
    eda.py
    classification/
    regression/
  legacy/
    modelling/
    settings_pipeline/
```

This is intentionally responsibility-based:

- core abstractions
- config
- data loading/bundling
- preprocessing
- features
- models
- training
- optimization
- metrics
- visualization
- legacy

If you do not want a `data/` package, then `bundle.py`, `kaggle.py`, and `loading.py` can live under `config/` or `io/`. The important part is not the exact name. The important part is that they do not live in `utils` or `preprocessing`.

## Package-by-Package Recommendation

### `base`

Keep the contents, but rename the package to something clearer like `core`.

Suggested moves:

- `ktools/base/model.py` -> `ktools/core/model.py`
- `ktools/base/preprocessor.py` -> `ktools/core/preprocessor.py`
- `ktools/base/joblib_mixin.py` -> `ktools/core/persistence.py`

Rationale:

- `base` is acceptable, but `core` reads more clearly when the package is meant to define foundational abstractions.

### `config`

Keep and expand carefully.

Current file:

- `ktools/config/dataset.py`

This is good. It is small, explicit, and actually belongs in config.

Potential additions in the future:

- dataset bundles or train/test path config
- task-level configuration objects
- experiment configuration objects

### `definitions.py`

Move this under a core/config namespace.

Suggested target:

- `ktools/core/constants.py`

Reason:

- Repository-wide constants are not really a sibling of `models`, `preprocessing`, and `metrics`.

### `logger_setup.py`

Move this under core.

Suggested target:

- `ktools/core/logging.py`

Also remove implicit logger setup from `ktools/__init__.py`.

### `fe`

Do not keep `fe` as the long-term canonical namespace.

Short names save a few characters but reduce readability. `features` is clearer and still concise.

Suggested outcome:

- Merge `fe` into `features/engineering`

Suggested moves:

- `ktools/fe/basic_feature_creators.py` -> `ktools/features/engineering/basic.py`
- `ktools/fe/deep_feature_creator.py` -> `ktools/features/engineering/deep.py`
- `ktools/fe/pca_feature_creator.py` -> `ktools/features/engineering/pca.py`
- `ktools/fe/interfaces/i_feature_creator.py` -> `ktools/features/engineering/interfaces.py`

### `feature_selection`

Keep the concern, but group it under the broader feature area.

Suggested moves:

- `ktools/feature_selection/shap_feature_selector.py` -> `ktools/features/selection/shap.py`

Then delete the top-level `feature_selection` package once imports are migrated.

### `fe/robust_feature_importance_checker.py`

This does not belong in feature engineering.

It is an analysis or inspection utility that runs repeated CV experiments to judge feature usefulness.

Suggested target:

- `ktools/features/inspection/robust_importance.py`

### `preprocessing`

This package currently contains both good modern code and legacy code.

Keep in `preprocessing`:

- `categorical.py`
- `numerical.py`
- `core.py`
- `pipe.py`

Move out of canonical preprocessing and treat as legacy:

- `basic_feature_transformers.py`
- `basic_preprocessor.py`
- `i_feature_transformer.py`
- `i_preprocessing_utility.py`

Reason:

- These modules use the older `DataSciencePipelineSettings` flow.
- They are not the same abstraction as the `BasePreprocessor` style classes.

If they still matter, move them to something like:

- `ktools/legacy/settings_pipeline/*`

### `models`

Keep this as the canonical model namespace.

It already has the right direction:

- `ktools/models/gbdt/*`
- `ktools/models/nn/*`
- `ktools/models/automl/flaml.py`

This package should become the only public home for model implementations.

Expand it by absorbing the useful parts of `modelling`.

### `modelling`

This should not survive as a top-level long-term package.

Break it up by responsibility:

- `modelling/Automl_models/*` -> `models/automl/*`
- `modelling/bayesian/*` -> `models/bayesian/*`
- `modelling/ensemble/*` -> `models/ensemble/*`
- `modelling/model_transform_wrappers/*` -> `models/wrappers/*`
- `modelling/pytorch_utils/*` -> probably `models/nn/utils/*`
- `modelling/ktools_models/*` -> split between `models/gbdt`, `models/nn`, `models/bayesian`, or delete if superseded

Anything still needed from `modelling` during migration should live behind import shims only.

### `fitting`

This should become a more clearly named training/orchestration package.

Suggested target:

- rename `fitting` to `training`

Likely contents:

- CV executors
- model pipelines
- validation drivers
- split orchestration

Important: first unify it around `BaseKtoolsModel`, not `IKtoolsModel`.

### `experiment`

This package is too small to exist on its own.

`validation_selector.py` belongs with training/validation orchestration.

Suggested move:

- `ktools/experiment/validation_selector.py` -> `ktools/training/validation.py`

### `model_selection`

This package mixes unrelated things.

What to keep conceptually:

- splitters such as `walk_forward_splits.py`

What does not belong here:

- `kernel_selector.py`

`kernel_selector.py` is really a Bayesian model helper, not model selection in the sklearn sense.

Suggested moves:

- `ktools/model_selection/walk_forward_splits.py` -> `ktools/training/splitters.py`
- `ktools/model_selection/kernel_selector.py` -> `ktools/models/bayesian/kernel_selector.py`

### `hyperopt`

Keep the responsibility, but rename the package.

Suggested target:

- `optimization`

Suggested structure:

- `optimization/optuna.py`
- `optimization/search_spaces/*`
- `optimization/feature_elimination.py`

Suggested moves:

- `ktools/hyperopt/optuna_optimiser.py` -> `ktools/optimization/optuna.py`
- `ktools/hyperopt/grids/*` -> `ktools/optimization/search_spaces/*`
- `ktools/hyperopt/recursive_feature_elimination_executor.py` -> `ktools/optimization/feature_elimination.py`
- `ktools/utils/loader.py` -> `ktools/optimization/grid_loader.py`

### `metrics`

Keep this package. It is small and coherent.

Only cleanup needed:

- move embedded tests to top-level `tests/metrics`
- keep names consistent

### `visualisation`

Keep the concern, but standardize the spelling.

My preference would be `visualization` because it is more common in Python package naming, but the important part is consistency.

Suggested structure:

- `visualization/eda.py`
- `visualization/classification/*`
- `visualization/regression/*`

Suggested moves:

- `ktools/visualisation/basic_eda.py` -> `ktools/visualization/eda.py`
- `ktools/visualisation/create_plots_for_binary_classication_problem.py` -> split or rename under `visualization/classification/`
- `ktools/visualisation/create_plots_for_regression_problem.py` -> split or rename under `visualization/regression/`

### `utils`

Shrink this package aggressively.

Proposed moves:

- `helpers.py` -> split by responsibility; `infer_task` likely belongs in `models/task.py` or `core/task.py`
- `loader.py` -> `optimization/grid_loader.py`
- `load_dataframes.py` -> `data/loading.py`
- `reduce_dataframe_usage.py` -> `preprocessing/memory.py` or `dataframe/memory.py`
- `tools.py` -> rename by what it actually does
- `data_science_pipeline_settings.py` -> `legacy/settings_pipeline/bundle.py` unless redesigned into a proper data bundle

As a rule, if a file has a specific responsibility, it should not be under `utils`.

## What I Would Treat As Canonical Public API

I would converge toward the following as the library surface:

- `ktools.models.*`
- `ktools.preprocessing.*`
- `ktools.training.*`
- `ktools.metrics.*`
- `ktools.features.*`
- `ktools.optimization.*`
- `ktools.config.*`

Everything else should either be internal/private or legacy compatibility.

## What I Would Mark As Legacy Immediately

These areas should be explicitly treated as migration surfaces, not normal package structure:

- `ktools.modelling`
- `ktools.modelling.Interfaces`
- `ktools.preprocesssing.basic_feature_transformers`
- `ktools.preprocesssing.basic_preprocessor`
- `ktools.utils.data_science_pipeline_settings`
- `ktools.fitting.cross_validation_executor`
- `ktools.fitting.safe_cross_validation_executor`
- `ktools.fitting.memory_efficient_cross_validate_test_executor`

The main point is to stop pretending these are equal peers to the newer architecture.

## Tests and Fixtures

This is the easiest cleanup win.

There should be one rule:

- installable package code lives under `ktools/`
- tests and fixtures live under top-level `tests/`

That means moving all of these out of `ktools`:

- `ktools/feature_selection/tests/*`
- `ktools/fitting/Tests/*`
- `ktools/model_selection/tests/*`
- `ktools/modelling/Tests/*`
- `ktools/modelling/ktools_models/pytorch_nns/Tests/*`
- `ktools/preprocessing/Tests/*`

Also move fixture data such as:

- pngs
- npy files
- sample csvs

Suggested target shape:

```text
tests/
  features/
  fitting/
  models/
  optimization/
  preprocessing/
  metrics/
  visualization/
  data/
```

If you want package-local test organization mirrored at the top level, that is fine. Just do not keep tests inside `ktools`.

## Naming Cleanup

These are small but worth fixing because they affect the feel of the whole repository.

I would standardize all of the following:

- one spelling: `models`, not `modelling`
- one spelling: `visualization` or `visualisation`, pick one and stick to it
- lowercase package names everywhere
- lowercase `tests`
- lowercase `interfaces`
- lowercase `automl`

I would also fix obvious typos:

- `create_plots_for_binary_classication_problem.py`
- `categorical_denoiser_prepreprocesser.py`
- `VariantionalKernelSelector`

These are low-risk cleanups that make the tree easier to trust.

## Current Worktree Context

The current working tree already suggests some reorganization is underway, especially a move from `feature_engineering` toward `fe` and migration of some tests to top-level `tests/`.

That direction is partly good:

- moving tests out of `ktools` is correct
- reducing duplicate package trees is correct

But I would not stop at `fe` as the final canonical name. I would still prefer `features` or `feature_engineering` over `fe` because it is clearer.

If brevity matters, `features` is a better compromise than `fe`.

## Proposed Migration Plan

### Phase 1: Freeze the target architecture

Before moving lots of files, decide these rules:

- `BaseKtoolsModel` is the only canonical model interface
- `DatasetConfig` plus object-oriented preprocessors is the canonical preprocessing path
- `models` is the only canonical model namespace
- tests do not live under `ktools`
- new code does not go into `modelling`
- new code does not go into `utils` unless it truly has no better home

This phase is mostly architectural policy.

### Phase 2: Clean the low-risk layout problems

Do these first because they are high-value and low-risk:

- Move embedded tests and fixtures to top-level `tests/`
- Delete `.DS_Store` files from tracked source directories
- Ensure `__pycache__` is ignored and not tracked
- Standardize `Tests` to `tests` if any remain temporarily
- Remove package-root side effects from `ktools/__init__.py`

This will already make the package feel less messy.

### Phase 3: Collapse feature packages

Move toward one feature namespace:

- merge `fe` and `feature_selection` into `features`
- move inspection/analysis utilities into a dedicated `inspection` subpackage
- keep preprocessing separate from feature work

This resolves one major overlap without touching model internals yet.

### Phase 4: Collapse training/orchestration around the new interfaces

This is the important code-level refactor.

Goal:

- training code should operate on `BaseKtoolsModel` and `PreprocessingPipeline`

Actions:

- replace or adapt old `IKtoolsModel`-based executors
- unify CV flow under one training package
- move splitters into `training/splitters.py`
- move experiment validation logic into training

Do not keep both orchestration styles long-term.

### Phase 5: Break up `modelling`

Move still-useful code out of `modelling` into:

- `models/automl`
- `models/bayesian`
- `models/ensemble`
- `models/wrappers`
- `models/nn`

Anything not worth preserving should be deleted instead of relocated.

Once imports are migrated, replace `modelling` with thin compatibility shims or remove it entirely.

### Phase 6: Eliminate the settings-object legacy path

Decide what to do with `DataSciencePipelineSettings`:

- either redesign it into a clean data bundle under `data/` or `config/`
- or quarantine it under `legacy/`

Do the same for:

- `basic_feature_transformers`
- settings-driven preprocessors
- wrapper flows that depend on the old mutation-heavy pipeline

This is where the architecture becomes genuinely clean rather than only cosmetically improved.

## Short-Term Priorities

If I were prioritizing the first concrete cleanups, I would do them in this order:

1. Move all tests and fixtures out of `ktools`.
2. Remove import side effects from `ktools/__init__.py`.
3. Declare `models` + `BaseKtoolsModel` as canonical.
4. Stop adding new code to `modelling`.
5. Split `fe` / `feature_selection` / inspection logic into one feature area.
6. Rename `hyperopt` to `optimization`.
7. Shrink `utils` by moving files to responsibility-based homes.
8. Standardize naming and fix typos.

## Things I Would Not Do

I would not do a giant file move all at once without first freezing the architecture and deciding the canonical public API.

I would not keep both `models` and `modelling` as long-term first-class packages.

I would not keep both `DatasetConfig` and `DataSciencePipelineSettings` as equally normal ways of representing data flow.

I would not keep package-local `Tests` trees when there is already a top-level `tests/` folder.

I would not add more code to `utils` unless there is genuinely no narrower responsibility.

## Bottom Line

The package is not messy because it has many files.
It is messy because it has duplicate architectural centers.

The cleanest end state is:

- one model interface
- one preprocessing/dataflow style
- one model namespace
- one feature namespace
- one training namespace
- one optimization namespace
- one visualization spelling
- zero tests inside `ktools`

If you follow that, the hierarchy becomes understandable very quickly.
