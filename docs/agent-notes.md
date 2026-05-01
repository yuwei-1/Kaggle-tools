# Rules to working in this repo

- When naming id/title inside `kernel-metadata.json`, keep them consistent as Kaggle prefers this. E.g. an id of "noodl35/foo" should have a title of "foo".
- Be pragmatic. When creating scripts, we can sometimes forgo unittests or smoke tests in order to move faster. When adding to library, definitely always add unittests.
- For deterministic library or utility behavior, prefer parameterized pytest cases with `pytest.param(..., id=...)` and dummy inputs that cover both happy paths and edge cases; avoid tests that only assert container shape or superficial structure when direct behavior can be asserted.
- For one-off Kaggle CLI variants, prefer local script changes over extracting new utilities unless the code is already reused or the shared abstraction is immediately justified.
- For new Kaggle CLI scripts, mirror the nearest existing script format unless there is a concrete repo-specific reason not to. Avoid introducing alternate dependency bootstraps or speculative environment guards before reproducing a real failure.
- When in doubt about the usage of a particular method or library, please stay grounded by either reading the docs online or looking into library code to confirm the correct usage instead of guessing.
