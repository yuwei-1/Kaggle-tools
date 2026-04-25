# Rules to working in this repo

- When naming id/title inside `kernel-metadata.json`, keep them consistent as Kaggle prefers this. E.g. an id of "noodl35/foo" should have a title of "foo".
- Be pragmatic. When creating scripts, we can sometimes forgo unittests or smoke tests in order to move faster. When adding to library, definitely always add unittests.
- For one-off Kaggle CLI variants, prefer local script changes over extracting new utilities unless the code is already reused or the shared abstraction is immediately justified.
