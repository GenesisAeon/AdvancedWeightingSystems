# Release Guide

This package follows the GenesisAeon ecosystem release process.

## Versioning

We use [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.

- **MAJOR** — breaking changes to the public API (`AeonLayer`, `WeightingEngine`,
  `SymbolicMirror`, `ResonanceCoupling`).
- **MINOR** — new features, backwards-compatible.
- **PATCH** — bug fixes, documentation, dependency bumps.

## Release types

| Tag pattern | Channel | Where it publishes |
|---|---|---|
| `vX.Y.Z` | Production | PyPI, GitHub Release, Zenodo (if integration enabled) |

## How to cut a release

1. Ensure `CHANGELOG.md` has an entry for the new version under `## [X.Y.Z]`.
2. Ensure `pyproject.toml`'s `[project].version` matches.
3. Ensure `.zenodo.json`'s `"version"` field matches.
4. Commit these changes (if any) to `main`.
5. Tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
6. `.github/workflows/release.yml` builds, tests, and publishes automatically.
7. If Zenodo-GitHub integration is enabled for this repo, a new Zenodo DOI
   version is minted automatically from the GitHub Release using
   `.zenodo.json` metadata.

## Dependency pins within the GenesisAeon ecosystem

This package depends on other `GenesisAeon/*` packages via the `stack` extra
(`mirror-machine`, `entropy-governance`, `sigillin`, `mandala-visualizer`,
`utac-core`, `cosmic-web`). Pin them with `>=` lower bounds matching the
minimum version that provides the API this package relies on. Do not pin
exact versions (`==`) for ecosystem dependencies.
