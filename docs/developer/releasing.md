# Releasing & Versioning

Deconvolve uses [`rooster`](https://github.com/astral-sh/rooster) for automated changelog generation and SemVer release management.

---

## Semantic Versioning (Pre-1.0)

Under SemVer below 1.0:

- `0.x.y`:
  - A **breaking change** bumps the minor version (`0.2.0` $\to$ `0.3.0`).
  - An **enhancement or bug fix** bumps the patch version (`0.2.0` $\to$ `0.2.1`).
  - Version `1.0.0` is reserved for the first official PyPI publication.

---

## PR Labels & Rooster Categories

`rooster` categorizes pull requests into sections in `CHANGELOG.md` using GitHub PR labels:

| Label           | Changelog Section    | SemVer Impact      |
| :-------------- | :------------------- | :----------------- |
| `breaking`      | **Breaking changes** | Minor bump (`0.x`) |
| `enhancement`   | **Enhancements**     | Patch bump         |
| `bug`           | **Bug fixes**        | Patch bump         |
| `performance`   | **Performance**      | Patch bump         |
| `configuration` | **Configuration**    | Patch bump         |
| `documentation` | **Documentation**    | Patch bump         |
| `preview`       | **Preview features** | Patch bump         |

PRs labeled with `internal`, `ci`, `testing`, or `automations` are excluded from the user-facing changelog.

---

## Release Process

1. **Ensure CI passes**:

   ```shell
   just ci
   ```

2. **Generate Changelog & Bump Version**:

   ```shell
   uv run rooster release
   ```

3. **Commit and Tag**:

   ```shell
   git commit -am "Release v0.x.y"
   git tag v0.x.y
   git push origin main --tags
   ```
