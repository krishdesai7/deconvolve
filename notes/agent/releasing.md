# Releasing

One button. **Actions -> Release -> Run workflow**, pick a bump (or leave it
blank), and the job does the rest: rewrite the version, write the changelog
stanza, re-lock, commit to master, tag, and publish the GitHub release.

Nothing is released by merging a PR. `deconvolve evaluate` and friends do not read the
version, so a release is a labelling act, not a build step — it exists to give
a result you can cite a fixed point in the code.

**The bump comes from PR labels.** rooster reads every PR merged since the last
tag and takes the largest bump any label implies: `breaking` -> minor, anything
else -> patch. Labels in `ignore-labels` (`internal`, `ci`, `testing`,
`automations`) contribute nothing, so a release consisting only of plumbing
aborts with "No pull requests found after applying ignored labels" — working as
intended, not a failure to debug. Give a PR a real label if you want it to show
up in the changelog. The `bump` input overrides the inference when you want to
force one.

**Versions stay below 1.0.** `major-labels` is empty and the workflow refuses to
tag anything outside `0.x`, so no label and no merge can walk the project into
1.0. That number is reserved for the first PyPI publication; see below.

**master is protected by a ruleset, and the release job is the one exception.**
Every human change to master goes through a PR with a green `ci`. Force-pushes
and deletions are blocked. The release job pushes directly because a
write-scoped deploy key is the ruleset's bypass actor, and `actions/checkout`
loads it from the `RELEASE_SSH_KEY` secret. `GITHUB_TOKEN` cannot be given a
bypass here: GitHub only accepts the GitHub Actions app as a bypass actor on
organization-owned repositories, and this repository belongs to a user. A PAT
would work but would carry a person's identity, which would hand that person
direct push access to master as a side effect.

**PyPI publishing uses Trusted Publishing.** The `publish_pypi` input defaults
to false, so a release never depends on credentials. When `publish_pypi` is true,
the workflow publishes to PyPI using PyPI Trusted Publishing via OIDC
(`id-token: write` and `uv publish --trusted-publishing always`), configured
for repository `krishdesai7/deconvolve` and package `deconvolve` on PyPI.
