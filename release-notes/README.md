# Release notes

One file per Centrix release, published verbatim as its GitHub release description. The process is
Skyline-PRISM's.

## Versions

Centrix uses calendar versions, `YY.feature.patch`:

- **YY**: the two-digit year (`26` for 2026)
- **feature**: raised for each release with new features, centroiding changes or new options
- **patch**: raised for a release of bug fixes only

`26.1.0` is the first feature release of 2026 (and the first .NET release), `26.1.1` a patch to it,
`26.2.0` the next feature release. The Rust releases before it are under the tag `rust-final`.

The version is raised only at release time, never during development, so every build between two
releases reports the last release's number. A change that moves a centroid also raises
`CentrixCentroider.ALGORITHM_VERSION`, which is a separate number.

## Files

```text
release-notes/
  README.md                 this file
  RELEASE_NOTES_next.md     the draft for the next release
  RELEASE_NOTES_v26.1.0.md  one per release, named for its tag
```

## During development

Add an entry to `RELEASE_NOTES_next.md` with each change a user would notice, in the pull request that
makes it. The draft carries no version, so the release it becomes can still change (a planned patch
becomes a feature release when a feature lands).

Each file has this structure; leave out the sections with nothing in them:

```markdown
# Centrix v{version}

One-sentence summary of the release.

## New features

## Changes to centroids

## Bug fixes

## Breaking changes

## Known issues
```

"Changes to centroids" is for anything that moves a centroid on existing data (and raised
`ALGORITHM_VERSION`): users comparing runs across versions need to know. "Breaking changes" is for
anything that needs action from a user (a renamed or removed option, a changed default). "Known
issues" is for what does not work yet in this release.

Write in the past tense ("Added", "Fixed"), lead with what a user sees, give numbers where there are
some, and name options by their command-line flag.

## Releasing

1. Rename `RELEASE_NOTES_next.md` to `RELEASE_NOTES_v{version}.md`, set its heading, **delete every
   heading with nothing under it** (the file is published as it is), and start a new
   `RELEASE_NOTES_next.md` with the empty headings.
2. Set `<Version>` in `Directory.Build.props` to `{version}`.
3. Commit, open a pull request, and merge it to `main` once CI passes; the push to `main` builds and
   installs the packages.
4. Tag `main` and push the tag:

   ```bash
   git tag v{version}
   git push origin v{version}
   ```

The tag builds the packages and creates the GitHub release; never create it by hand. CI fails the tag,
before building anything, if it is not `v` + `<Version>` or its notes file is missing. A version with
`alpha`, `beta` or `rc` in it (`26.2.0-rc1`) is published as a prerelease.

To correct a published release's description:

```bash
gh release edit v{version} --notes-file release-notes/RELEASE_NOTES_v{version}.md
```
