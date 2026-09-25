# Anonymous release checklist

The automated audit checks the public artifact's contents and provenance. It
does not determine whether a hosting account, repository URL, issue discussion,
or external website reveals an author's identity. Review those surfaces before
sharing a submission link.

## 1. Review the files and scientific coverage

- Include every implementation, configuration, data-preparation tool, guide,
  aggregate result and replicate-count table listed in `docs/PAPER_ARTIFACT_MAP.md`.
- Include primary spectral MIRT, direct logistic MIRT, gap and population
  sensitivity, item-signature stability, content audit, discrimination matching,
  within-family specificity, all capability-profile matching scenarios, and the
  maintainer quick-start tool.
- Preserve the null and limited-overlap capability-profile results alongside
  significant results. Coverage and matching distances are part of the result.
- Include numerical annotation labels only with fresh release-local
  `record_0000`, `pair_0000`, and `cell_0000` identifiers. The mapping to original
  item identifiers, prompt text, provider transcripts, and free-text rationales
  must remain outside the artifact. The labels file is the sole permitted CSV
  exception for `blinded_id` and `pair_id` columns.
- Keep response matrices, licensed raw data, model/item mappings, API credentials,
  caches, logs, checkpoints and local paths outside the distributed artifact.
  See `docs/DATA.md` for acquisition and redistribution boundaries.

## 2. Test scientific reproducibility

Use the pinned environment and prevent local cache files in the release tree:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
```

`tests/test_artifact_integrity.py` independently pools integer reversal counts
from all published random replicates and recomputes primary/direct-MIRT,
discrimination, within-family specificity, capability-profile paired contrasts,
and gap-threshold summaries. The remaining tests cover estimators, dimensions,
weights, matching, content-label replay, maintainer input handling, and plotting.
Do not treat the number of test functions as proof that the scientific results
were reproduced; record the actual test outcome and any full-data reruns.

## 3. Review anonymity

- Review README, code comments, configuration values, citation files, figures,
  PDF metadata, filenames and archives for names, affiliations and local paths.
- Check external links: public upstream datasets must be distinguishable from
  author-owned repositories, model hubs, experiment dashboards and private
  infrastructure. A public source citation need not be removed merely because
  the source has an owner.
- If Git history is included in the repository, use `Anonymous` for both author
  and committer and a neutral email, such as `anonymous@example.org`. The audit
  checks every reachable commit; `--publisher-check`, `--refresh`, and ZIP
  packaging also check local identity configuration. Reader verification does
  not require changing the reader's Git identity. A new commit
  with an anonymous name does not anonymize existing history.
- A personal account URL still identifies the account owner even when commit
  names are anonymous. Inspect the reviewer-facing URL and repository profile
  separately; the automated report makes no claim about hosting anonymity.

Supply any private identifiers as command-line exclusions without storing them
in public configuration files. Repeat `--deny-term` for author names, account
handles, affiliations, internal hostnames, or prohibited development markers.
The report records exclusion numbers and affected relative paths, never the
private search terms or matching text; terms in filenames are redacted too.

## 4. Verify the existing manifest without changing it

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/audit_release.py --verify --publisher-check
```

`--verify` is the default. Missing, added, removed or modified public files make
verification fail. Verification never rewrites the manifest or audit report,
and a missing manifest does not trigger automatic creation.

The audit parses all Python files, checks required module/configuration/result
coverage, scans CSV columns and text for common identity/credential leaks, and
checks PDF properties when `pypdf` is installed. The JSON output reports whether
PDF metadata inspection was available. Optional private checks can be added,
for example after setting local environment variables:

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/audit_release.py --verify \
  --deny-term "$PRIVATE_NAME" \
  --deny-term "$PRIVATE_AFFILIATION" \
  --deny-term "$PRIVATE_DEVELOPMENT_MARKER"
```

Do not pass empty values. The generic scanner cannot know every affiliation or
private identifier without those exclusions.

## 5. Explicitly refresh only after reviewing intended changes

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/audit_release.py --refresh
PYTHONDONTWRITEBYTECODE=1 python tools/audit_release.py --verify
```

`--refresh` replaces `provenance/public_files.sha256` and
`provenance/release_audit.json` only when the content audit passes. It does not
repair scientific outputs or suppress failed checks. Pass the same private
exclusions when refreshing if those checks are required for the release.
The manifest and audit report are excluded from their own checksums to avoid
recursive hashes, but their contents are still scanned.

Local `.git`, virtual environments, package metadata, `build/`, `dist/`, and
regenerated `outputs/` are excluded from the public inventory. They must not
be included in the submission ZIP. Bytecode and test caches elsewhere in the
public tree are rejected and should be removed before packaging.

## 6. Inspect the actual ZIP

Create the archive from the reviewed public inventory, excluding local
operational directories. Then verify it against the working-tree manifest:

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/audit_release.py --verify \
  --zip path-to-submission.zip
```

The ZIP may contain one enclosing directory. Legitimate hidden files
`.gitignore`, `.github/`, and `.env.example` are allowed. Git metadata, virtual
environments, raw matrices, logs, caches, `.DS_Store`, `__MACOSX`, path traversal,
symbolic links, duplicate file entries, and any content differing from the
audited public tree are rejected. The embedded manifest must match too.
Add the private exclusions to this command to check the archive payload.

## 7. Record release decisions

- Record the tested dependency versions and audit outcome.
- Include the MIT `LICENSE` and preserve the third-party exclusions in the
  README. Verify that built packages include the license and identify it as
  MIT; this does not grant rights to redistribute upstream benchmark data.
- Confirm that the reviewer-facing hosting URL preserves anonymity.
- After the review period, update authorship/citation information deliberately
  and archive an immutable, versioned release.
