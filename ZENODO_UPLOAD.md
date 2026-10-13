# Zenodo Release Notes

This repository is prepared for archiving through the GitHub–Zenodo
integration.

Version `v1.0.0` is archived at
<https://doi.org/10.5281/zenodo.23334136>. The steps below apply to later
releases.

## Before the GitHub Release

1. Confirm the repository is clean:

   ```bash
   git status
   ```

2. Confirm that raw sequences, figures, model weights, documents, and archives
   are not tracked:

   ```bash
   git ls-files | grep -E '(^data/raw/|^outputs/|\.(npz|pt|pth|png|pdf|svg|tif|tiff|eps|zip|docx)$|^~\$)'
   ```

   This should print nothing.

3. Run the release validation:

   ```bash
   python scripts/validate_release.py
   pytest -q
   ```

4. Review `CITATION.cff` and `.zenodo.json`, especially the creator list,
   title, version, and release date.

5. Create and push the version tag:

   ```bash
   git tag -a v1.0.0 -m "Version 1.0.0"
   git push origin main --tags
   ```

## Zenodo GitHub Integration

1. Sign in at <https://zenodo.org/> using the GitHub account linked to
   `simonator420`.
2. Open <https://zenodo.org/account/settings/github/>, click **Sync now**, find
   `fitrec-personalization`, and enable the repository.
3. In GitHub, create a release from tag `v1.0.0` with title `Version 1.0.0`.
4. Wait for Zenodo to process and archive the release. Open the resulting DOI
   record and check the title, creators, version, license, description,
   repository link, file list, and archival status under **External
   resources**.
5. Copy the release DOI into the manuscript. If later releases are expected,
   use the concept DOI where a citation should always resolve to the newest
   version and the version DOI where the exact archived release matters.

The GitHub release is ingested automatically once the repository is enabled.
Therefore, approve the creator list and metadata before creating the GitHub
release. Because both metadata files are present, Zenodo uses `.zenodo.json`
for the archive; keep `CITATION.cff` synchronized for GitHub and users.

## Manual Upload Alternative

If the GitHub integration is unavailable, download the GitHub release source
ZIP and upload it as a Zenodo **Software** record. Use version `v1.0.0`, license
`MIT`, and the metadata in `.zenodo.json`.
