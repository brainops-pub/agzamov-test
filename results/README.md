# Result storage policy

`results/` is the local output root for exploratory and publication-candidate
runs. Each run directory is immutable after completion and is intentionally
ignored by Git because raw provider envelopes can be hundreds of megabytes.

The retained historical comparison is published as a compact, checksummed
review corpus under:

`audit-packets/kqk-three-model-independent-review-20260726/`

That packet contains normalized full logs, calibration records, model profiles,
summaries, source hashes, the relevant harness snapshot, and independent-review
instructions. Local raw runs remain available for forensic work but are not the
recommended interface or a publication claim.

New experiments must use the named public interface:

```bash
agzamov chess calibrate --profile <profile-id> --output <calibration-dir>
agzamov chess run \
  --profile <profile-id> \
  --protocol kqk-random-legal-defender-v1 \
  --calibration-from <calibration-dir> \
  --output <run-dir> \
  --yes
```

Standalone launchers under `scripts/` are historical research surfaces.
