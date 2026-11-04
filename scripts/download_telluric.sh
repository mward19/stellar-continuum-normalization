#!/usr/bin/env bash
# Downloads the PypeIt PCA-compressed LBLRTM telluric grid used by
# Casey et al. (2026, ApJ 998:192). One 36 MiB file; no Slurm job needed.
set -euo pipefail

# The host comes from pypeit/data/s3_url.txt in the PypeIt repo and may change
base_url="https://s3-west.nrp-nautilus.io/pypeit/telluric/atm_grids"
filename="TellPCA_3000_10500_R120000.fits"
expected_size=37748160
out_dir="${TELLURIC_DIR:-$HOME/nobackup/autodelete/telluric}"

mkdir -p "$out_dir"
output="${out_dir}/${filename}"

if [[ -f "$output" ]] && [[ "$(stat -c %s "$output")" == "$expected_size" ]]; then
  printf '%s already downloaded\n' "$output"
  exit 0
fi

temporary="$(mktemp "${output}.XXXXXX")"
printf 'Downloading %s\n' "$filename"
if ! curl --fail --location --retry 3 --output "$temporary" "${base_url}/${filename}"; then
  rm -f -- "$temporary"
  printf 'Failed to download %s/%s\n' "$base_url" "$filename" >&2
  exit 1
fi

actual_size="$(stat -c %s "$temporary")"
if [[ "$actual_size" != "$expected_size" ]]; then
  rm -f -- "$temporary"
  printf 'Unexpected size %s (expected %s); the file may have changed upstream\n' "$actual_size" "$expected_size" >&2
  exit 1
fi

mv -f -- "$temporary" "$output"
printf 'Saved %s\n' "$output"
