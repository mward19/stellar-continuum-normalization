#!/usr/bin/env bash
set -euo pipefail

base_url="https://archive.stsci.edu/hlsps/bosz/bosz2024/download_scripts"
script_dir="./scripts"

broadening=(rorig)
metallicity=(
  -2.50 -2.25 -2.00 -1.75 -1.50 -1.25 -1.00
  -0.75 -0.50 -0.25 +0.00 +0.25 +0.50 +0.75
)

for resolution in "${broadening[@]}"; do
  for metal in "${metallicity[@]}"; do
    # Encode plus signs in URLs while preserving them in local filenames.
    url_metal="${metal//+/%2B}"
    filename="hlsp_bosz_bosz2024_sim_${resolution}_m${metal}_v1_bulkdl.sh"
    url="${base_url}/hlsp_bosz_bosz2024_sim_${resolution}_m${url_metal}_v1_bulkdl.sh"
    output="${script_dir}/${filename}"
    temporary="$(mktemp "${output}.XXXXXX")"

    printf 'Downloading %s\n' "$filename"
    if ! curl --fail --location --retry 3 --output "$temporary" "$url"; then
      rm -f -- "$temporary"
      printf 'Failed to download %s\n' "$url" >&2
      exit 1
    fi
    mv -f -- "$temporary" "$output"
  done
done

printf 'Downloaded %d scripts to %s\n' \
  "$(( ${#broadening[@]} * ${#metallicity[@]} ))" "$script_dir"