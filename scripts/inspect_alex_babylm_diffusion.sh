#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

search_root=/home/atuin/c107fa/c107fa12
inventory="$RUN_DIR/babylm_diffusion_inventory.txt"
source_dump="$RUN_DIR/babylm_diffusion_sources.txt"
: >"$inventory"
: >"$source_dump"

find "$search_root" -maxdepth 4 -type d \
  \( -iname 'BabyLM' -o -iname '*babylm*' \) -print >"$inventory"

while IFS= read -r root; do
  [[ -d "$root" ]] || continue
  find "$root" -maxdepth 6 -type f \
    \( -iname '*.py' -o -iname '*.yaml' -o -iname '*.yml' -o -iname '*.json' \) \
    -print0 | while IFS= read -r -d '' file; do
      if [[ "$file" =~ [Mm][Dd][Ll][Mm]|[Ss][Ee][Dd][Dd]|[Dd]iffusion|[Nn]oise[_-]schedule ]]; then
        printf '\n===== %s =====\n' "$file" >>"$source_dump"
        sed -n '1,1200p' "$file" >>"$source_dump"
      elif rg -q -i 'mdlm|score entropy|sedd|masked diffusion|subs parameter' "$file"; then
        printf '\n===== %s =====\n' "$file" >>"$source_dump"
        sed -n '1,1200p' "$file" >>"$source_dump"
      fi
    done
done <"$inventory"

if [[ ! -s "$source_dump" ]]; then
  echo "No BabyLM diffusion source matched under $search_root" >&2
  exit 3
fi
