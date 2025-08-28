#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs
: > logs/ocr_failures.log

find source_pdfs -type f -name "*.pdf" ! -name "*_clean.pdf" -print0 |
while IFS= read -r -d '' f; do
  out="${f%.*}_clean.pdf"
  echo "OCR: $f -> $out"
  ocrmypdf --force-ocr --rotate-pages --deskew --jobs 2 --language eng+fra "$f" "$out" \
    || { echo "FAIL: $f" >> logs/ocr_failures.log; continue; }
done

find source_pdfs -type f -name "*_clean.pdf" -print0 | \
xargs -0 -I {} python tools/content_bank.py ingest "{}"

python tools/content_bank.py build_index_all

echo "DONE. Failures (if any) in logs/ocr_failures.log"
