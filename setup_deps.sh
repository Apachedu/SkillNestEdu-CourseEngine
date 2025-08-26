set -e
python -m pip install --upgrade pip
TMP=.req.tmp
cat > "$TMP" <<'R'
streamlit
plotly
sympy
pint
numpy
pandas
pillow
scikit-learn
joblib
PyMuPDF
R
touch requirements.txt
cat requirements.txt "$TMP" | sed '/^\s*$/d' | awk '{key=tolower($0); if(!seen[key]++){print}}' > .req.merged
mv .req.merged requirements.txt
rm -f "$TMP"
pip install -r requirements.txt
