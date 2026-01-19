import zipfile
import os

zip_path = "submission.zip"
model_path = "model.py"
if not os.path.exists(zip_path):
    raise SystemExit(f"Zip file not found: {zip_path}")
if not os.path.exists(model_path):
    raise SystemExit(f"model.py not found in workspace: {model_path}")

out_zip = "submission_updated.zip"
with zipfile.ZipFile(zip_path, 'r') as zin:
    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if os.path.basename(item.filename) == 'model.py':
                continue
            data = zin.read(item.filename)
            zout.writestr(item, data)
        with open(model_path, 'rb') as f:
            data = f.read()
        zout.writestr('model.py', data)

print(f"Wrote updated zip to {out_zip}")
