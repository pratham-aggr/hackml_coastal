import zipfile
import os

zip_path = "submission.zip"
model_path = "model.py"
if not os.path.exists(zip_path):
    raise SystemExit(f"Zip file not found: {zip_path}")
if not os.path.exists(model_path):
    raise SystemExit(f"model.py not found in workspace: {model_path}")

tmp = zip_path + ".tmp"
with zipfile.ZipFile(zip_path, 'r') as zin:
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            # skip top-level model.py if present
            if os.path.basename(item.filename) == 'model.py':
                continue
            data = zin.read(item.filename)
            zout.writestr(item, data)
        # add the current workspace model.py at top-level
        with open(model_path, 'rb') as f:
            data = f.read()
        zout.writestr('model.py', data)

# replace original zip
os.replace(tmp, zip_path)
print(f"Updated {zip_path} with workspace model.py")
