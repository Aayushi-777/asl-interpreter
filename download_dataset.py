import kagglehub, shutil, os
path = kagglehub.dataset_download("grassknoted/asl-alphabet")
print("Downloaded to:", path)
os.makedirs("data/raw", exist_ok=True)
dest = "data/raw/asl_alphabet"
if os.path.exists(dest):
    shutil.rmtree(dest)
shutil.move(path, dest)
print("Moved to:", dest)