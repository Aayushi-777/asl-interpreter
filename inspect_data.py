# inspect_data.py
import os, cv2, matplotlib.pyplot as plt
base = "data/raw/asl_alphabet/asl_alphabet_train/asl_alphabet_train"
classes = os.listdir(base)
print(f"{len(classes)} classes found:", sorted(classes))

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for ax, cls in zip(axes, classes[:5]):
    img_path = os.path.join(base, cls, os.listdir(os.path.join(base, cls))[0])
    ax.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    ax.set_title(cls); ax.axis("off")
plt.tight_layout()
plt.savefig("data_sample_check.png")
print("Saved data_sample_check.png")