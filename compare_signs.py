# compare_signs.py
import os, cv2, matplotlib.pyplot as plt

base = "data/raw/asl_alphabet/asl_alphabet_train/asl_alphabet_train"
labels_to_check = ["M", "N", "del"]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, label in zip(axes, labels_to_check):
    class_dir = os.path.join(base, label)
    img_path = os.path.join(class_dir, os.listdir(class_dir)[0])
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.set_title(label)
    ax.axis("off")
plt.tight_layout()
plt.savefig("compare_signs.png")
print("Saved compare_signs.png")