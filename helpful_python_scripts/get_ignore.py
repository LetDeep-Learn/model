import os

# Set your root directory here
root_dir = r"C:\Users\Omkar.Sonawale\Desktop\model"  # Change this to your actual root

# Get only visible subdirectories directly under root
subdirs = [
    name + "/"  # Add trailing slash for .gitignore
    for name in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, name)) and not name.startswith(".")
]

# Write to .gitignore
with open(".gitignore", "w", encoding="utf-8") as f:
    for d in sorted(subdirs):
        f.write(d + "\n")

print(f".gitignore created with {len(subdirs)} top-level directories.")