import os

# Move up one directory to access the Dataset folder
dataset_path = "Dataset"  # Go one level up from CNN_Model to CHATHUMA, then access Dataset

for subset in ["Training", "Testing"]:
    subset_path = os.path.join(dataset_path, subset)
    print(f"{subset} contains:")
    for category in os.listdir(subset_path):
        category_path = os.path.join(subset_path, category)
        print(f"  {category}: {len(os.listdir(category_path))} images")
