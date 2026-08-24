import json

with open("notebooks/02_semi_supervised_approach.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "train_transform = transforms.Compose([" in source:
            new_source = source.replace(
                "transforms.RandomRotation(degrees=15), \n",
                "transforms.RandomRotation(degrees=15), \n    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Ajout pour l'imagerie médicale\n    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Légères déformations\n"
            )
            
            # Update the source lines
            cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in new_source.split("\n")]
            # Clean up the trailing empty lines
            if cell["source"][-1] == "\n":
                cell["source"].pop()
            break

with open("notebooks/02_semi_supervised_approach.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

