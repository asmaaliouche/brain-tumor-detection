import json

with open("notebooks/02_semi_supervised_approach.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "labels_true, preds_supervised = evaluate_model(" in source:
            new_source = source.replace(
                "print(\"==========================================================\")\nprint(\"               SUPERVISED BASELINE RESULTS                \")",
                """# --- MISCLASSIFICATION ANALYSIS ---\n\ntest_image_paths = df_test_strong['image_name'].tolist()\n\nprint(\"==========================================================\")\nprint(\"               MISCLASSIFIED IMAGES ANALYSIS              \")\nprint(\"==========================================================\")\nmisclass_sup = [test_image_paths[i] for i in range(len(labels_true)) if labels_true[i] != preds_supervised[i]]\nmisclass_semi = [test_image_paths[i] for i in range(len(labels_true)) if labels_true[i] != preds_semi[i]]\n\nprint(f\"Supervised Baseline Misclassified ({len(misclass_sup)} images): {misclass_sup}\")\nprint(f\"Semi-Supervised Misclassified ({len(misclass_semi)} images): {misclass_semi}\\n\")\n\nkmeans_test_errors = df_test_strong[df_test_strong['true_label'] != df_test_strong['weak_label']]['image_name'].tolist()\nprint(f\"K-Means Misclassified in Test Set ({len(kmeans_test_errors)} images): {kmeans_test_errors}\")\ncommon_errors = set(misclass_semi).intersection(set(kmeans_test_errors))\nprint(f\"Images misclassified by both Semi-Supervised CNN and K-Means: {common_errors}\\n\")\n\nprint(\"==========================================================\")\nprint(\"               SUPERVISED BASELINE RESULTS                \")"""
            )
            
            # Update the source lines
            cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in new_source.split("\n")]
            # Clean up the trailing empty lines
            if cell["source"][-1] == "\n":
                cell["source"].pop()
            break

with open("notebooks/02_semi_supervised_approach.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

