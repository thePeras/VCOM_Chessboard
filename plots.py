import pickle, os

import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, root_mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

def load_results(filename):
    with open(f"{filename}.pkl", "rb") as f:
        data = pickle.load(f)
    return data["preds"], data["true"]

def process_results(
    result_dirs: list[str],
    results_type: str,
    *,
    labels: Optional[str] = None,
    legend_label: Optional[str] = None,
    show_single_plots: bool = False,
    set_title: bool = True,
):
    """
    Args:
        result_dirs (list[str]): The model's result directories to process
        results_type (str): The results type: validation or test
    """
    if labels is None:
        labels = result_dirs
    assert(len(labels) == len(result_dirs))

    all_metrics = []
    for model_label, result_dir in zip(labels, result_dirs):
        pred, true = load_results(os.path.join(result_dir, results_type))
        if isinstance(pred, torch.Tensor):
            pred = pred.tolist()
        if isinstance(true, torch.Tensor):
            true = true.tolist()

        rounded_pred = [
            round(p.item()) if isinstance(p, torch.Tensor) else round(p)
            for p in pred
        ]
        if show_single_plots:
            cm = confusion_matrix(true, rounded_pred, labels=range(2, 33))  # 2 to 32
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(2, 33))
            disp.plot(xticks_rotation=90, cmap='Blues')
            if set_title:
                plt.title('Confusion Matrix of Piece Count')
            plt.show()

        errors = np.array(pred) - np.array(true)
        if show_single_plots:
            plt.hist(errors, bins=20, edgecolor='black')
            plt.xlabel('Prediction Error (Pred - True)')
            plt.ylabel('Frequency')
            if set_title:
                plt.title('Histogram of Prediction Errors')
            plt.grid(True)
            plt.show()

        if show_single_plots:
            plt.scatter(true, pred, alpha=0.6)
            plt.plot([2, 32], [2, 32], 'r--')  # y=x line for perfect predictions
            plt.xlabel('True Number of Pieces')
            plt.ylabel('Predicted Number of Pieces')
            if set_title:
                plt.title('True vs. Predicted Piece Count')
            plt.grid(True)
            plt.show()

        mae = mean_absolute_error(true, pred)
        rmse = root_mean_squared_error(true, pred)
        acc_within_1 = np.mean(np.abs(np.array(pred) - np.array(true)) <= 1)
        acc_with_rounded = accuracy_score(true, rounded_pred)

        print(f">> Result {model_label}")
        print(f'MAE: {mae:.2f}')
        print(f'RMSE: {rmse:.2f}')
        print(f'Accuracy within ±1: {acc_within_1:.2%}')
        print(f"Accuracy with rounded values: {acc_with_rounded:.2%}")

        all_metrics.append({
            "model": model_label,
            "MAE ($\\downarrow$)": mae,
            "RMSE ($\\downarrow$)": rmse,
            "Acc $\\pm$ 1 ($\\uparrow$)": acc_within_1,
            "Rounded Acc ($\\uparrow$)": acc_with_rounded
        })

    # Bar plot to compare models/parameters
    if len(all_metrics) > 1:
        metric_names = ["MAE ($\\downarrow$)", "RMSE ($\\downarrow$)", "Acc $\\pm$ 1 ($\\uparrow$)", "Rounded Acc ($\\uparrow$)"]
        num_metrics = len(metric_names)
        num_models = len(all_metrics)
        
        width = 0.15  # Width of each bar
        group_width = width * num_models + 0.05  # Include padding
        x = np.arange(num_metrics) * group_width  # Space out groups based on model count

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, metric_set in enumerate(all_metrics):
            values = [metric_set[m] for m in metric_names]
            offset = i * width
            bars = ax.bar(x + offset, values, width, label=metric_set["model"])

            # Add annotations
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

        ax.set_ylabel('Score')
        if set_title:
            ax.set_title('Model Comparison by Metrics')
        ax.set_xticks(x + (num_models - 1) * width / 2)
        ax.set_xticklabels(metric_names)
        if legend_label is not None:
            ax.legend(title=legend_label)
        ax.grid(True, axis='y')
        plt.tight_layout()
        plt.show()

"""
Efficient Net models
1 - manual augmentations
2 - RandAugment
3 - forward relu with learnable scale and bias
4 - forward relu (direct)
5 - (unused) forward relu with learnable scale and bias, AutoAugment
6 - smooth l1 loss
7 - l1 loss
8 - AutoAugment, sigmoid
"""

if __name__ == "__main__":
    effnet = lambda num: f"results-numpieces_final_effnetv2s{num}"
    resnet = lambda num: f"results-numpieces_final{num}"
    model_result_dirs = [resnet(i) for i in range(1, 7)]
    process_results(model_result_dirs, "valid", set_title=False)
    # model_result_dirs = [resnet(i) for i in range(1, 6)] + \
    #     [f"results-numpieces3"] + \
    #     [f"results-numpieces_final_resnext1", f"results-numpieces_final_swinv2s1"] + \
    #     [effnet(i) for i in range(1, 6)] + \
    #     ["results-numpieces_final_effnetv2s6_smoothl1loss"] + \
    #     ["results-numpieces_final_effnetv2s7"] + \
    #     ["results-best_model"]
    # process_results(model_result_dirs, "valid", set_title=False)

    # comparisons_architectures = [
    #     "results-numpieces_final2",
    #     "results-numpieces_final_resnext1",
    #     "results-numpieces_final_effnetv2s1",
    #     "results-numpieces_final_swinv2s1",
    # ]
    # labels = ["ResNet50", "ResNeXt", "EffNetV2-S", "SwinV2-S"]
    # process_results(comparisons_architectures, "valid", labels=labels, legend_label="Models", set_title=False)

    # comparisons_augmentations = [effnet(1), effnet(2), effnet(8)]
    # labels = ["Manual", "Random", "Auto"]
    # process_results(comparisons_augmentations, "valid", labels=labels, legend_label="Augmentations", set_title=False)

    # comparisons_activations = [effnet(2), effnet(4), effnet(3)]
    # labels = ["Sigmoid ($30 \\cdot \\sigma + 2$)", "ReLU", "ReLU ($scaling \\cdot ReLU + bias$)"]
    # process_results(comparisons_activations, "valid", labels=labels, legend_label="Activations", set_title=False)

    comparisons_loss = [effnet(3), effnet(6) + "_smoothl1loss", effnet(7)]
    labels = ["MSE", "SmoothL1Loss", "L1Loss"]
    # process_results(comparisons_loss, "valid", labels=labels, legend_label="Losses", set_title=False)

    best_model = "results-best_model"
    process_results([best_model], "test", labels=["Best Model"], show_single_plots=True, set_title=False)

    process_results(comparisons_loss + [best_model], "valid", labels=labels + ["Best Model"], legend_label="Losses", set_title=False)

    # process_results(
    #     [effnet(6) + "_smoothl1loss", "results-best_model"],
    #     "valid",
    #     labels=["Previous Best", "Best Model (Tuned)"],
    #     legend_label="Models",
    #     set_title=False,
    # )
