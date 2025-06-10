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
    skip_single_plot_shows: bool = True,
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
        if not skip_single_plot_shows:
            cm = confusion_matrix(true, rounded_pred, labels=range(2, 33))  # 2 to 32
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(2, 33))
            disp.plot(xticks_rotation=90, cmap='Blues')
            if set_title:
                plt.title('Confusion Matrix of Piece Count')
            plt.show()

        errors = np.array(pred) - np.array(true)
        if not skip_single_plot_shows:
            plt.hist(errors, bins=20, edgecolor='black')
            plt.xlabel('Prediction Error (Pred - True)')
            plt.ylabel('Frequency')
            if set_title:
                plt.title('Histogram of Prediction Errors')
            plt.grid(True)
            plt.show()

        if not skip_single_plot_shows:
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

    # Maybe compare results?
    # E.g., make a bars plot with different models/parameters

    # Bar plot to compare models
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
        ax.legend(title="Models")
        ax.grid(True, axis='y')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    effnet = lambda num: f"results-numpieces_final_effnetv2s{num}"
    resnet = lambda num: f"results-numpieces_final{num}"

    model_result_dirs = [resnet(i) for i in range(1, 6)] + \
        [f"results-numpieces3"] + \
        [f"results-numpieces_final_resnext1", f"results-numpieces_final_swinv2s1"] + \
        [effnet(i) for i in range(1, 6)] + \
        ["results-numpieces_final_effnetv2s6_smoothl1loss"] + \
        ["results-numpieces_final_effnetv2s7"]
    process_results(model_result_dirs, "valid", set_title=False)

    comparisons_architectures = [
        "results-numpieces_final2",
        "results-numpieces_final_resnext1",
        "results-numpieces_final_effnetv2s1",
        "results-numpieces_final_swinv2s1",
    ]
    process_results(comparisons_architectures, "valid", labels=["ResNet50", "ResNeXt", "EffNetV2-S", "SwinV2-S"], set_title=False)

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

    comparisons_augmentations = [effnet(1), effnet(2), effnet(8)]
    process_results(comparisons_augmentations, "valid", labels=["Manual", "Random", "Auto"], set_title=False)

    comparisons_activations = [effnet(2), effnet(4), effnet(3)]
    labels = ["Sigmoid ($30 \\cdot \\sigma + 2$)", "ReLU", "ReLU ($scaling \\cdot ReLU + bias$)"]
    process_results(comparisons_activations, "valid", labels=labels, set_title=False)

    comparisons_loss = [effnet(3), effnet(6) + "_smoothl1loss", effnet(7)]
    process_results(comparisons_loss, "valid", labels=["MSE", "SmoothL1Loss", "L1Loss"], set_title=False)

    best_model = effnet(6) + "_smoothl1loss"
    process_results([best_model], "valid", labels=["Best Model"], skip_single_plot_shows=False, set_title=False)

    # model_result_dirs = ["results-numpieces_final_effnetv2s3"]
    # process_results(model_result_dirs, "test")
