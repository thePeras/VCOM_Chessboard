import pickle, os

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, root_mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def load_results(filename):
    with open(f"{filename}.pkl", "rb") as f:
        data = pickle.load(f)
    return data["preds"], data["true"]

def process_results(result_dirs: list[str], results_type: str):
    """
    Args:
        result_dirs (list[str]): The model's result directories to process
        results_type (str): The results type: validation or test
    """
    for result_dir in result_dirs:
        pred, true = load_results(os.path.join(result_dir, results_type))
        rounded_pred = [round(p) for p in pred]

        cm = confusion_matrix(true, rounded_pred, labels=range(2, 33))  # 2 to 32
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(2, 33))
        disp.plot(xticks_rotation=90, cmap='Blues')
        plt.title('Confusion Matrix of Piece Count')
        plt.show()

        errors = np.array(pred) - np.array(true)

        plt.hist(errors, bins=20, edgecolor='black')
        plt.xlabel('Prediction Error (Pred - True)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Prediction Errors')
        plt.grid(True)
        plt.show()
        
        plt.scatter(true, pred, alpha=0.6)
        plt.plot([2, 32], [2, 32], 'r--')  # y=x line for perfect predictions
        plt.xlabel('True Number of Pieces')
        plt.ylabel('Predicted Number of Pieces')
        plt.title('True vs. Predicted Piece Count')
        plt.grid(True)
        plt.show()
        
        mae = mean_absolute_error(true, pred)
        rmse = root_mean_squared_error(true, pred)
        acc_within_1 = np.mean(np.abs(np.array(pred) - np.array(true)) <= 1)
        acc_with_rounded = accuracy_score(true, rounded_pred)
        print(f'MAE: {mae:.2f}')
        print(f'RMSE: {rmse:.2f}')
        print(f'Accuracy within ±1: {acc_within_1:.2%}')
        print(f"Accuracy with rounded values: {acc_with_rounded:.2%}")


if __name__ == "__main__":
    model_result_dirs = ["results-numpieces3"]
    # process_results(model_result_dirs, "valid")
    process_results(model_result_dirs, "test")
