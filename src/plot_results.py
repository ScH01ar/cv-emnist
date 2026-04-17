import json
import matplotlib.pyplot as plt
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def plot_history(run_names):
    plt.figure(figsize=(14, 6))
    
    # 子图1：Loss曲线
    plt.subplot(1, 2, 1)
    for run in run_names:
        history_path = PROJECT_ROOT / "runs" / run / "history.json"
        if not history_path.exists():
            print(f"找不到 {run} 的日志文件: {history_path}")
            continue
            
        with open(history_path, 'r', encoding='utf-8') as f:
            data = json.load(f)["history"]
            
        epochs = [d["epoch"] for d in data]
        train_loss = [d["train_loss"] for d in data]
        val_loss = [d["val_loss"] for d in data]
        
        plt.plot(epochs, train_loss, label=f'{run} (Train Loss)', linestyle='--')
        plt.plot(epochs, val_loss, label=f'{run} (Val Loss)')
        
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 子图2：Accuracy曲线
    plt.subplot(1, 2, 2)
    for run in run_names:
        history_path = PROJECT_ROOT / "runs" / run / "history.json"
        if not history_path.exists():
            continue
            
        with open(history_path, 'r', encoding='utf-8') as f:
            data = json.load(f)["history"]
            
        epochs = [d["epoch"] for d in data]
        train_acc = [d["train_accuracy"] for d in data]
        val_acc = [d["val_accuracy"] for d in data]
        
        plt.plot(epochs, train_acc, label=f'{run} (Train Acc)', linestyle='--')
        plt.plot(epochs, val_acc, label=f'{run} (Val Acc)')

    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    # 保存图片到本地
    save_path = PROJECT_ROOT / "runs" / "training_curves.png"
    plt.savefig(save_path, dpi=300)
    print(f"曲线图已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    # 在这里填入你想对比的实验 run_name
    runs_to_plot = ["cnn_baseline", "resnet_baseline"]
    plot_history(runs_to_plot)