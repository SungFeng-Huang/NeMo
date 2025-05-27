import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.colors import LogNorm

parser = argparse.ArgumentParser(description="Plot relationship between WER and VQScore.")
parser.add_argument("--subset", type=str, required=True, help="Subset to process (e.g., 'test', 'train').")
parser.add_argument(
    "--score_type",
    type=str,
    choices=["", "encoder_True", "encoder_False", "decoder_True", "decoder_False", "vq"],
    default="",
    help="Type of score to use for plotting."
)
parser.add_argument("--jsonl_file_path", type=str, required=True, help="Path to the JSONL file.")
parser.add_argument("--vq_file_path", type=str, required=False, help="Path to the VQScore file (required if --use_vq_score is set).")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the plots.")
args = parser.parse_args()

subset = args.subset

# 定義文件路徑
jsonl_file_path = args.jsonl_file_path
use_vq_score = args.score_type == "vq"
if use_vq_score:
    if args.vq_file_path is None:
        raise ValueError("VQScore file path must be provided when using VQScore.")
    vq_file_path = args.vq_file_path
output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 初始化數據存儲
audio_files = []
wer_scores = []
scores = []

# 讀取 JSONL 文件
with open(jsonl_file_path, "r", encoding="utf-8") as file:
    for line in file.readlines():
        line = line.strip()  # 確保行不為空
        data = json.loads(line)
        audio_files.append(data["audio_filepath"])
        wer_scores.append(data["wer"])
        if not use_vq_score:
            # 如果不使用 VQScore，則將其設置為 None 或 0
            scores.append(data[f"score_{args.score_type}" if args.score_type else "score"])

if use_vq_score:
    with open(vq_file_path, "r", encoding="utf-8") as file:
        for i, line in enumerate(file.readlines()):
            line = line.strip()  # 確保行不為空
            data = json.loads(line)
            assert audio_files[i] == data["audio_filepath"]
            scores.append(float(data["VQScore"]))  # 替換為實際字段名稱

wer_scores = np.array(wer_scores)
scores = np.array(scores)
print(len(wer_scores), wer_scores)
print(len(scores), scores)

# 過濾非有限值
valid_indices = np.isfinite(wer_scores) & np.isfinite(scores)
if not use_vq_score:
    valid_indices &= (scores != 0)  # 過濾掉分數為 0 的情況
wer_scores = wer_scores[valid_indices]
scores = scores[valid_indices]

score_name = f"Score_{args.score_type}" if args.score_type else "Score"
_score_name = f"score_{args.score_type}" if args.score_type else "score"

# 繪製散點圖
plt.figure(figsize=(10, 6))
plt.scatter(wer_scores, scores, alpha=0.7, color="blue")
plt.title(f"WER vs {score_name} ({subset})")
plt.xlabel("WER (Word Error Rate)")
plt.ylabel(score_name)
plt.grid(True)
plt.savefig(f"{output_dir}/wer_{_score_name}_scatter_{subset}.png")
plt.show()
plt.close()

# 過濾 WER 在 0 到 1 之間的數據
filtered_indices = (wer_scores >= 0) & (wer_scores <= 1)
wer_scores = wer_scores[filtered_indices]
scores = scores[filtered_indices]

# 繪製 2D 直方圖（使用對數刻度頻率）
plt.figure(figsize=(10, 6))
hist, xedges, yedges, im = plt.hist2d(wer_scores, scores, bins=100, cmap="Blues", norm=LogNorm())
plt.colorbar(im, label="Log-Frequency")
plt.title(f"2D Histogram of WER and {score_name} (Log-Scale) ({subset})")
plt.xlabel("WER (Word Error Rate)")
plt.ylabel(score_name)
plt.grid(True)
plt.savefig(f"{output_dir}/wer_{_score_name}_2d_histogram_log_{subset}.png")
plt.show()
plt.close()

# 使用 WER 和 Score 數據
x = wer_scores
y = scores

# 計算相關係數矩陣
data = np.vstack([x, y])
correlation_matrix = np.corrcoef(data)

# 繪製熱圖
plt.figure(figsize=(6, 5))
plt.imshow(correlation_matrix, cmap="coolwarm", interpolation="nearest")
plt.colorbar()
plt.xticks(ticks=[0, 1], labels=["WER", score_name])
plt.yticks(ticks=[0, 1], labels=["WER", score_name])
plt.title("Correlation Heatmap", fontsize=14)
for i in range(correlation_matrix.shape[0]):
    for j in range(correlation_matrix.shape[1]):
        plt.text(j, i, f"{correlation_matrix[i, j]:.2f}", ha="center", va="center", color="black")
plt.savefig(f"{output_dir}/wer_{_score_name}_correlation_heatmap_{subset}.png")
plt.show()
