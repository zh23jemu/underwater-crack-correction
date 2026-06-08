"""
对比两个模型的逐图评估 CSV，并自动挑选后续可视化诊断样本。

典型用途：
1. 输入旧模型和新模型的 `eval_per_image.csv`。
2. 按裂缝 EPE、Dice、边缘保持和 folding 指标计算逐图差异。
3. 自动导出三类样本：
   - 新模型明显更好的样本；
   - 旧模型明显更好的样本；
   - 两个模型都表现较差的共同失败样本。

输出文件可直接作为 `utils/export_crack_roi_visuals.py --image_list` 的输入，
用于生成 ROI 局部放大图，帮助判断问题来自坐标映射、恢复幅度、边缘约束
还是 folding/位移场可逆性。
"""

import argparse
import csv
import json
import os
from pathlib import Path


LOWER_IS_BETTER = {
    "crack_epe_px",
    "global_epe_px",
    "folding_rate",
}

HIGHER_IS_BETTER = {
    "warp_crack_dice",
    "crack_edge_fidelity",
    "global_edge_fidelity",
}


def read_csv_by_image(path):
    """读取逐图评估 CSV，并用图片文件名建立索引。"""
    rows = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image = row.get("image")
            if not image:
                continue
            parsed = {"image": image}
            for key, value in row.items():
                if key == "image":
                    continue
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows[image] = parsed
    return rows


def improvement(old_row, new_row):
    """
    计算新模型相对旧模型的综合改善分。

    分数越大代表新模型越好。这里不做复杂归一化，主要用于排序和挑样本：
    - EPE/folding 下降记为正向改善；
    - Dice/edge 上升记为正向改善；
    - 对 crack EPE 和 Dice 给更高权重，因为当前项目最关心裂缝主体复原。
    """
    crack_epe_gain = old_row["crack_epe_px"] - new_row["crack_epe_px"]
    global_epe_gain = old_row["global_epe_px"] - new_row["global_epe_px"]
    dice_gain = (new_row["warp_crack_dice"] - old_row["warp_crack_dice"]) * 100.0
    crack_edge_gain = (new_row["crack_edge_fidelity"] - old_row["crack_edge_fidelity"]) * 100.0
    folding_gain = (old_row["folding_rate"] - new_row["folding_rate"]) * 100.0
    return (
        1.0 * crack_epe_gain
        + 0.4 * global_epe_gain
        + 0.8 * dice_gain
        + 0.4 * crack_edge_gain
        + 0.4 * folding_gain
    )


def merged_rows(old_rows, new_rows):
    """合并两个 CSV 中共同存在的图片，并补充差异字段。"""
    images = sorted(set(old_rows) & set(new_rows))
    out = []
    for image in images:
        old = old_rows[image]
        new = new_rows[image]
        row = {
            "image": image,
            "score_new_minus_old": improvement(old, new),
        }
        for key in [
            "crack_epe_px",
            "global_epe_px",
            "warp_crack_dice",
            "crack_edge_fidelity",
            "global_edge_fidelity",
            "folding_rate",
            "crack_ratio",
        ]:
            row[f"old_{key}"] = old.get(key, "")
            row[f"new_{key}"] = new.get(key, "")
            if key in LOWER_IS_BETTER:
                row[f"delta_{key}_positive_is_better"] = old.get(key, 0.0) - new.get(key, 0.0)
            elif key in HIGHER_IS_BETTER:
                row[f"delta_{key}_positive_is_better"] = new.get(key, 0.0) - old.get(key, 0.0)
        # 共同失败样本优先关注新旧模型裂缝 EPE 都高、Dice 都低、folding 偏高的情况。
        row["joint_failure_score"] = (
            0.5 * old.get("crack_epe_px", 0.0)
            + 0.5 * new.get("crack_epe_px", 0.0)
            - 60.0 * min(old.get("warp_crack_dice", 0.0), new.get("warp_crack_dice", 0.0))
            + 40.0 * max(old.get("folding_rate", 0.0), new.get("folding_rate", 0.0))
        )
        out.append(row)
    return out


def write_csv(path, rows):
    """写出 CSV；即使没有数据也保留表头，便于排查输入问题。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = ["image"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_image_list(path, rows):
    """写出图片文件名列表，每行一个文件名，供 ROI 可视化脚本复用。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['image']}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare old/new eval_per_image.csv and select diagnostic samples")
    parser.add_argument("--old_csv", required=True, help="旧模型 eval_per_image.csv")
    parser.add_argument("--new_csv", required=True, help="新模型 eval_per_image.csv")
    parser.add_argument("--out_dir", required=True, help="输出目录")
    parser.add_argument("--topk", type=int, default=20, help="每类挑选样本数量")
    args = parser.parse_args()

    old_rows = read_csv_by_image(args.old_csv)
    new_rows = read_csv_by_image(args.new_csv)
    rows = merged_rows(old_rows, new_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 新模型更好：综合改善分最高。
    new_better = sorted(rows, key=lambda x: x["score_new_minus_old"], reverse=True)[: args.topk]
    # 旧模型更好：综合改善分最低，即新模型相对退化最明显。
    old_better = sorted(rows, key=lambda x: x["score_new_minus_old"])[: args.topk]
    # 共同失败：两个模型在裂缝 EPE、Dice、folding 上都不理想。
    joint_failures = sorted(rows, key=lambda x: x["joint_failure_score"], reverse=True)[: args.topk]

    write_csv(str(out_dir / "all_compare.csv"), rows)
    write_csv(str(out_dir / "new_better.csv"), new_better)
    write_csv(str(out_dir / "old_better.csv"), old_better)
    write_csv(str(out_dir / "joint_failures.csv"), joint_failures)

    write_image_list(str(out_dir / "new_better_images.txt"), new_better)
    write_image_list(str(out_dir / "old_better_images.txt"), old_better)
    write_image_list(str(out_dir / "joint_failure_images.txt"), joint_failures)

    summary = {
        "old_csv": args.old_csv,
        "new_csv": args.new_csv,
        "matched_images": len(rows),
        "topk": args.topk,
        "mean_score_new_minus_old": sum(x["score_new_minus_old"] for x in rows) / max(len(rows), 1),
        "num_new_score_positive": sum(1 for x in rows if x["score_new_minus_old"] > 0),
        "num_new_score_negative": sum(1 for x in rows if x["score_new_minus_old"] < 0),
    }
    with open(out_dir / "compare_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== Compare Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
