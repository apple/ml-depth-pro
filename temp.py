import json
import os

input_file = "/dataset/sharedir/research/Hypersim/new_final_valid_files.txt"

output_file = "/dataset/sharedir/research/Hypersim/valid_files.txt"

cnt = 0
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        # 解析路径，移除多余的目录
        cnt += 1
        parts = line.strip().split("/")
        # 替换路径中第二个 `Hypersim` 为 `data`
        parts[5] = "data"
        # 移除多余的 `001/002` 目录（索引6和7）
        new_path = "/".join(parts[:6] + parts[8:])
        # 写入到输出文件
        outfile.write(new_path + "\n")

valid_cnt = 0
# Check if every line exists in the output file
with open(output_file, "r", encoding="utf-8") as infile:
    for line in infile:
        if not os.path.exists(line.strip()):
            print(f"File not found: {line.strip()}")
        else:
            valid_cnt += 1

print(f"Total: {cnt}, Valid: {valid_cnt}")

output_json = "/dataset/sharedir/research/Hypersim/valid_files.json"

json_data = []

with open(output_file, "r", encoding="utf-8") as infile:
    for idx, line in enumerate(infile):
        # 获取图像路径
        img_path = line.strip()
        # 生成深度图路径
        depth_path = img_path.replace("_final_hdf5", "_geometry_hdf5", 1).replace(".color.hdf5", ".depth_meters.hdf5",
                                                                                  1)

        entry = {
            "id": idx + 1,  # ID 升序，从 1 开始
            "img_path": img_path,
            "depth_path": depth_path
        }
        # 添加到 JSON 数据列表
        json_data.append(entry)

# 写入到 JSON 文件
with open(output_json, "w", encoding="utf-8") as outfile:
    json.dump(json_data, outfile)
    outfile.write("\n")
