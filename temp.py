import os

input_file = "/dataset/sharedir/research/Hypersim/new_final_valid_files.txt"

output_file = "/dataset/sharedir/research/Hypersim/valid_files.txt"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        # 解析路径，移除多余的目录
        parts = line.strip().split("/")
        # 替换路径中第二个 `Hypersim` 为 `data`
        parts[5] = "data"
        # 移除多余的 `001/002` 目录（索引6和7）
        new_path = "/".join(parts[:6] + parts[8:])
        # 写入到输出文件
        outfile.write(new_path + "\n")

# Check if every line exists in the output file
with open(output_file, "r", encoding="utf-8") as infile:
    for line in infile:
        if not os.path.exists(line.strip()):
            print(f"File not found: {line.strip()}")
