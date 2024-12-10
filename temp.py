import os

txt_file = "/dataset/sharedir/research/Hypersim/new_final_valid_files.txt"

output_file = "/dataset/sharedir/research/Hypersim/valid_files.txt"

with open(txt_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        # 替换第二个 'Hypersim' 为 'data'
        updated_line = line.replace("/Hypersim/Hypersim/", "/Hypersim/data/", 1)
        # 写入到输出文件
        outfile.write(updated_line)

# Check if every line exists in the output file
with open(output_file, "r", encoding="utf-8") as infile:
    for line in infile:
        if not os.path.exists(line.strip()):
            print(f"File not found: {line.strip()}")
