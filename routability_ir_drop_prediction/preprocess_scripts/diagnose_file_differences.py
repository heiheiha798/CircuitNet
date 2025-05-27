import os
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose file differences across subdirectories.")
    parser.add_argument("--data_path", required=True, help="Path to the dataset directory")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 定义需要检查的所有数据子文件夹
    sub_dirs = [
        'macro_region', 
        'RUDY', 
        'RUDY_pin',
        'congestion_GR_horizontal_overflow',
        'congestion_GR_vertical_overflow'
    ]

    print("--- Starting File Discrepancy Diagnosis ---")

    # 1. 读取每个文件夹的文件列表
    file_sets = {}
    for sub_dir in sub_dirs:
        dir_path = os.path.join(args.data_path, sub_dir)
        try:
            # 获取所有文件名，并去除.npy后缀（如果有）
            files = set(os.path.splitext(f)[0] for f in os.listdir(dir_path))
            file_sets[sub_dir] = files
            print(f"Found {len(files)} files in directory: {sub_dir}")
        except FileNotFoundError:
            print(f"Error: Directory not found at {dir_path}. Exiting.")
            return
            
    # 2. 计算所有文件的并集，即所有出现过的文件
    all_files_union = set.union(*file_sets.values())
    print(f"\nTotal unique filenames across all directories: {len(all_files_union)}")
    print("--- Discrepancy Report ---")

    # 3. 检查每个文件夹相对于并集缺失了哪些文件
    all_complete = True
    for sub_dir in sub_dirs:
        missing_files = all_files_union - file_sets[sub_dir]
        
        if not missing_files:
            print(f"\n✅ Directory '{sub_dir}' is complete.")
        else:
            all_complete = False
            print(f"\n❌ Directory '{sub_dir}' is missing {len(missing_files)} files:")
            # 为了方便查看，只打印前10个缺失的文件名
            missing_list = sorted(list(missing_files))
            for i, filename in enumerate(missing_list[:10]):
                print(f"   - {filename}")
            if len(missing_list) > 10:
                print(f"   ... and {len(missing_list) - 10} more.")

    print("\n--- Diagnosis Summary ---")
    if all_complete:
        print("✅ All directories have identical file lists. Your dataset seems consistent.")
    else:
        print("❌ Inconsistency found. The report above lists the missing files for each directory.")
        print("This confirms that a copy or unzip error likely occurred.")

if __name__ == '__main__':
    main()