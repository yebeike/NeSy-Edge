import pandas as pd
import numpy as np
import os

# --- 智能路径设置 (新代码) ---
# 获取脚本文件所在的目录 (e.g., .../NuSy-Edge/experiments)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 从脚本目录向上移动一级，得到项目根目录 (e.g., .../NuSy-Edge)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# 将输出文件名和项目根目录拼接成一个绝对路径
OUTPUT_FILENAME = os.path.join(PROJECT_ROOT, "experiments/showloghead_output.txt")
# --- 路径设置结束 ---


# --- 全局设置 ---
pd.set_option('display.max_colwidth', None)

# ★★★ 你需要配置的文件列表 ★★★
# 所有文件路径现在都应该相对于项目根目录
FILES_TO_PROCESS = [
    # --- HDFS v1 ---
    {'title': "HDFS: anomaly_label.csv", 'path': 'data/raw/HDFS_v1/preprocessed/anomaly_label.csv'},
    {'title': "HDFS: Event_occurrence_matrix.csv", 'path': 'data/raw/HDFS_v1/preprocessed/Event_occurrence_matrix.csv'},
    {'title': "HDFS: Event_traces.csv", 'path': 'data/raw/HDFS_v1/preprocessed/Event_traces.csv'},
    {'title': "HDFS: HDFS.npz (Numpy Archive)", 'path': 'data/raw/HDFS_v1/preprocessed/HDFS.npz'},
    {'title': "HDFS: HDFS.log", 'path': 'data/raw/HDFS_v1/HDFS.log', 'options': {'header': None}},

    # --- OpenStack 2 ---
    {'title': "OpenStack: anomaly_labels.txt", 'path': 'data/raw/OpenStack_2/anomaly_labels.txt'},
    {'title': "OpenStack: openstack_abnormal.log", 'path': 'data/raw/OpenStack_2/openstack_abnormal.log', 'options': {'header': None}},
    {'title': "OpenStack: openstack_normal1.log", 'path': 'data/raw/OpenStack_2/openstack_normal1.log', 'options': {'header': None}},
    {'title': "OpenStack: openstack_normal2.log", 'path': 'data/raw/OpenStack_2/openstack_normal2.log', 'options': {'header': None}},
]


def process_file(file_info, output_file_handle):
    """根据文件信息处理单个文件，并将结果写入打开的文件句柄。"""
    
    # 路径拼接现在基于项目根目录
    path = os.path.join(PROJECT_ROOT, file_info['path'])
    title = file_info['title']
    
    output_file_handle.write(f"--- {title} ---\n")
    
    if not os.path.exists(path):
        output_file_handle.write(f"错误：文件未找到 -> {path}\n\n")
        return

    try:
        if path.endswith('.npz'):
            data = np.load(path, allow_pickle=True)
            output_file_handle.write(f"文件包含的数组: {data.files}\n")
            for key in data.files:
                output_file_handle.write(f"\n数组 '{key}' (前5个元素):\n")
                output_file_handle.write(str(data[key][:5]) + "\n")
        else:
            read_options = file_info.get('options', {})
            read_options['nrows'] = 5
            df = pd.read_csv(path, **read_options)
            output_file_handle.write(df.to_string() + "\n")
            
    except Exception as e:
        output_file_handle.write(f"处理文件时发生错误: {e}\n")
    
    output_file_handle.write("\n\n")


# --- 主程序逻辑 ---
if __name__ == "__main__":
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        print(f"正在处理文件，结果将保存到: {OUTPUT_FILENAME} ...")
        
        for item in FILES_TO_PROCESS:
            process_file(item, f)
            
    print("所有文件处理完成。")