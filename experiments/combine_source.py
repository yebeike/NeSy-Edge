import os

# --- 智能路径设置 ---
# 获取此脚本所在的目录 (假设这个脚本放在项目根目录)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- 全局配置 ---

# 1. 定义最终输出文件的名字和完整路径
#    所有源代码将被合并到这个文件中
OUTPUT_FILENAME = os.path.join(PROJECT_ROOT, "combine_source_output.txt")

# 2. ★★★ 在此处配置你要包含的源文件路径 ★★★
#    你只需要修改这个列表即可！
#    所有路径都应相对于项目根目录 (即 'NuSy-Edge' 目录)。
SOURCE_FILES_TO_COMBINE = [
    # 示例：添加你想要导出的Python文件
    # '../experiments/run_rq2_step1_process_data.py',
    # '../experiments/run_rq2_step2_causal_analysis.py',
    # '../experiments/run_rq2_auto_tuner.py',
    # '../src/reasoning/dynotears.py',
    # '../experiments/debug_rq2_intra_focus.py',
    # '../experiments/run_rq2_debug_comprehensive_check.py'
    # 你可以继续在这里添加其他文件的路径...

    # RQ1 相关文件
    # './experiments/run_rq1_benchmark.py',
    # '../src/utils/noise_injector.py',
    # '../src/utils/metrics.py',
    # '../src/utils/data_loader.py',
    '../src/perception/drain_parser.py',
    '../src/system/edge_node.py',
    '../src/utils/llm_client.py',
    
    # 不仅仅是Python文件，其他文本文件也可以
    # 'requirements.txt',
    # '.gitignore',
    # 'README.md',
]


def main():
    """主函数，执行源代码合并任务。"""
    try:
        # 使用 'with' 语句安全地打开输出文件
        # 'w' 模式每次运行时都会覆盖旧文件，确保输出最新
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as outfile:
            print(f"开始合并源代码... 输出文件为: {OUTPUT_FILENAME}")
            
            # 遍历文件列表
            for relative_path in SOURCE_FILES_TO_COMBINE:
                # 构建每个源文件的绝对路径
                absolute_path = os.path.join(PROJECT_ROOT, relative_path)
                
                print(f"  -> 正在处理: {relative_path}")
                
                try:
                    # 打开并读取源文件
                    with open(absolute_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                    
                    # --- 在输出文件中写入格式化的内容 ---
                    
                    # 写入一个显眼的分隔符和文件名头
                    outfile.write("# " + "=" * 80 + "\n")
                    outfile.write(f"# --- 源文件: {relative_path} ---\n")
                    outfile.write("# " + "=" * 80 + "\n\n")
                    
                    # 写入该文件的全部源代码内容
                    outfile.write(content)
                    
                    # 在文件内容结束后添加一些空行，用于分隔下一个文件
                    outfile.write("\n\n\n")
                    
                except FileNotFoundError:
                    # 如果列表中的某个文件不存在，打印警告并跳过
                    warning_msg = f"  [警告] 文件未找到，已跳过: {relative_path}"
                    print(warning_msg)
                    # 同时也在输出文件中记录这个错误，方便溯源
                    outfile.write("=" * 80 + "\n")
                    outfile.write(f"--- 错误: 文件未找到 -> {relative_path} ---\n")
                    outfile.write("=" * 80 + "\n\n\n")
                except Exception as e:
                    # 捕获其他可能的读取错误 (例如非文本文件的编码问题)
                    error_msg = f"  [错误] 读取文件时出错 {relative_path}: {e}"
                    print(error_msg)
                    outfile.write("=" * 80 + "\n")
                    outfile.write(f"--- 错误: 读取文件时出错 -> {relative_path} ---\n")
                    outfile.write(f"--- 错误信息: {e} ---\n")
                    outfile.write("=" * 80 + "\n\n\n")

        print("所有文件处理完成！")
        
    except IOError as e:
        print(f"致命错误：无法写入输出文件 {OUTPUT_FILENAME}。请检查目录权限。错误: {e}")

# --- 脚本执行入口 ---
if __name__ == "__main__":
    main()