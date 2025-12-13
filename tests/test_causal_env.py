import sys
import os
import warnings
import numpy as np
import pandas as pd

# 忽略 CausalNex 的一些未来版本警告
warnings.filterwarnings("ignore")

def run_test():
    print("🚀 Starting CausalNex Environment Check...")
    
    try:
        # 1. 测试 CausalNex 导入
        import causalnex
        from causalnex.structure.dynotears import from_pandas_dynamic
        from causalnex.structure import StructureModel
        print(f"✅ Library Import Success: causalnex {causalnex.__version__}")
        
        # 2. 测试 PyGraphviz (这是最容易挂的地方)
        import pygraphviz
        print(f"✅ Library Import Success: pygraphviz {pygraphviz.__version__}")

        # 3. 测试 DYNOTEARS 算法 (核心算法验证)
        print("\n🧪 Running DYNOTEARS Simulation (p=1)...")
        
        # 构造简单的时序数据: 
        # 假设 A 在 t-1 时刻的值 影响 B 在 t 时刻的值
        data = {
            'A': np.random.randn(100),
            'B': np.random.randn(100)
        }
        df = pd.DataFrame(data)
        
        # 运行算法
        # p=1 表示寻找滞后1步的因果关系
        graph = from_pandas_dynamic(df, p=1, w_threshold=0.0)
        
        print("✅ Algorithm Execution Success!")
        print(f"   Resulting Edges: {graph.edges}")
        print("\n🎉 Environment is READY for Phase 4!")

    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("   Hint: Did you use 'conda install pygraphviz'?")
    except Exception as e:
        print(f"\n❌ Runtime Error: {e}")

if __name__ == "__main__":
    run_test()