import pandas as pd
import re
from src.config import OPENSTACK_LOG_PATH

class TopologyExtractor:
    def __init__(self):
        self.log_path = OPENSTACK_LOG_PATH

    def extract_components(self):
        """
        从日志文件名的前缀或日志内容中提取组件列表。
        OpenStack LogHub 格式通常在行首包含文件名，如:
        'nova-api.log.1.2017... 2017-05-16 ...'
        """
        components = set()
        print(f"🔍 Scanning components from {self.log_path}...")
        
        with open(self.log_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 提取行首的文件名部分，直到第一个空格
                # e.g., "nova-api.log.1.2017..."
                first_part = line.split(' ')[0]
                
                # 简单的启发式规则：提取组件名 (nova-api, neutron-server 等)
                # 通常是 xxx.log 之前的部分
                if ".log" in first_part:
                    comp_name = first_part.split('.log')[0]
                    # 进一步清洗：去掉数字后缀 (nova-compute-1 -> nova-compute)
                    comp_name = re.sub(r'-\d+$', '', comp_name)
                    components.add(comp_name)
        
        # 如果日志里提取不到（比如格式不一样），我们根据 OpenStack 知识库补全常见组件
        # 以保证实验能跑下去
        if len(components) < 3:
            print("⚠️ Warning: Not enough components found in logs. Using OpenStack default topology.")
            return ["nova-api", "nova-scheduler", "nova-compute", "neutron-server", "cinder-api", "rabbitmq", "keystone"]
            
        print(f"✅ Found {len(components)} components: {components}")
        return list(components)

if __name__ == "__main__":
    extractor = TopologyExtractor()
    comps = extractor.extract_components()
    print(comps)