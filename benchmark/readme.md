# AgentMemorySystem Benchmark

本目录包含了 AgentMemorySystem 的基准测试和评估框架，用于评估智能体记忆系统在不同任务上的性能表现。

## 📁 目录结构

```
benchmark/
├── readme.md                    # 本文件
├── baselines/                   # 基线方法实现
├── dataset/                     # 测试数据集
│   └── locomo/                 # LoCoMo 数据集
│       ├── locomo10.json       # 主要数据文件
│       ├── msc_personas_all.json  # 人物角色数据
│       └── multimodal_dialog/  # 多模态对话数据
├── results/                    # 评估结果存储
├── scripts/                    # 工具脚本
│   └── env.sh                 # 环境配置脚本
└── task_eval/                 # 任务评估模块
    └── locomo_test.py         # LoCoMo 数据集评估脚本
```

## 🎯 支持的数据集

### LoCoMo (Long Context Memory)
- **描述**: 长上下文记忆评估数据集，包含多轮对话和相关问答
- **规模**: 10个对话样本 (locomo10.json)
- **任务类型**: 
  - 类别1: 事实性问答
  - 类别2: 推理性问答  
  - 类别3: 综合性问答
- **评估指标**: 准确率、Hit@K、检索成功率

## 🚀 快速开始

### 环境配置
```bash
# 设置环境变量
source scripts/env.sh

# 安装依赖（如果还未安装）
pip install -r ../requirements.txt
```

### 运行评估
```bash
# 运行 LoCoMo 数据集评估
cd task_eval
python locomo_test.py
```

## 📊 评估指标

### 主要指标
- **检索成功率**: 成功检索到相关信息的问题比例
- **Hit@K**: Top-K 检索结果的命中率
- **平均相似度**: 检索结果与查询的平均语义相似度
- **按类别准确率**: 不同问题类型的性能表现

### 输出结果
评估结果将自动保存到 `results/` 目录，包含：
- 详细的评估日志
- JSON 格式的结果文件
- 按类别的性能统计

## 🔧 添加新的基线方法

在 `baselines/` 目录下创建新的基线方法：

```python
class YourBaseline:
    def __init__(self):
        pass
    
    def retrieve(self, query: str, k: int = 5):
        """实现检索逻辑"""
        pass
    
    def answer(self, query: str, context: str):
        """实现答案生成逻辑"""
        pass
```

## 📝 添加新的数据集

1. 在 `dataset/` 下创建新的数据集目录
2. 确保数据格式符合评估框架的要求
3. 在 `task_eval/` 下添加对应的评估脚本
4. 更新本 README 文件

### 数据格式规范
```json
{
  "sample_id": "唯一标识符",
  "qa": [
    {
      "question": "问题文本",
      "answer": "标准答案",
      "evidence": ["证据来源"],
      "category": 1
    }
  ]
}
```

## 📈 实验结果

### LoCoMo 数据集基准结果

| 方法 | 检索成功率 | Hit@1 | Hit@3 | Hit@5 | 平均相似度 |
|------|------------|-------|-------|-------|------------|
| AgentMemorySystem | - | - | - | - | - |

*注: 运行评估脚本后将自动更新此表格*

## 🤝 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/new-benchmark`)
3. 提交更改 (`git commit -am 'Add new benchmark'`)
4. 推送分支 (`git push origin feature/new-benchmark`)
5. 创建 Pull Request

## 📚 引用

如果你在研究中使用了本基准测试框架，请引用：

```bibtex
@misc{agentmemorysystem2024,
  title={AgentMemorySystem Benchmark},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/AgentMemorySystem}
}
```

## 📄 许可证

本项目采用 [MIT License](../LICENSE) 许可证。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 创建 Issue: [GitHub Issues](https://github.com/your-repo/AgentMemorySystem/issues)
- 邮箱: your.email@example.com

---

**注意**: 本基准测试框架正在持续开发中，欢迎贡献新的数据集、评估方法和基线算法。