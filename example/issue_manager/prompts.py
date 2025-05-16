from datetime import datetime


def generate_issue_label_prompt(issue: dict, labels: list):
    return f"""**Task Context**
You're a GitHub issue triage AI. Analyze the following new issue and suggest appropriate labels from EXCLUSIVELY the provided list. Current timestamp: {datetime}.

**Available Labels (NAME - DESCRIPTION)**
{labels}

**Instructions**
1. Match tags based on the actual content of the issue
2. Each tag must have an attached confidence score (1-10)
3. Output JSON format: {{
   "suggested_labels": [
     {{"name": tag_name, "confidence": confidence, "reason": "matching_basis"}}, ...
   ]
}}
4. Maximum of 3 tags, minimum of 0
5. Do not create non-existent tags
6. Ensure tags do not overlap (e.g., selecting both bug and enhancement)
7. Consider the current time to evaluate if the issue is urgent or related to current events.

**Issue**
{issue}

Classification Results:
"""


def generate_issue_classification_prompt(issue: dict):
    prompt = f"""Issue : {issue}
    
Analyze the GitHub issue above and classify it as either 'BUG_FIX' or 'FEATURE_REQUEST'. Respond ONLY with the category name.
"""
    return prompt


def generate_issue_complexity_prompt(issue: dict, context: dict):
    issues = context.get("similar_issues")
    if issues:
        total_time = 0
        for i in issues:
            total_time += i.metadata.get("fix_time")
        avg_time = total_time / len(context.get("similar_issues"))
    else:
        avg_time = None
    return f"""
   [问题上下文]
   {str(issue)}
   
   [历史参考]
   相似问题修复时间：{avg_time}天
   可能影响代码文件数量：{len(context.get("fixed_chain").get("files_changed"))}
   
   [评估标准]
   请综合以下因素判断问题复杂度：
   1. 技术实现难度（1-5分）
   2. 所需跨模块协作程度（1-5分） 
   3. 预估工作量（人小时）
   
   [输出要求]
   按JSON格式返回结果，包含complexity_level(SIMPLE/COMPLEX)和reason字段
"""


def generate_issue_urgency_fixtime_prompt(issue: dict, context: dict, category: str):
    return f"""# Issue优先级与时间预估分析框架

## 输入参数
**新Issue描述**:
{str(issue)}

**关联上下文**:
- 类别: {category}
- 关联修复链路径: {context.get("fixed_chain")}
- 相关开发者: {context.get("related_contributors")}

## 分析规则
请分两步执行：

1. **优先级判定（P0-P3）**
- P0（紧急）: 影响核心业务流程的崩溃性bug
- P1（高）: 关键功能异常或安全漏洞
- P2（中）: 局部功能缺陷或重要增强需求 
- P3（低）: 界面优化等非阻塞性问题

判断依据应包含：
- 与历史问题的严重性类比
- 受影响代码模块的业务权重（核心>边缘）
- 当前是否有开发者正在修改相关代码文件

2. **时间预估模型**
- 基础值：相似历史issue解决时长中位数
- 调整系数：基于差异点判断（范围: 0.8-1.5）：
  - 代码复杂度（新增/修改/配置变更）
  - 是否存在跨模块依赖
  - 负责开发者当前任务负载

## 输出要求
严格使用JSON格式：
{{
  "priority": {{
    "level": "Px", 
    "reason": "与#[历史issue]相似的XX问题，但涉及更高权重的[模块名]"
  }},
  "time_estimate": {{
    "base_value": "X小时",
    "adjusted_value": "Y小时",
    "confidence": 0-1（置信度评分）
  }}
}}
"""


def generate_issue_fix_code_prompt(issue: dict, context: dict):
    prompt = f"""# 结构化代码修复指令

## 任务类型
简单级别代码变更（需同时输出**修改要点**和**纯净代码块**）

## 输入上下文
**目标Issue**: 
{str(issue)}

**关联知识图谱**:
- 关联修复链: 
  {context.get("fixed_chain")}
- 历史相似PR的修复模式: 
  {context.get("pr_code_fixes")}

## 输出格式要求
**严格遵循以下分隔格式：**

```output
## 修改要点 ##
[必须包含]
1. 文件路径: /src/.../{{filename}}.py
2. 变更类型: [函数修改/配置更新/类扩展]
3. 受影响行号: Line {{X}}-{{Y}} 或 [新增]
4. 技术描述: 
   - 基于 #{{参考PR}} 的 {{方法名}} 模式
   - 需解决的核心矛盾: {{问题关键词}}
5. 附带影响检查: [需要/无需] 同步更新测试用例

## 代码变更 ##
'''python
{{纯净的代码片段（含必要参数注释）}}
'''
```

## 示例引导
```output
## 修改要点 ##
1. 文件路径: /src/api/client.py
2. 变更类型: 函数修改
3. 受影响行号: Line 45-52
4. 技术描述:
   - 基于 #45 的 timeout 动态配置模式
   - 需解决: 硬编码超时导致偶发失败
5. 附带影响检查: 需更新 test_timeout_policy()

## 代码变更 ##
'''python
def request_handler():
    timeout = config.get('timeout', 30)  # PR#45
    client = httpx.Client(timeout=timeout)
    return _wrap_retry(client)
'''
```

## 验证规则
- 若修改涉及多文件需分区块多次输出
- 禁止在代码块中添加解释性注释（仅允许引用PR的简短标注）
- 修改要点中必须包含至少一个历史参考来源
```
"""
    #     return f"""# 自动化代码修复生成指令

    # ## 任务背景
    # 你是一个资深代码助手，需要为GitHub Issue生成**可直接审查**的修复代码。该问题已被分类为**简单级**，要求解决方案应满足：
    # 1. **准确性**：精确命中问题描述中的核心矛盾点
    # 2. **最小化修改**：优先使用相似PR中的已验证模式
    # 3. **代码健康度**：符合仓库规范
    # 4. **可追溯性**：在注释中引用关联Issue/PR

    # ## 输入上下文
    # **目标Issue**:
    # {str(issue)}

    # **关联知识图谱**:
    # - 关联修复链:
    #   {context.get("fixed_chain")}
    # - 历史相似PR的修复模式:
    #   {context.get("pr_code_fixes")}

    # ## 代码约束
    # - **文件定位**：[必须匹配修复链中的路径]
    # - **语言规范**: Python 3.9+类型注解/PEP8格式
    # - **防御编程**:
    #   - 必须包含关键参数的校验逻辑
    #   - 优先使用仓库utils中的公共错误处理模块

    # ## 输出要求
    # 生成经过深思熟虑的代码块，包含：
    # ```python
    # # [必须] 用三引号包裹带注释的代码
    # '''python
    # # Fix for #{{issue_id}} - {{问题摘要}}
    # # Reference: PR #{{关联PR编号}}
    # def updated_function(...):
    #     \"\"\"
    #     关键修改点：
    #     - 增加XXX校验（来自Issue #123经验）
    #     - 使用utils.error_handler装饰器（遵循PR #45模式）
    #     \"\"\"
    #     # 实现逻辑...
    # '''
    # ```

    # ## 决策树（供LLM自检）
    # 若问题符合以下情况：
    # 1. 精确匹配历史修复模式 → 直接重用代码片段并调整参数
    # 2. 需扩展历史方案 → 使用相同设计模式增强
    # 3. 无直接匹配 → 调用utils模块封装新逻辑

    # ## 验证提示
    # - 检查参数类型是否与调用方兼容？
    # - 是否需要同步更新单元测试或文档？
    # - 修改是否会触发既有功能的副作用？
    # """
    return prompt


def generate_issue_subtasks_prompt(issue: dict, context: dict):
    return f"""# 复杂Issue多维分解指令

## 输入上下文
**目标Issue**: 
{str(issue)}

**关联知识图谱**:
- 历史解决方案锚点: 
  {context.get("fixed_chain")}
- 历史相似Commit: 
  {context.get("similar_commits")}

## 分解原则
采用分层正交分割策略：
1️⃣ **功能解耦层**：将核心需求拆分为可独立开发的功能单元
2️⃣ **时序依赖层**：识别必须顺序执行的阶段（如A模块须先于B测试）
3️⃣ **资源分配层**：确保每个子任务工作量≈1人周内可完成

## 操作规范
对每个子任务需包含：
- **智能摘要**: 用动词开头的精确定义（如"重构XX模块的YY接口"）
- **技术约束**: 必须遵循的设计规范或依赖项

## 输出格式
严格返回JSON数组，每个对象包含：
```json
[
  {{
    "subtask": "子任务描述",
    "technical_components": ["代码文件/模块列表"],
    "reference_solutions": ["历史PR/Issue编号"]
  }},
  // 保持3个元素
]

"""


def generate_simple_draft(
    classification: str,
    labels: list,
    issue: dict,
    urgency: str,
    fix_time: datetime,
    similar_issues: list,
    similar_commits: list,
    advice: str,
    generated_code: str,
    contributors: list,
):
    prompt = (
        f"""
## [自动生成] # {issue["number"]} 解决方案草案

**分类**：{classification}

**标签**：{labels}

**优先级评估**：{urgency}

**时间预估**：预计{fix_time}小时

**关联上下文**：

· **相似历史Issue**：

    """
        + ",\n    ".join(
            [
                f'#{i.raw_data["number"]}: {i.raw_data["title"]} 相似度: {i.metadata["similarity_score"] * 100:.4f}%'
                for i in similar_issues
            ]
        )
        + f""",

· **相似历史Commit**：

    """
        + ",\n    ".join(
            [
                f"#{i.uid[14:21]}  相似度: {i.metadata['similarity_score'] * 100:.4f}%"
                for i in similar_commits
            ]
        )
        + f""",

**修复建议**:
    {advice}

**参考代码方案**：  
```python
{generated_code}
```

**建议审查员**：
    {contributors[0].raw_data["login"] if contributors else "未指定"}
"""
    )
    return prompt


def generate_complex_draft(
    category: str,
    labels: list,
    issue: dict,
    urgency: str,
    fix_time: datetime,
    tasks: list,
    similar_issues: list,
    similar_commits: list,
    contributors: list,
):
    prompt = (
        f"""
## [自动生成] # {issue["number"]} 解决方案草案

**分类**：{category}

**标签**：{labels}

**优先级评估**：{urgency}

**时间预估**：预计{fix_time}小时

**关联上下文**：

· **相似历史Issue**：

    """
        + ",\n    ".join(
            [
                f'#{i.raw_data["number"]}: {i.raw_data["title"]} 相似度: {i.metadata["similarity_score"] * 100:.4f}%'
                for i in similar_issues
            ]
        )
        + f""",

· **相似历史Commit**：

    """
        + ",\n    ".join(
            [
                f"#{i.uid[14:21]}  相似度: {i.metadata['similarity_score'] * 100:.4f}%"
                for i in similar_commits
            ]
        )
        + f"""

**任务分解与指派**：  

    """
        + ",\n    ".join(
            [
                f'· 任务: {tasks[i]} 分配人员: {contributors[i].raw_data["login"] if contributors else None}'
                for i in range(len(tasks))
            ]
        )
        + ","
    )
    return prompt
