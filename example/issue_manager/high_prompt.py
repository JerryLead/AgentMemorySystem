from util import query_llm

import re
import json

from memory_core.SemanticMap import MemoryUnit


def technical_prompt(
    repo_name: str,
    contributor_name: str,
    proficient_languages: dict,
    led_modules: list,
    toolchain_proficiency: dict,
):
    prompt = f"""系统指令：
您是代码仓库分析专家，请根据提供的GitHub仓库数据和开发者活动，生成符合要求的英文JSON响应。**除JSON外不得输出任何其他内容**。

输入数据：
1. GitHub仓库名称：{repo_name}
2. 开发者ID：{contributor_name}
3. 文件后缀统计：{proficient_languages}
4. 主导模块目录：{led_modules}
5. 工具链关键词统计：{toolchain_proficiency}

分析要求：
- **编程语言分析**：
  - 根据文件后缀统计占比（需考虑无后缀文件的上下文推断，如SQL文件在`database/`目录）
  - 语言占比总和必须为100%（允许±0.1%误差）

- **工具链与框架识别**：
  - **结合模块目录和关键词**判断开发者擅长的工具链：
    - 模块路径（如`k8s/deployments`暗示Kubernetes）
    - PR关键词（如`terraform`出现5次）
  - 高级框架需满足：
    1. 在模块目录或关键词中出现2次及以上
    2. 属于知名框架/库（如React、TensorFlow、Docker）

输出格式（仅英文JSON）：
{{
  "primary_languages": {{  // 语言名称到百分比的字典
    "string": number  // 示例："Python": 55.0
  }},
  "advanced_frameworks": ["string", ...],  // 如["React", "Docker"]
  "toolchain_specialties": ["string", ...] // 工具链专长（如["Kubernetes", "CI/CD"]）
}}

输出限制：
1. 语言百分比总和必须为100%（允许±0.1%误差）
2. 框架需是知名库/框架（排除通用工具如"git"）
3. 工具链专长需同时满足：
   - 在模块目录或关键词中明确出现
   - 属于开发工具链范畴（如构建、部署、CI/CD）
4. 仅返回有效JSON，禁止任何解释性文本或格式错误"""

    print(f"technical prompt: {prompt}")
    response = query_llm(prompt)
    print(f"response: {response}")
    pattern = r"\`\`\`json(.*?)\`\`\`"
    response = re.search(pattern, response, re.DOTALL)
    if response:
        response = response.group(1).strip()
    else:
        raise ValueError("Response does not match expected JSON format.")
    result = json.loads(response)

    return result


def developer_profile_prompt(
    developer: MemoryUnit,
    commit_stats: dict,
    pr_stats,
    core_files,
    # primary_expertise,
    # extended_expertise,
    file_type_distribution,
    collaboration_patterns,
):
    new_line = "\n"
    tab = "\t"
    prompt = (
        f"""你是一个专业的开发者分析系统。请基于提供的数据，分析并生成开发者的专业知识画像。

## 开发者基本信息:
用户名: {developer.raw_data["login"]}
邮箱: {developer.raw_data["email"]}

## 提交统计:
提交总数: {commit_stats["authored"]}
平均提交大小: {commit_stats["typical_size"]} 修改行数
提交时间偏好: {commit_stats["preferred_times"]}

## PR统计:
提交PR数量: {pr_stats["submitted"]}
审查PR数量: {pr_stats["reviewed"]}
代码审查通过率: {pr_stats["approval_rate"]}%

## 核心贡献文件:
"""
        + "\n".join(
            [
                f"- {f['path']}{new_line}{tab}概要：{f['summary']}{new_line}{tab}贡献频率：{f['contribution_frequency']}"
                for f in core_files
            ]
        )
        + """

## 文件类型分布:
"""
        + "\n".join([f"- {k}: {v}%" for k, v in file_type_distribution.items()])
        + f"""

## 协作模式:
"""
        + "\n".join(
            [f"- {k['description']}: {k['frequency']}" for k in collaboration_patterns]
        )
        + """

请仅生成符合以下JSON格式的开发者专业知识画像，不要包含任何其他文本。以下是一个示例输出，请参考此格式但根据提供的实际数据生成内容：

{
  "developer_id": "sara_chen",
  "profile_type": "开发者专业知识画像",
  "creation_date": "2025-04-26",
  "core_expertise": {
    "primary_domains": ["图数据库", "知识图谱", "搜索优化"],
    "expertise_level": "高级",
    "technical_strengths": ["图算法实现", "大规模数据处理", "系统性能优化"],
    "preferred_technologies": ["Python", "C++", "Neo4j", "PyTorch"]
  },
  "code_proficiency": {
    "mastered_modules": ["图检索引擎", "语义匹配器", "数据索引器"],
    "contribution_areas": ["核心检索算法", "图数据结构", "API设计"],
    "code_quality_traits": ["高度模块化", "注重性能", "详尽的错误处理"],
    "complexity_handling": "擅长分解复杂图算法问题，通过渐进式优化解决性能瓶颈"
  },
  "work_patterns": {
    "commit_style": "中等大小的提交，平均每次修改70-100行代码，通常在工作日上午提交",
    "pr_patterns": "综合型PR，通常包含完整功能实现，附有详细的技术说明和性能指标",
    "review_approach": "严格审查算法复杂度和边界情况，特别关注性能影响",
    "documentation_habits": "提供详尽的API文档和算法原理说明，代码内注释简洁精确"
  },
  "team_dynamics": {
    "collaboration_style": "技术引导型，在设计讨论中提供关键技术见解",
    "typical_roles": ["技术决策者", "复杂问题解决者", "知识分享者"],
    "communication_preferences": ["技术细节导向", "数据支持的讨论", "直接且有建设性的反馈"]
  },
  "growth_trajectory": {
    "skill_evolution": "从通用算法工程师发展为专注于图算法和知识图谱的专家",
    "learning_focus": ["分布式图计算", "向量数据库优化", "多模态图嵌入"],
    "potential_growth_areas": ["图神经网络应用", "跨模态检索系统"]
  },
  "hidden_potential": {
    "untapped_skills": ["自然语言处理集成", "图可视化设计"],
    "recommended_exploration": ["图数据的实时分析", "边缘计算在图数据中的应用"]
  },
  "project_fit": {
    "best_task_types": ["核心算法改进", "性能优化", "系统架构设计"],
    "ideal_project_roles": ["技术架构师", "算法专家", "技术导师"],
    "challenge_recommendation": "设计下一代分布式图检索系统，整合最新的向量搜索技术"
  },
  "confidence_score": 0.92
}

仔细分析所有提供的数据，确保JSON中的内容完全基于事实依据。不要在输出中包含任何引导性文字、说明或JSON之外的内容。"""
    )
    print(f"developer profile prompt: {prompt}")

    response = query_llm(prompt)
    pattern = r"\`\`\`json(.*?)\`\`\`"
    result = re.search(pattern, response, re.DOTALL)
    result = result.group(1).strip() if result else None
    result = json.loads(result)
    return result


def bug_fix_pattern_prompt(issue, pr, commits, files, comments):
    prompt = f"""你是一位经验丰富的软件开发专家，专门分析GitHub仓库中的问题修复模式。我将为你提供一个完整的问题修复上下文，包括Issue详情、相关PR、Commits、修改的文件以及相关评论。

        请详细分析这些信息，总结该问题的修复模式。你必须以有效的JSON格式返回结果，包含以下字段：

        {{
          "问题概述": "简洁描述问题的性质(2-3句)",
          "根本原因": "识别导致问题的技术根源",
          "修复策略": "概述采用的解决方案策略",
          "修复步骤": [
            "步骤1描述",
            "步骤2描述",
            "步骤3描述",
            ...
          ],
          "修改的关键组件": [
            {{
              "文件路径": "文件完整路径",
              "功能描述": "该组件的功能",
              "修改内容": "具体做了什么修改"
            }},
            ...
          ],
          "验证方法": [
            "验证方法1",
            "验证方法2",
            ...
          ],
          "相关知识点": [
            {{
              "知识点": "知识点名称",
              "描述": "简要解释"
            }},
            ...
          ],
          "复现条件": "问题出现的环境或触发条件",
          "通用性": {{
            "适用范围": "该修复模式适用的问题范围",
            "限制条件": "适用的限制条件"
          }},
          "预防措施": [
            "措施1",
            "措施2",
            ...
          ]
        }}

        请确保你的输出是有效的JSON格式，没有Markdown标记或其他格式。总结应简洁清晰，使用技术准确的语言，突出关键见解。

        ===== 问题修复上下文 =====
        Issue: {issue}
        PR: {pr}
        Commits: {commits}
        修改的文件: {files}
        相关评论: {comments}"""

    print(f"bug fix pattern prompt: {prompt}")

    response = query_llm(prompt)
    result = re.search(r"\`\`\`json(.*?)\`\`\`", response, re.DOTALL)
    result = result.group(1).strip() if result else None
    result = json.loads(result)
    return result
