import sys
import os

# 获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录
project_root = os.path.abspath(os.path.join(current_dir, ".."))  # 上一级目录
sys.path.append(project_root)

from memory_core.SemanticMap import MemoryUnit, SemanticMap
from memory_core.SemanticGraph import SemanticGraph

from github_analyzer import GithubParser
from prompts import *
from util import query_llm, get_github_client

from github.NamedUser import NamedUser

import re
import json
import datetime
from typing import List, Dict
import logging
from collections import defaultdict

from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)


def print_c(data, color: int = 32):
    """
    颜色样式打印输出功能
    :param data: 打印内容
    :param color: 指定颜色, 默认为绿色(32)
    :return:
    """
    if isinstance(color, int):
        color = str(color)
    print(f"\033[1;{color}m{data}\033[0m")


class IssueManager:
    def __init__(self, sgraph: SemanticGraph):
        self.sgraph = sgraph

    # 核心处理流程
    def process_issue(
        self,
        new_issue: dict,
        category: str = None,
        labels: list = None,
        urgency: str = None,
        fix_time: str = None,
        complexity: str = None,
    ) -> str:
        code_context = self.retrieve_related_codes(new_issue, 5)
        logging.info(f"relative codes: {code_context}")
        return

        # 1. 分类、标签
        category = self.classify_issue(new_issue) if not category else category
        logging.info(f"issue category: {category}")
        labels = self.select_labels(new_issue) if not labels else labels
        logging.info(f"issue labels: {labels}")

        # 2. 上下文检索
        context = self.retrieve_related_memories(new_issue, 5)

        # 3. 优先级、时间评估
        urgency, fix_time = (
            self.assess_urgency_fixtime(new_issue, context, category)
            if not (urgency and fix_time)
            else (urgency, fix_time)
        )
        fix_time = fix_time[:-2]  # 小时数
        logging.info(f"urgency: {urgency}, fix_time: {fix_time}小时")

        # 4. 复杂度评估
        complexity = (
            self.assess_complexity_with_llm(new_issue, context)
            if not complexity
            else complexity
        )
        logging.info(f"complexity: {complexity}")

        # 5. 生成差异化草案
        draft = self.generate_draft(
            new_issue, category, labels, context, urgency, fix_time, complexity
        )
        logging.info(f"draft: {draft}")
        logging.info("=" * 100)

        return draft

    def classify_issue(self, issue: dict):
        prompt = generate_issue_classification_prompt(issue)
        logging.info(f"Querying LLM for issue classification...")
        response = query_llm(prompt)
        logging.info(f"response: {response}")

        pattern = re.compile(r"BUG_FIX|FEATURE_REQUEST")
        match = pattern.search(response)

        while not match:
            logging.info(f"Querying LLM for issue classification...")
            response = query_llm(prompt)
            logging.info(f"response: {response}")
            pattern = re.compile(r"BUG_FIX|FEATURE_REQUEST")
            match = pattern.search(response)

        return match.group()

    def select_labels(self, issue: dict):
        def build_labels_context(repo):
            return [
                {"name": label.name, "description": label.description}
                for label in repo.get_labels()
            ]

        result = []

        prompt = generate_issue_label_prompt(issue, build_labels_context(repo))
        logging.info(f"Querying LLM for label selection...")
        response = query_llm(prompt)
        match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
        if not match:
            return []
        json_str = match.group(1)
        data_dict = json.loads(json_str.strip())

        while not data_dict.get("suggested_labels"):
            logging.info(f"Querying LLM for label selection...")
            response = query_llm(prompt)
            match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
            if not match:
                return []
            json_str = match.group(1)
            data_dict = json.loads(json_str.strip())

        for label in data_dict["suggested_labels"]:
            result.append(label["name"])
        return result

    def retrieve_related_memories(self, issue: dict, top_k: int = 3) -> dict:
        # 语义相似性检索，Issue、Pr
        similar_issues = self.sgraph.smap.find_similar_units(
            str(issue), ms_names=["github_issues"], top_k=top_k
        )
        similar_prs = self.sgraph.smap.find_similar_units(
            str(issue), ms_names=["github_prs"], top_k=top_k
        )

        # 计算每个相似Issue修复时间
        for sim_issue in similar_issues:
            related_prs = self.sgraph.find_relations(
                target=sim_issue.uid, rel_type="fixes"
            )
            if related_prs:
                for pr_r in related_prs:
                    pr = self.sgraph.smap.get_unit_by_uid(pr_r["source"])
                    sim_issue.metadata["fix_time"] = (
                        sim_issue.raw_data.get("created_at")
                        - pr.raw_data.get("created_at")
                    ).days
            else:
                sim_issue.metadata["fix_time"] = (
                    sim_issue.raw_data.get("created_at")
                    - sim_issue.raw_data.get("closed_at")
                ).days

        # 关联开发者检索
        related_contributors = {}
        # Issue->Pr
        for sim_issue in similar_issues:
            related_prs = self.sgraph.find_relations(
                target=sim_issue.uid, rel_type="fixes"
            )
            # Pr->Contributor
            for pr in related_prs:
                devs = self.sgraph.find_relations(pr["target"], rel_type="authored")
                for dev in devs:
                    d = self.sgraph.smap.get_unit_by_uid(dev["target"])
                    if d:
                        if d.uid not in related_contributors:
                            related_contributors[d.uid] = {"contributor": d, "count": 0}
                        related_contributors[d.uid]["count"] += 1
        # Pr->Contributor
        for sim_pr in similar_prs:
            devs = self.sgraph.find_relations(sim_pr.uid, rel_type="authored")
            for dev in devs:
                d = self.sgraph.smap.get_unit_by_uid(dev["target"])
                if d:
                    if d.uid not in related_contributors:
                        related_contributors[d.uid] = {"contributor": d, "count": 0}
                    related_contributors[d.uid]["count"] += 1

        # Sort contributors by frequency
        related_contributors_list = sorted(
            related_contributors.values(), key=lambda x: x["count"], reverse=True
        )
        related_contributors = [
            item["contributor"] for item in related_contributors_list
        ]

        # 最相似的修复链
        fix_chain = {}
        fix_chain["files_changed"] = []
        if similar_issues:
            k = 0
            while k < top_k:
                i = similar_issues[k]  # 第k个相似Issue
                for attr in ["embeddings", "versions", "data_fingerprint"]:
                    if hasattr(i, attr):
                        delattr(i, attr)
                fix_chain["issue"] = i

                pr_rel = self.sgraph.find_relations(
                    target=fix_chain["issue"].uid, rel_type="fixes"
                )
                if not pr_rel:
                    k += 1
                    continue
                fix_chain_pr_rel = pr_rel[0]
                p = self.sgraph.smap.get_unit_by_uid(fix_chain_pr_rel["source"])
                if p:
                    for attr in ["embeddings", "versions", "data_fingerprint"]:
                        if hasattr(p, attr):
                            delattr(p, attr)
                    fix_chain["pr"] = p
                    fix_chain_files_rel = self.sgraph.find_relations(
                        source=fix_chain_pr_rel["source"], rel_type="modifies"
                    )
                    for f in fix_chain_files_rel:
                        fix_chain["files_changed"].append(
                            self.sgraph.smap.get_unit_by_uid(f["target"])["raw_data"][
                                "filename"
                            ]
                        )
                else:
                    fix_chain["pr"] = None

        if len(related_contributors) < top_k:
            related_contributors.extend(
                self.sgraph.smap.find_similar_units(
                    str(issue),
                    "github_contributors",
                    top_k=top_k - len(related_contributors),
                )
            )

        for item in [similar_issues, related_contributors]:
            for i in item:
                for attr in ["embeddings", "versions", "data_fingerprint"]:
                    if hasattr(i, attr):
                        delattr(i, attr)

        # 相似Commit
        similar_commits: List[MemoryUnit] = []
        for sim_issue in similar_issues:
            commit_rel = self.sgraph.find_relations(
                target=sim_issue.uid,
                rel_type="semantic_similarity",
            )
            commit_rel = sorted(commit_rel, key=lambda x: x["score"], reverse=True)[
                :top_k
            ]
            for c in commit_rel:
                commit = self.sgraph.smap.get_unit_by_uid(c["source"])
                if commit:
                    similar_commits.append(commit)
        if len(similar_commits) < top_k:
            similar_commits.extend(
                self.sgraph.smap.find_similar_units(
                    str(issue), "github_commits", top_k=top_k - len(similar_commits)
                )
            )

        # 相似PR的代码修复
        code_fixes = {}
        most_similar_pr = similar_prs[0]
        if most_similar_pr:
            pr = repo.get_pull(int(most_similar_pr.uid[10:]))
            assert pr
            code_fixes["pr"] = f"#{pr.number} {pr.title}\n{pr.body}"
            code_fixes["files_changed"] = []
            for f in pr.get_files():
                if f.patch:
                    code_fixes["files_changed"].append({f.filename: f.patch[:1000]})

        data = {
            "similar_issues": similar_issues,
            "similar_commits": similar_commits,
            "similair_prs": similar_prs,
            "pr_code_fixes": code_fixes,
            "related_contributors": related_contributors,
            "fixed_chain": fix_chain,
        }

        logging.info(f"context: {data}")
        logging.info("-" * 30)

        return data

    def retrieve_related_codes(self, issue: dict, top_k: int = 3):
        similar_codes: List[MemoryUnit] = []

        # 关联PR追踪
        similar_issues = self.sgraph.smap.find_similar_units(
            str(issue), ms_names=["github_issues"], top_k=top_k
        )
        logging.info(f"similar issues: {similar_issues}")
        rel_prs: List[MemoryUnit] = []
        for i in similar_issues:
            prs = self.sgraph.find_relations(target=i.uid, rel_type="fixes")
            logging.info(f"prs: {prs}")
            for pr in prs:
                rel_prs.append(self.sgraph.smap.get_unit_by_uid(pr["source"]))
        for pr in rel_prs:
            rel_codes = self.sgraph.find_relations(source=pr.uid, rel_type="modifies")
            for code in rel_codes:
                c = self.sgraph.smap.get_unit_by_uid(code["target"])
                if c:
                    similar_codes.append(c)
        rel_prs = []
        logging.info(f"similar codes after pr: {similar_codes}")

        # 贡献者线索
        ori_issue = repo.get_issue(issue["number"])
        contributors: List[str] = [c.user.login for c in ori_issue.get_comments()]
        contributors = (
            contributors.append(ori_issue.user.login)
            if ori_issue.user.login not in contributors
            else contributors
        )
        logging.info(f"contributors: {contributors}")

        for c_uid in contributors:
            if c_uid in self.sgraph.smap.memoryspaces["github_contributors"].units:
                prs = self.sgraph.find_relations(
                    source=c_uid,
                    rel_type="authored",
                )
                for pr in prs:
                    rel_prs.append(self.sgraph.smap.get_unit_by_uid(pr["target"]))
                for pr in rel_prs:
                    rel_codes = self.sgraph.find_relations(
                        source=pr.uid, rel_type="modifies"
                    )
                    for code in rel_codes:
                        code_unit = self.sgraph.smap.get_unit_by_uid(code["target"])
                        if code_unit not in similar_codes and code_unit:
                            similar_codes.append(code_unit)
        logging.info(f"similar codes after contrib: {similar_codes}")

        # 时间轴关联
        time_threshold = datetime.timedelta(days=30)
        first_commit = self.sgraph.smap.get_unit_by_uid(
            f"github_commit_81b81cf60b2c1771791b92250f07bcbee86190d4"
        )
        file_frequency = defaultdict(int)

        rel_commit = self.sgraph.find_relations(
            source=first_commit.uid, rel_type="next"
        )
        while rel_commit:
            next_commit: MemoryUnit = self.sgraph.smap.get_unit_by_uid(
                rel_commit[0]["target"]
            )
            if issue["created_at"] - next_commit.raw_data["timestamp"] > time_threshold:
                break
            for file_name in next_commit.raw_data["files"]:
                file_frequency[file_name] += 1
            rel_commit = self.sgraph.find_relations(
                source=next_commit.uid, rel_type="next"
            )
        most_frequent_files = sorted(
            file_frequency.items(), key=lambda x: x[1], reverse=True
        )[:top_k]
        for f in most_frequent_files:
            code_unit = self.sgraph.smap.get_unit_by_uid(f"github_code_{f[0]}")
            if code_unit not in similar_codes and code_unit:
                similar_codes.append(code_unit)
        logging.info(f"similar codes after commit: {similar_codes}")

        query_vec = self.sgraph.smap.text_model.encode(str(issue))
        code_vecs = [
            code_unit.raw_data["desc_embedding"] for code_unit in similar_codes
        ]
        if not code_vecs:
            return {}
        cosine_similarities = cosine_similarity([query_vec], code_vecs)[0]
        # Create a list of (index, similarity) tuples
        similarity_scores = list(enumerate(cosine_similarities))

        # Sort the list by similarity score in descending order
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Get the indices of the top_k most similar items
        top_indices = [i[0] for i in sorted_scores[:top_k]]

        # Retrieve the corresponding code units
        top_similar_codes = [similar_codes[i] for i in top_indices]

        data = {
            "similar_codes": top_similar_codes,
        }
        logging.info(f"context: {data}")
        logging.info("-" * 30)

        return data

    def assess_urgency_fixtime(self, issue: dict, context: dict, category: str):
        prompt = generate_issue_urgency_fixtime_prompt(issue, context, category)
        logging.info(f"Querying LLM for urgency and fix time...")
        response = query_llm(prompt)
        match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
        if not match:
            return []
        json_str = match.group(1)
        data_dict = json.loads(json_str.strip())

        while not (
            data_dict.get("priority").get("level")
            and data_dict.get("time_estimate").get("adjusted_value")
        ):
            logging.info(f"Querying LLM for urgency and fix time...")
            response = query_llm(prompt)
            match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
            if not match:
                return []
            json_str = match.group(1)
            data_dict = json.loads(json_str.strip())

        return data_dict.get("priority").get("level"), data_dict.get(
            "time_estimate"
        ).get("adjusted_value")

    def assess_complexity_with_llm(self, issue: dict, context: dict):
        prompt = generate_issue_complexity_prompt(issue, context)
        logging.info(f"Querying LLM for complexity assessing...")
        response = query_llm(prompt)
        match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
        if not match:
            return []
        json_str = match.group(1)
        data_dict = json.loads(json_str.strip())

        while not data_dict.get("complexity_level"):
            logging.info(f"Querying LLM for complexity assessing...")
            response = query_llm(prompt)
            match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
            if not match:
                return []
            json_str = match.group(1)
            data_dict = json.loads(json_str.strip())

        return data_dict.get("complexity_level")

    def generate_fix_code_with_llm(self, issue: dict, context: dict):
        prompt = generate_issue_fix_code_prompt(issue, context)
        logging.info(f"Querying LLM for fix code generating...")
        response = query_llm(prompt)
        logging.info(f"response: {response}")
        logging.info("-" * 30)

        advice_match = re.search(
            r"## 修改要点  ##(.*?)\n## 代码变更", response, re.DOTALL
        )
        advice: str
        code_match = re.search(r"\'\'\'python\n(.*?)\n\'\'\'", response, re.DOTALL)
        code: str

        if not advice_match:
            advice = ""
        else:
            advice = advice_match.group(1)
        if not code_match:
            code = ""
        else:
            code = code_match.group(1)

        response = (advice, code)
        return response

    def breakdown_issue_with_llm(self, issue: dict, context: dict):
        prompt = generate_issue_subtasks_prompt(issue, context)
        logging.info(f"Querying LLM for task breaking down...")
        response = query_llm(prompt)
        match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)

        while not match:
            logging.info(f"Querying LLM for task breaking down...")
            response = query_llm(prompt)
            match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)

        json_str = match.group(1)
        data_dict = json.loads(json_str.strip())
        return [subtask["subtask"] for subtask in data_dict]

    def generate_draft(
        self,
        issue,
        category,
        labels,
        context,
        urgency,
        fix_time,
        complexity,
    ):
        if complexity == "COMPLEX":
            tasks = self.breakdown_issue_with_llm(issue, context)
            # tasks = [
            #     "重构Update Final Entities步骤中的LLM调用逻辑，引入批量处理机制",
            #     "优化现有实体更新流程，确保与批量处理兼容",
            #     "添加配置管理，支持自定义批次大小参数",
            # ]
            logging.info(f"tasks: {tasks}")
            logging.info("-" * 30)
            draft = generate_complex_draft(
                category,
                labels,
                issue,
                urgency,
                fix_time,
                tasks,
                context.get("similar_issues"),
                context.get("similar_commits"),
                context.get("related_contributors"),
            )
        else:
            advice, fix_code = self.generate_fix_code_with_llm(issue, context)
            logging.info(f"advice: {advice}")
            logging.info(f"fix code: {fix_code}")
            logging.info("-" * 30)
            draft = generate_simple_draft(
                category,
                labels,
                issue,
                urgency,
                fix_time,
                context.get("similar_issues"),
                context.get("similar_commits"),
                advice,
                fix_code,
                context.get("related_contributors"),
            )
        return draft


def _init() -> IssueManager:
    global repo

    smap = SemanticMap()
    sgraph = SemanticGraph(smap)
    sgraph.load_graph("data/github_graph.pkl")
    sgraph.smap.load_data("data/github_smap.pkl")

    github_client = get_github_client()
    repo = github_client.get_repo("microsoft/graphrag")

    issue_manager = IssueManager(sgraph)
    return issue_manager


if __name__ == "__main__":
    issue_manager = _init()

    issues = [
        1538,
        1833,
        1832,
        1831,
        1829,
        1823,
        1820,
        1814,
        1806,
        1804,
        1793,
        1783,
        1776,
        1767,
        1725,
        1717,
        1715,
        1714,
        1711,
        1707,
        1702,
        1688,
    ]
    for num in issues:
        issue = repo.get_issue(num)
        if issue.state_reason == "not_planned":
            continue

        # issue = repo.get_issue(1707)  # pr1709
        # issue = repo.get_issue(1814)  # pr1818
        # issue = repo.get_issue(1725)

        data = GithubParser.parse_github_issue(issue)
        del data["labels"]
        del data["closed_at"]
        del data["comments"]
        logging.info(f"issue: {data}")
        logging.info("-" * 30)

        issue_manager.process_issue(data)

        break

    # issue = repo.get_issue(1833)
    # data = GithubParser.parse_github_issue(issue)
    # del data["labels"]
    # del data["closed_at"]
    # del data["comments"]

    # logging.info(f"issue: {data}")
    # logging.info("-" * 30)

    # issue_manager.process_issue(
    #     data,
    #     category="BUG_FIX",
    #     labels=["bug", "backlog"],
    #     urgency="P0",
    #     fix_time="6小时",
    #     complexity="SIMPLE",
    # )

    # issue_manager.process_issue(  # 1707
    #     data,
    #     category="BUG_FIX",
    #     labels=["bug"],
    #     urgency="P0",
    #     fix_time="12小时",
    #     complexity="COMPLEX",
    # )

    # issue_manager.process_issue( # 1814
    #     data,
    #     category="BUG_FIX",
    #     labels=["bug"],
    #     urgency="P2"
    #     )

    # issue_manager.process_issue(  # 1725
    #     data,
    #     category="BUG_FIX",
    #     labels=["bug"],
    #     urgency="P2",
    #     fix_time="3小时",
    #     complexity="SIMPLE",
    # )
