import sys
import os

# 获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录
project_root = os.path.abspath(os.path.join(current_dir, ".."))  # 上一级目录
sys.path.append(project_root)

import logging
from collections import defaultdict
import os
import pickle
from typing import List

from IssueManagerSemanticMap import MemoryUnit, SemanticMap
from IssueManagerSemanticGraph import SemanticGraph
from util import get_github_client, query_llm
from high_prompt import *

from github import GithubException


global smap, sgraph, repo
smap = SemanticMap(enable_embedding_model=False)
sgraph = SemanticGraph(smap)

sgraph.load_graph(
    "/mnt/data1/home/guozy/gzy/SDS/multimodal_semantic_map_dev/issue_manager/data/github_graph.pkl"
)
smap.load_data(
    "/mnt/data1/home/guozy/gzy/SDS/multimodal_semantic_map_dev/issue_manager/data/github_smap.pkl"
)

github_client = get_github_client()
repo = github_client.get_repo("microsoft/graphrag")


class DeveloperProfile:
    def __init__(self, contributor: MemoryUnit):

        assert isinstance(
            contributor, MemoryUnit
        ), "contributor must be a MemoryUnit instance"
        assert (
            contributor.uid[:19] == "github_contributor_"
        ), "contributor must in contributor MemorySpace"
        self.contributor: MemoryUnit = contributor

        self.profile: dict = {
            "technical_expertise": {
                "proficient_languages": defaultdict(int),
                "led_modules": [],
                "toolchain_proficiency": {},
            },
            "collaboration_patterns": {
                "frequent_collaborators": ["Developer 2"],
                "cross_team_tendency": "Frontend + Backend joint debugging",
            },
            "resolution_efficiency": {
                "average_issue_resolution_time": "24 hours",
                "pr_merge_success_rate": "92%",
                "code_review_pass_rate": "85%",
            },
            "workload_dynamics": {
                "active_tasks": 3,
                "recent_response_latency": "<2 hours",
            },
            "knowledge_graph_relations": {
                "implicit_module_relations": "Modified a.py most frequently",
                "technical_debt_expertise": "Technical debt resolution expert",
            },
        }

    def __eq__(self, value):
        if not isinstance(value, DeveloperProfile):
            return False
        return self.contributor == value.contributor

    def __repr__(self):
        return f"DeveloperProfile({self.contributor.__repr__()}: {self.profile['technical_expertise']})"

    def cal_lang_distribution(self):
        prs = smap.memoryspaces["github_prs"].units.values()
        for pr in prs:
            print(f"Processing pr {pr.uid}")
            try:
                pull_request = repo.get_pull(int(pr.uid[10:]))
            except GithubException as e:
                if e.status == 404:
                    print(f"Pull request {pr.uid} not found.")
                    continue
                else:
                    raise

            if pull_request.user.login == self.contributor.uid[19:]:
                print(f"Processing author's files...")
                for f in pull_request.get_files():
                    print(f"Processing file {f.filename}...")
                    ext = f.filename.split(".")[-1]
                    self.profile["technical_expertise"]["proficient_languages"][
                        ext
                    ] += 1


def summary_developer_profile():

    keywords = [
        "ci",
        "spacy",
        "neo4j",
        "cosmosdb",
        "docker",
        "pytest",
        "mypy",
        "mkdocs",
        "jupyter",
    ]

    profiles = {}
    modified_times = {}
    important_prs = []

    # for contributor in smap.memoryspaces["github_contributors"].units.values():
    #     if contributor.uid[19:] == "dependabot[bot]":
    #         continue
    #     dev_profile = DeveloperProfile(contributor)
    #     profiles[contributor.uid[19:]] = dev_profile

    # prs = smap.memoryspaces["github_prs"].units.values()
    # # 对于每个PR，获取其修改的文件
    # for pr in prs:
    #     print(f"Processing pr {pr.uid}")

    #     try:
    #         pull_request = repo.get_pull(int(pr.uid[10:]))
    #     except GithubException as e:
    #         if e.status == 404:
    #             print(f"Pull request {pr.uid} not found.")
    #             continue
    #         else:
    #             raise

    #     if pull_request.user.login == "dependabot[bot]":
    #         continue

    #     if pull_request.additions + pull_request.deletions > 500:
    #         important_prs.append(pull_request.number)

    #     # user -> [提交] -> pr
    #     user_rels = sgraph.find_relations(
    #         target=pr.uid,
    #         rel_type="authored",
    #     )

    #     if not user_rels:
    #         print(f"PR {pr.uid} has no user.")
    #         continue

    #     user_name = user_rels[0]["source"][19:]

    #     code_rels = pull_request.get_files()
    #     for code_rel in code_rels:
    #         print(f"Processing code {code_rel.filename}")
    #         # 统计用户修改的文件类型
    #         ext = code_rel.filename.split(".")[-1]
    #         profiles[user_name].profile["technical_expertise"]["proficient_languages"][
    #             ext
    #         ] += 1

    #         # 统计文件被用户修改的次数
    #         # TODO: 是否需要改为修改的行数？
    #         dir_name = os.path.dirname(code_rel.filename)
    #         if dir_name not in modified_times:
    #             modified_times[dir_name] = defaultdict(int)

    #         modified_times[dir_name][user_name] += 1

    #     # 将关键词匹配加入到开发者画像中
    #     if pull_request.body:
    #         for keyword in keywords:
    #             if keyword in pull_request.body.lower():
    #                 if (
    #                     profiles[user_name]
    #                     .profile["technical_expertise"]["toolchain_proficiency"]
    #                     .get(keyword)
    #                 ):
    #                     profiles[user_name].profile["technical_expertise"][
    #                         "toolchain_proficiency"
    #                     ][keyword] += 1
    #                 else:
    #                     profiles[user_name].profile["technical_expertise"][
    #                         "toolchain_proficiency"
    #                     ][keyword] = 1
    #                 print(f"{user_name} is proficient in {keyword}")

    # for f_name, f_dict in modified_times.items():
    #     f_dict = sorted(f_dict.items(), key=lambda x: x[1], reverse=True)
    #     contributor_name = f_dict[0][0]
    #     profiles[contributor_name].profile["technical_expertise"]["led_modules"].append(
    #         f_name
    #     )

    # print(f"Modified times: {modified_times}")
    # for p in profiles.values():
    #     print(f"{p.contributor.uid[19:]}: {p.profile['technical_expertise']}")

    # with open(
    #     "/mnt/data1/home/guozy/gzy/SDS/multimodal_semantic_map_dev/issue_manager/data/developer_profiles.pkl",
    #     "wb",
    # ) as f:
    #     pickle.dump(profiles, f)

    with open(
        "/mnt/data1/home/guozy/gzy/SDS/multimodal_semantic_map_dev/issue_manager/data/developer_profiles.pkl",
        "rb",
    ) as f:
        profiles = pickle.load(f)

    for profile in profiles.values():
        # 生成编程语言专长描述
        response = technical_prompt(
            repo_name=repo.full_name,
            contributor_name=profile.contributor.uid[19:],
            proficient_languages=profile.profile["technical_expertise"][
                "proficient_languages"
            ],
            led_modules=profile.profile["technical_expertise"]["led_modules"],
            toolchain_proficiency=profile.profile["technical_expertise"][
                "toolchain_proficiency"
            ],
        )
        # response = {
        #     "primary_languages": {
        #         "Python": 99.22,
        #         "JavaScript": 0.43,
        #         "Shell": 0.17,
        #         "CSS": 0.17,
        #     },
        #     "advanced_frameworks": [],
        #     "toolchain_specialties": ["CI/CD", "Yarn"],
        # }

        for key in [
            "primary_languages",
            "advanced_frameworks",
            "toolchain_specialties",
        ]:
            if key not in response:
                raise KeyError(f"Response does not contain {key} key.")
            profile.profile["technical_expertise"][key] = response[key]

        print(
            f"Developer {profile.contributor.uid[19:]} profile: {profile.profile['technical_expertise']}"
        )


def summarize_bug_fix_pattern(issue: MemoryUnit):
    print("-" * 100)
    # 获取issue及其关联的PR
    related_prs = [
        sgraph.smap.get_unit_by_uid(r["source"])
        for r in sgraph.find_relations(target=issue.uid, rel_type="fixes")
    ]
    related_prs = [pr for pr in related_prs if pr]
    print(f"Related PRs: {related_prs}")

    patterns = []
    for pr_node in related_prs:
        pull = repo.get_pull(int(pr_node.uid[10:]))

        # 获取PR关联的commits
        commits = [
            sgraph.smap.get_unit_by_uid(r["target"])
            for r in sgraph.find_relations(source=pr_node.uid, rel_type="including")
        ]
        commits = [commit for commit in commits if commit]
        print(f"Related commits: {commits}")

        # 获取修改的文件
        modified_files: dict = {}
        for f in pull.get_files():
            modified_files[f.filename] = f.patch
        print(f"modified_files: {modified_files}")

        # 获取相关评论
        comments = [c.body for c in pull.get_comments()]

        # 使用LLM总结修复模式
        pattern = bug_fix_pattern_prompt(
            issue=issue.raw_data,
            pr=pr_node.raw_data,
            commits=[c.raw_data for c in commits],
            files=modified_files,
            comments=comments,
        )

        patterns.append(pattern)

    return patterns


if __name__ == "__main__":
    # summary_developer_profile()

    # issues = smap.memoryspaces["github_issues"].units.values()
    # for issue in issues:
    #     pattern = summarize_bug_fix_pattern(issue)
    #     print(f"Pattern for issue {issue.uid}: {pattern}")

    issue = smap.get_unit_by_uid("github_issue_1538")
    print(issue)
    pattern = summarize_bug_fix_pattern(issue)
    print(f"Pattern for issue {issue.uid}: {pattern}")
