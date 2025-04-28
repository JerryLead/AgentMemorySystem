import sys
import os

# 获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录
project_root = os.path.abspath(os.path.join(current_dir, ".."))  # 上一级目录
sys.path.append(project_root)

from github import Github
from github.Repository import Repository
from github.Issue import Issue
from github.PullRequest import PullRequest
from github.NamedUser import NamedUser
from github.Label import Label
from github.Commit import Commit
from github.File import File
from github.ContentFile import ContentFile
from github import GithubException

from util import get_github_client

from typing import Dict, Any, List, Optional, Callable
from memory_core.SemanticMap import SemanticMap, MemorySpace, MemoryUnit
from memory_core.SemanticGraph import SemanticGraph
import re
import ast
import logging

from collections import defaultdict

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)


class GithubParser:
    @staticmethod
    def parse_github_issue(issue: Issue) -> Dict:
        """将GitHub Issue转换为内存单元格式"""
        return {
            "number": issue.number,
            "title": issue.title,
            "body": issue.body,
            "created_at": issue.created_at,
            "updated_at": issue.updated_at,
            "closed_at": issue.closed_at,
            "state": issue.state,
            "labels": [label.name for label in issue.get_labels()],
            "comments": [comment.body for comment in issue.get_comments()],
        }

    @staticmethod
    def parse_github_pr(pr: PullRequest) -> Dict:
        """将GitHub PR转换为内存单元格式"""
        return {
            "number": pr.number,
            "title": pr.title,
            "body": pr.body,
            "user": pr.user.login,
            "created_at": pr.created_at,
            "updated_at": pr.updated_at,
            "closed_at": pr.closed_at,
            "state": pr.state,
            "labels": [label.name for label in pr.get_labels()],
            "files_changed": [f.filename for f in pr.get_files()],
            "commits": [c.commit.sha for c in pr.get_commits()],
            "reviewers": (
                [r.user.login for r in pr.get_reviews()] if pr.get_reviews() else []
            ),
            "merged_at": pr.merged_at,
        }

    @staticmethod
    def parse_github_code_file(repo: Repository, git_file: ContentFile) -> Dict:
        """解析代码文件内容"""
        return {
            "filename": git_file.path,
            "content": git_file.decoded_content.decode(),
            "url": git_file._url,
        }

    @staticmethod
    def parse_github_code_block(file_content: str) -> List:
        """解析代码内容并分段"""

        def is_main_check(test_node):
            """检查是否为if __name__ == '__main__'条件"""
            if isinstance(test_node, ast.Compare):
                left = test_node.left
                comparators = test_node.comparators
                return (
                    isinstance(left, ast.Name)
                    and left.id == "__name__"
                    and isinstance(comparators[0], ast.Constant)
                    and comparators[0].value == "__main__"
                )
            return False

        def get_block_type(node):
            """判断节点所属的代码块类型"""
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return "import"
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return "function"
            elif isinstance(node, ast.ClassDef):
                return "class"
            elif isinstance(node, ast.If) and is_main_check(node.test):
                return "main"
            else:
                return "other"

        def get_end_line(node):
            """获取节点的结束行号（兼容Python <3.8）"""
            if hasattr(node, "end_lineno"):
                return node.end_lineno
            max_line = getattr(node, "lineno", 0)
            for child in ast.iter_child_nodes(node):
                max_line = max(max_line, get_end_line(child))
            return max_line

        def get_block_range(node):
            """获取节点的起始和结束行（考虑装饰器）"""
            start_line = node.lineno
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                decorator_lines = [
                    dec.lineno for dec in node.decorator_list if hasattr(dec, "lineno")
                ]
                if decorator_lines:
                    start_line = min(decorator_lines)
            end_line = get_end_line(node)
            return start_line, end_line

        tree = ast.parse(file_content)

        blocks = []
        current_block = None
        lines = file_content.split("\n")

        for node in tree.body:
            block_type = get_block_type(node)
            start_line, end_line = get_block_range(node)

            # 处理main块（独立成段）
            if block_type == "main":
                if current_block:
                    blocks.append(current_block)
                    current_block = None
                blocks.append(
                    {
                        "type": "main",
                        "start_line": start_line,
                        "end_line": end_line,
                        "name": None,
                    }
                )
                continue

            # 处理其他块
            if (
                current_block
                and current_block["type"] == block_type
                and block_type in ["import", "other"]
            ):
                current_block["end_line"] = end_line
            else:
                if current_block:
                    blocks.append(current_block)
                current_block = {
                    "type": block_type,
                    "start_line": start_line,
                    "end_line": end_line,
                    "name": getattr(node, "name", None),
                }

        if current_block:
            blocks.append(current_block)

        for block in blocks:
            block["content"] = "\n".join(
                [
                    line.strip()
                    for line in lines[block["start_line"] - 1 : block["end_line"]]
                ]
            )

        return blocks

    @staticmethod
    def parse_github_contributor(contributor: NamedUser) -> Dict:
        """解析GitHub贡献者信息"""
        return {
            "login": contributor.login,
            "id": contributor.id,
            "avatar_url": contributor.avatar_url,
            "url": contributor.html_url,
            "email": contributor.email,
            "expertise": contributor.bio,
            "team": contributor.company,
            "activity": {
                "commit_count": contributor.contributions,
                "last_active": contributor.updated_at,
            },
            "recent_activity": contributor.get_events()[:1].__repr__(),
        }

    @staticmethod
    def parse_github_label(label: Label) -> Dict:
        """解析GitHub标签信息"""
        return {
            "color": label.color,
            "default": label.default,
            "description": label.description,
            "id": label.id,
            "name": label.name,
            "node_id": label.node_id,
            "url": label.url,
        }

    @staticmethod
    def parse_github_commit(commit: Commit) -> Dict:
        """解析GitHub Commit信息"""
        return {
            "sha": commit.sha,
            "author": commit.author.login if commit.author else None,
            "committer": commit.committer.login if commit.committer else None,
            "message": commit.commit.message,
            "timestamp": commit.commit.author.date,
            "files": [f.filename for f in commit.files],
            "url": commit.url,
        }


class GitHubImporter:
    def __init__(
        self, semantic_map: SemanticMap, semantic_graph: SemanticGraph, repo: Repository
    ):
        self.sgraph = semantic_graph
        self.smap = semantic_map
        self.repo = repo
        self._init_namespaces()

    def _init_namespaces(self):
        """创建GitHub相关命名空间"""
        self.smap.create_namespace(
            ms_name="github_issues",
            ms_type="github/issue",
            embedding_fields=["title", "body"],
        )
        self.smap.create_namespace(
            ms_name="github_prs",
            ms_type="github/pr",
            embedding_fields=["title", "body"],
        )
        self.smap.create_namespace(
            ms_name="github_code",
            ms_type="github/code",
            embedding_fields=["content"],
        )

        self.smap.create_namespace(
            ms_name="github_code_blocks",
            ms_type="github/code_blocks",
            embedding_fields=["content"],
        )

        self.smap.create_namespace(
            ms_name="github_contributors",
            ms_type="github/contributor",
            embedding_fields=["expertise", "activity"],
        )

        self.smap.create_namespace(
            ms_name="github_commits",
            ms_type="github/commit",
            embedding_fields=["message"],
        )

    def import_repo(self, repo: Repository, max_item: int = 50):
        """导入仓库数据到内存系统"""

        # 导入Issue
        issues = repo.get_issues(state="closed")[:max_item]
        for issue in issues:
            data = GithubParser.parse_github_issue(issue)
            uid = f"github_issue_{issue.number}"
            unit = MemoryUnit(uid=uid, raw_data=data)
            self.smap.insert_unit("github_issues", unit)

        # 导入PR
        prs = repo.get_pulls(state="closed")[:max_item]
        for pr in prs:
            data = GithubParser.parse_github_pr(pr)
            uid = f"github_pr_{pr.number}"
            unit = MemoryUnit(uid=uid, raw_data=data)
            self.smap.insert_unit("github_prs", unit)

        # 导入代码文件（选取py文件）
        contents = repo.get_contents("graphrag")
        py_files = []

        def collect_py_files(contents):
            for content in contents:
                if content.type == "dir":
                    # 递归地处理子目录
                    collect_py_files(repo.get_contents(content.path))
                elif content.path.endswith(".py"):
                    py_files.append(content)

        collect_py_files(contents)
        py_files = py_files[:max_item]
        for f in py_files:
            data = GithubParser.parse_github_code_file(repo, f)
            uid = f"github_code_{f.path}"
            unit = MemoryUnit(uid=uid, raw_data=data)
            self.smap.insert_unit("github_code", unit)

        # 导入贡献者信息
        contributors = repo.get_contributors()[:max_item]
        for contributor in contributors:
            data = GithubParser.parse_github_contributor(contributor)
            uid = f"github_contributor_{contributor.login}"
            unit = MemoryUnit(uid=uid, raw_data=data)
            self.smap.insert_unit("github_contributors", unit)

        # 导入Commit信息
        commits = repo.get_commits()[:max_item]
        for commit in commits:
            data = GithubParser.parse_github_commit(commit)
            uid = f"github_commit_{commit.sha}"
            unit = MemoryUnit(uid=uid, raw_data=data)
            self.smap.insert_unit("github_commits", unit)

    def _parse_closed_issues(self, body: str) -> List[int]:
        """从PR描述中解析关闭的Issue编号"""
        if not body:
            return []

        patterns = [
            r"## Related Issues\n\n#(\d+)",
            r"https://github\.com/microsoft/graphrag/pull/(\d+)",
        ]
        issues = []
        for pattern in patterns:
            matches = re.findall(pattern, body, re.IGNORECASE)
            for match in matches:
                # 如果match是字符串，则直接转换；如果是元组，则取第一个元素（因为re.findall在有捕获组时返回元组）
                issue_number = int(match[0]) if isinstance(match, tuple) else int(match)
                issues.append(issue_number)
        return list(set(issues))

    def _build_relationships(self):
        """构建关系网络"""
        # 添加commit关系
        commits = self.smap.memoryspaces["github_commits"].units.values()
        pre_commit = None
        for commit in commits:
            commit_uid = commit.uid

            # 提交者
            if commit.raw_data["author"]:
                user_uid = f"github_contributor_{commit.raw_data['author']}"
                print(f"author {user_uid} -> commit {commit_uid}")
                self.sgraph._add_explicit_edge(
                    source=user_uid,
                    target=commit_uid,
                    rel_type="authored",
                    weight=1.0,
                    metadata={"role": "author"},
                )

            # 审查者
            if commit.raw_data["committer"]:
                user_uid = f"github_contributor_{commit.raw_data['committer']}"
                if user_uid != "github_contributor_web-flow":
                    print(f"committer {user_uid} -> commit {commit_uid}")
                    self.sgraph._add_explicit_edge(
                        source=user_uid,
                        target=commit_uid,
                        rel_type="reviewed",
                        weight=1.0,
                        metadata={"role": "reviewer"},
                    )

            # 修改文件
            if commit.raw_data["files"]:
                for f in commit.raw_data["files"]:
                    file_uid = f"github_code_{f}"
                    print(f"commit {commit_uid} -> file {file_uid}")
                    self.sgraph._add_explicit_edge(
                        source=commit_uid,
                        target=file_uid,
                        rel_type="modified",
                        weight=1.0,
                        metadata={"role": "modified"},
                    )

            # 仓库commit：对应唯一pr
            pattern = r"\(#([1-9]\d*)\)"
            search_result = re.search(pattern, commit.raw_data["message"])
            if search_result:
                pr_id = f"github_pr_{search_result.group(1)}"
                print(f"commit {commit_uid} -> pr {pr_id}")
                self.sgraph._add_explicit_edge(
                    source=commit_uid,
                    target=pr_id,
                    rel_type="corresponding",
                    metadata={"merged_time": commit.raw_data["timestamp"]},
                )

            # 添加仓库commit的时间主链
            if pre_commit:
                pre_commit_id = pre_commit.uid
                print(f"commit {pre_commit_id} -> commit {commit_uid}")
                self.sgraph._add_explicit_edge(
                    source=commit_uid,
                    target=pre_commit_id,
                    rel_type="next",
                )

        # 添加Pr关系
        prs = self.smap.memoryspaces["github_prs"].units.values()
        for pr in prs:
            pr_uid = pr.uid
            pull_request = self.repo.get_pull(int(pr.uid[10:]))

            # 修复Issue
            issue_ids = self._parse_closed_issues(pull_request.body)
            for issue in issue_ids:
                issue_uid = f"github_issue_{issue}"
                print(f"pr {pr_uid} -> issue {issue_uid}")
                self.sgraph._add_explicit_edge(
                    source=pr_uid,
                    target=issue_uid,
                    rel_type="fixes",
                    metadata={
                        "merge_commit": pull_request.merge_commit_sha,
                        "merged_at": pull_request.merged_at,
                    },
                )

            # 提交者
            user_uid = f"github_contributor_{pr.raw_data['user']}"
            print(f"user {user_uid} -> pr {pr_uid}")
            self.sgraph._add_explicit_edge(
                source=user_uid,
                target=pr_uid,
                rel_type="authored",
                weight=1.0,
                metadata={"role": "author"},
            )

            # 审查者
            for reviewer in pr.raw_data["reviewers"]:
                user_uid = f"github_contributor_{reviewer}"
                print(f"reviewer {user_uid} -> pr {pr_uid}")
                self.sgraph._add_explicit_edge(
                    source=user_uid,
                    target=pr_uid,
                    rel_type="reviewed",
                    weight=1.0,
                    metadata={"role": "reviewer"},
                )

            # Pr commit：小子链
            pre_commit = None
            for commit_sha in pr.raw_data["commits"]:
                commit_uid = f"github_commit_{commit_sha}"
                repo_commit = self.repo.get_commit(commit_sha)
                data = GithubParser.parse_github_commit(repo_commit)
                unit = MemoryUnit(uid=commit_uid, raw_data=data)
                self.smap.insert_unit("github_commits", unit)

                print(f"pr {pr_uid} -> commit {commit_uid}")
                self.sgraph._add_explicit_edge(
                    source=pr_uid,
                    target=commit_uid,
                    rel_type="including",
                )

                # 创建子链
                if pre_commit:
                    pre_commit_id = f"github_commit_{pre_commit}"
                    print(f"commit {pre_commit_id} -> commit {commit_uid}")
                    self.sgraph._add_explicit_edge(
                        source=pre_commit_id,
                        target=commit_uid,
                        rel_type="next",
                    )
                pre_commit = commit_sha

            # PR与代码文件的关系
            for git_file in pull_request.get_files():
                if not git_file.filename.endswith(".py"):
                    continue

                file_uid = f"github_code_{git_file.filename}"
                print(f"pr {pr_uid} -> file {file_uid}")
                self.sgraph._add_explicit_edge(
                    source=pr_uid,
                    target=file_uid,
                    rel_type="modifies",
                    metadata={
                        "changes": {
                            "additions": git_file.additions,
                            "deletions": git_file.deletions,
                        }
                    },
                )

                # PR对应Issue影响代码文件
                for issue_id in issue_ids:
                    issue_uid = f"github_issue_{issue_id}"
                    print(f"issue {issue_uid} ·> file {file_uid}")
                    self.sgraph._add_implicit_edge(
                        source=issue_uid,
                        target=file_uid,
                        rel_type="affects",
                        score=0.8,
                    )

        # PR修复Issue的关系还可以从以关闭Issue的评论中提取
        issues = smap.memoryspaces["github_issues"].units.values()
        for issue in issues:
            try:
                i = repo.get_issue(int(issue.uid[13:]))
            except GithubException as e:
                if e.status == 404:
                    logging.warning(f"issue {issue.uid} not found")
                    continue
                else:
                    raise e

            if i.state == "closed" and i.state_reason != "not_planned":
                comments = i.get_comments()
                pattern = r"https://github\.com/microsoft/graphrag/pull/(\d+)(?![\/\d])"
                for comment in comments:
                    match = re.search(pattern, comment.body)
                    if match:
                        pr_num = match.group(1)
                        pr = repo.get_pull(int(pr_num))
                        pr_uid = f"github_pr_{pr_num}"

                        print(f"pr {pr_uid} -> issue {issue.uid}")
                        sgraph._add_explicit_edge(
                            source=pr_uid,
                            target=issue.uid,
                            rel_type="fixes",
                            metadata={
                                "merge_commit": pr.merge_commit_sha,
                                "merged_at": pr.merged_at,
                            },
                        )

        # def _infer_implicit_relationships():
        #     """推断隐式关系"""
        #     self.sgraph.infer_implicit_edges("github_contributors")
        #     self.sgraph.infer_implicit_edges("github_prs")
        #     self.sgraph.infer_implicit_edges("github_issues")
        #     self.sgraph.infer_implicit_edges("github_code")
        #     self.sgraph.infer_implicit_edges("github_commits")
        #     self.sgraph.infer_implicit_edges("github_commits", "github_issues")

        # _infer_implicit_relationships()


if __name__ == "__main__":
    github_client = get_github_client()
    repo = github_client.get_repo("microsoft/graphrag")

    # 初始化记忆系统
    smap = SemanticMap()
    sgraph = SemanticGraph(smap)

    # 创建GitHub数据导入器
    importer = GitHubImporter(sgraph.smap, sgraph, repo)
    importer.import_repo(repo, 99999)

    # 导入真实Python仓库示例数据（需提前安装PyGithub）
    importer._build_relationships()

    # print(smap)
    # for ns in smap.memoryspaces.values():
    #     print(ns)
    #     for unit in ns.units.values():
    #         print(unit)

    sgraph.save_graph("data/github_graph.pkl")
    smap.save_data("data/github_smap.pkl")

    # 测试语义检索
    query = "如何处理HTTP 404错误"
    results = smap.find_similar_units(
        query_text=query, ms_names=["github_issues", "github_prs"], top_k=3
    )

    print(f"与「{query}」相关的GitHub项目记忆：")
    for unit in results:
        print(f"[相似度 {unit.metadata['similarity_score']:.5f}] {unit.uid}")
        print(f"原始数据摘要：{unit.raw_data['title'][:50]}...\n")
