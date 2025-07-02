import sys
import os

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

from typing import Dict, Any, List, Optional, Callable
from core.Hippo import SemanticGraph, SemanticMap, MemorySpace, MemoryUnit
import re
import ast
import logging

from collections import defaultdict
from tqdm import tqdm

import argparse

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)


class GithubParser:
    @staticmethod
    def parse_github_issue(issue: Issue) -> Dict:
        """将GitHub Issue转换为内存单元格式"""
        return {
            "number": getattr(issue, "number", None),
            "title": getattr(issue, "title", None),
            "body": getattr(issue, "body", None),
            "created_at": getattr(issue, "created_at", None),
            "updated_at": getattr(issue, "updated_at", None),
            "closed_at": getattr(issue, "closed_at", None),
            "state": getattr(issue, "state", None),
            "labels": [
                getattr(label, "name", None)
                for label in getattr(issue, "get_labels", lambda: [])()
            ],
            "comments": [
                getattr(comment, "body", None)
                for comment in getattr(issue, "get_comments", lambda: [])()
            ],
        }

    @staticmethod
    def parse_github_pr(pr: PullRequest) -> Dict:
        """将GitHub PR转换为内存单元格式"""
        return {
            "number": getattr(pr, "number", None),
            "title": getattr(pr, "title", None),
            "body": getattr(pr, "body", None),
            "user": getattr(getattr(pr, "user", None), "login", None),
            "created_at": getattr(pr, "created_at", None),
            "updated_at": getattr(pr, "updated_at", None),
            "closed_at": getattr(pr, "closed_at", None),
            "state": getattr(pr, "state", None),
            "labels": [
                getattr(label, "name", None)
                for label in getattr(pr, "get_labels", lambda: [])()
            ],
            "files_changed": [
                getattr(f, "filename", None)
                for f in getattr(pr, "get_files", lambda: [])()
            ],
            "commits": [
                getattr(getattr(c, "commit", None), "sha", None)
                for c in getattr(pr, "get_commits", lambda: [])()
            ],
            "reviewers": (
                [
                    getattr(getattr(r, "user", None), "login", None)
                    for r in getattr(pr, "get_reviews", lambda: [])()
                ]
                if getattr(pr, "get_reviews", None)
                else []
            ),
            "merged_at": getattr(pr, "merged_at", None),
        }

    @staticmethod
    def parse_github_comment(comment) -> Dict:
        """解析GitHub评论对象，返回包含文本内容的dict"""
        return {
            "body": getattr(comment, "body", None),
            "user": getattr(getattr(comment, "user", None), "login", None),
            "created_at": getattr(comment, "created_at", None),
            "updated_at": getattr(comment, "updated_at", None),
            "text_content": getattr(comment, "body", None),
        }

    @staticmethod
    def parse_github_review(review) -> Dict:
        """解析GitHub评审对象，返回包含文本内容的dict"""
        return {
            "body": getattr(review, "body", None),
            "user": getattr(getattr(review, "user", None), "login", None),
            "state": getattr(review, "state", None),
            "submitted_at": getattr(review, "submitted_at", None),
            "text_content": getattr(review, "body", None),
        }

    @staticmethod
    def parse_github_content_file(repo: Repository, git_file: ContentFile) -> Dict:
        """解析代码文件内容，若为二进制或无法解码则只保留元数据，不处理内容"""
        content = None
        try:
            content = git_file.decoded_content.decode("utf-8")
        except Exception:
            logging.warning(
                f"Non-text or undecodable file: {getattr(git_file, 'path', None)}, only metadata kept."
            )
        return {
            "filename": getattr(git_file, "path", None),
            "content": content,  # 若为二进制则为None
            "url": getattr(git_file, "_url", None),
            "size": getattr(git_file, "size", None),
            "sha": getattr(git_file, "sha", None),
            "type": getattr(git_file, "type", None),
        }

    @staticmethod
    def parse_github_contributor(contributor: NamedUser) -> Dict:
        def safe_get(attr):
            try:
                return getattr(contributor, attr)
            except GithubException as e:
                if e.status == 404:
                    return None
                else:
                    raise
            except Exception:
                return None

        return {
            "login": safe_get("login"),
            "id": safe_get("id"),
            "avatar_url": safe_get("avatar_url"),
            "url": safe_get("html_url"),
            "email": safe_get("email"),
            "expertise": safe_get("bio"),
            "team": safe_get("company"),
            "activity": {
                "commit_count": safe_get("contributions"),
                "last_active": safe_get("updated_at"),
            },
            "recent_activity": None,
        }

    @staticmethod
    def parse_github_label(label: Label) -> Dict:
        """解析GitHub标签信息"""
        return {
            "color": getattr(label, "color", None),
            "default": getattr(label, "default", None),
            "description": getattr(label, "description", None),
            "id": getattr(label, "id", None),
            "name": getattr(label, "name", None),
            "node_id": getattr(label, "node_id", None),
            "url": getattr(label, "url", None),
        }

    @staticmethod
    def parse_github_commit(commit: Commit) -> Dict:
        """解析GitHub Commit信息"""
        return {
            "sha": getattr(commit, "sha", None),
            "author": getattr(getattr(commit, "author", None), "login", None),
            "committer": getattr(getattr(commit, "committer", None), "login", None),
            "message": getattr(getattr(commit, "commit", None), "message", None),
            "timestamp": getattr(
                getattr(getattr(commit, "commit", None), "author", None), "date", None
            ),
            "files": [
                getattr(f, "filename", None) for f in getattr(commit, "files", [])
            ],
            "url": getattr(commit, "url", None),
        }


class GitHubImporter:
    def __init__(
        self, semantic_map: SemanticMap, semantic_graph: SemanticGraph, repo: Repository
    ):
        self.smap = semantic_map
        self.sgraph = semantic_graph
        self.repo = repo
        # 根节点
        self.ms_repo = MemorySpace("github_repo")
        # 一级空间
        self.ms_issues = MemorySpace("github_issues")
        self.ms_prs = MemorySpace("github_prs")
        self.ms_code_files = MemorySpace("github_code_files")
        self.ms_contributors = MemorySpace("github_contributors")
        self.ms_commits = MemorySpace("github_commits")
        self.ms_reviews = MemorySpace("github_reviews")
        self.ms_comments = MemorySpace("github_comments")
        # 用户空间字典
        self.user_spaces = {}
        # 组织嵌套结构
        self.ms_repo.add(self.ms_issues)
        self.ms_repo.add(self.ms_prs)
        self.ms_repo.add(self.ms_code_files)
        self.ms_repo.add(self.ms_contributors)
        self.ms_repo.add(self.ms_commits)
        self.ms_repo.add(self.ms_reviews)
        self.ms_repo.add(self.ms_comments)

    def get_or_create_user_space(self, login):
        if login not in self.user_spaces:
            ms = MemorySpace(f"user_{login}")
            self.user_spaces[login] = ms
            self.ms_contributors.add(ms)
        return self.user_spaces[login]

    def import_repo(self, repo: Repository, max_item: int = 50):
        print("[导入] 开始导入 Issue...")
        issues = list(repo.get_issues(state="closed"))[:max_item]
        for issue in tqdm(issues, desc="Issues", ncols=80):
            data = GithubParser.parse_github_issue(issue)
            data["text_content"] = "\n".join(
                [data.get("title") or "", data.get("body") or ""]
            )
            uid = f"github_issue_{issue.number}"
            unit = MemoryUnit(uid=uid, raw_data=data)
            self.ms_issues.add(unit)
            user = getattr(getattr(issue, "user", None), "login", None)
            if user:
                self.get_or_create_user_space(user).add(unit)
            # 使用parser处理每条评论
            for idx, comment in enumerate(getattr(issue, "get_comments", lambda: [])()):
                comment_data = GithubParser.parse_github_comment(comment)
                comment_uid = f"github_comment_{issue.number}_{idx}"
                comment_unit = MemoryUnit(
                    uid=comment_uid,
                    raw_data=comment_data,
                )
                self.ms_comments.add(comment_unit)
                if user:
                    self.get_or_create_user_space(user).add(comment_unit)
        print(f"[导入] Issue导入完成，共{len(issues)}条")

        print("[导入] 开始导入 PR...")
        prs = list(repo.get_pulls(state="closed"))[:max_item]
        for pr in tqdm(prs, desc="PRs", ncols=80):
            data = GithubParser.parse_github_pr(pr)
            data["text_content"] = "\n".join(
                [data.get("title") or "", data.get("body") or ""]
            )
            uid = f"github_pr_{pr.number}"
            unit = MemoryUnit(uid=uid, raw_data=data)
            self.ms_prs.add(unit)
            user = data.get("user")
            if user:
                self.get_or_create_user_space(user).add(unit)
            # 使用parser处理每个reviewer（如有get_reviews方法）
            if hasattr(pr, "get_reviews"):
                for idx, review in enumerate(pr.get_reviews()):
                    review_data = GithubParser.parse_github_review(review)
                    review_uid = f"github_review_{pr.number}_{idx}"
                    review_unit = MemoryUnit(uid=review_uid, raw_data=review_data)
                    self.ms_reviews.add(review_unit)
                    reviewer = review_data.get("user")
                    if reviewer:
                        self.get_or_create_user_space(reviewer).add(review_unit)
            else:
                # 兼容旧逻辑
                for idx, reviewer in enumerate(data.get("reviewers", [])):
                    review_uid = f"github_review_{pr.number}_{idx}"
                    review_unit = MemoryUnit(
                        uid=review_uid, raw_data={"reviewer": reviewer, "pr": uid}
                    )
                    self.ms_reviews.add(review_unit)
                    if reviewer:
                        self.get_or_create_user_space(reviewer).add(review_unit)
        print(f"[导入] PR导入完成，共{len(prs)}条")

        print("[导入] 开始导入代码文件...")
        contents = repo.get_contents("")
        code_files = []

        def collect_py_files(contents):
            for content in contents:
                if content.type == "dir":
                    collect_py_files(repo.get_contents(content.path))
                else:
                    code_files.append(content)

        collect_py_files(contents)
        code_files = code_files[:max_item]
        for f in tqdm(code_files, desc="CodeFiles", ncols=80):
            data = GithubParser.parse_github_content_file(repo, f)
            data["text_content"] = data["content"]
            uid = f"github_code_{f.path}"
            unit = MemoryUnit(uid=uid, raw_data=data)
            self.ms_code_files.add(unit)
        print(f"[导入] 代码文件导入完成，共{len(code_files)}个")

        print("[导入] 开始导入贡献者...")
        contributors = list(repo.get_contributors())[:max_item]
        for contributor in tqdm(contributors, desc="Contributors", ncols=80):
            data = GithubParser.parse_github_contributor(contributor)
            data["text_content"] = "\n".join(
                [str(data.get("expertise")) or "", str(data.get("activity")) or ""]
            )
            uid = f"github_contributor_{contributor.login}"
            unit = MemoryUnit(uid=uid, raw_data=data)
            self.ms_contributors.add(unit)
            self.get_or_create_user_space(contributor.login).add(unit)
        print(f"[导入] 贡献者导入完成，共{len(contributors)}人")

        print("[导入] 开始导入Commit...")
        commits = list(repo.get_commits())[:max_item]
        for commit in tqdm(commits, desc="Commits", ncols=80):
            data = GithubParser.parse_github_commit(commit)
            data["text_content"] = data["message"]
            uid = f"github_commit_{commit.sha}"
            unit = MemoryUnit(uid=uid, raw_data=data)
            self.ms_commits.add(unit)
            author = data.get("author")
            if author:
                self.get_or_create_user_space(author).add(unit)
        print(f"[导入] Commit导入完成，共{len(commits)}条")

        print("[导入] 正在注册所有unit到SemanticMap并构建索引...")
        self.smap.register_units_from_space(self.ms_repo)
        self.smap.build_index()
        print("[导入] 注册与索引完成。\n")

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
        commits = self.ms_commits.get_all_units()
        pre_commit = None
        for commit in commits:
            commit_uid = commit.uid
            # 提交者
            if commit.raw_data.get("author"):
                user_uid = f"github_contributor_{commit.raw_data['author']}"
                print(f"author {user_uid} -> commit {commit_uid}")
                if not self.sgraph.semantic_map.get_unit(user_uid):
                    unit = MemoryUnit(uid=user_uid, raw_data={})
                    self.sgraph.add_unit(unit)
                self.sgraph.add_unit(commit)
                self.sgraph.add_explicit_edge(
                    src_uid=user_uid,
                    tgt_uid=commit_uid,
                    rel_type="authored_commit",
                )
            # 审查者
            if commit.raw_data.get("committer"):
                user_uid = f"github_contributor_{commit.raw_data['committer']}"
                if user_uid != "github_contributor_web-flow":
                    print(f"committer {user_uid} -> commit {commit_uid}")
                    if not self.sgraph.semantic_map.get_unit(user_uid):
                        unit = MemoryUnit(uid=user_uid, raw_data={})
                        self.sgraph.add_unit(unit)
                    self.sgraph.add_unit(commit)
                    self.sgraph.add_explicit_edge(
                        src_uid=user_uid,
                        tgt_uid=commit_uid,
                        rel_type="reviewed_commit",
                    )
            # 修改文件
            if commit.raw_data.get("files"):
                for f in commit.raw_data["files"]:
                    file_uid = f"github_code_{f}"
                    print(f"commit {commit_uid} -> file {file_uid}")
                    if not self.sgraph.semantic_map.get_unit(file_uid):
                        # 文件不存在可选创建
                        pass
                    else:
                        self.sgraph.add_unit(commit)
                        self.sgraph.add_explicit_edge(
                            src_uid=commit_uid,
                            tgt_uid=file_uid,
                            rel_type="modified_file",
                        )
            # 仓库commit：对应唯一pr
            pattern = r"\(#([1-9]\d*)\)"
            search_result = re.search(pattern, commit.raw_data["message"])
            if search_result:
                pr_id = f"github_pr_{search_result.group(1)}"
                print(f"commit {commit_uid} -> pr {pr_id}")
                if not self.sgraph.semantic_map.get_unit(pr_id):
                    pr = self.repo.get_pull(int(search_result.group(1)))
                    data = GithubParser.parse_github_pr(pr)
                    unit = MemoryUnit(uid=pr_id, raw_data=data)
                    self.sgraph.add_unit(unit)
                self.sgraph.add_unit(commit)
                self.sgraph.add_explicit_edge(
                    src_uid=commit_uid,
                    tgt_uid=pr_id,
                    rel_type="corresponding_pr",
                    metadata={"merged_time": commit.raw_data["timestamp"]},
                )
            # 添加仓库commit的时间主链
            if pre_commit:
                pre_commit_id = pre_commit.uid
                print(f"commit {pre_commit_id} -> commit {commit_uid}")
                self.sgraph.add_unit(commit)
                self.sgraph.add_unit(pre_commit)
                self.sgraph.add_explicit_edge(
                    src_uid=commit_uid,
                    tgt_uid=pre_commit_id,
                    rel_type="next_repo_commit",
                )
            pre_commit = commit
        # 添加Pr关系
        prs = self.ms_prs.get_all_units()
        for pr in prs:
            pr_uid = pr.uid
            pull_request = self.repo.get_pull(int(pr.uid[10:]))
            # 修复Issue
            issue_ids = self._parse_closed_issues(pull_request.body)
            for issue in issue_ids:
                issue_uid = f"github_issue_{issue}"
                print(f"pr {pr_uid} -> issue {issue_uid}")
                if not self.sgraph.semantic_map.get_unit(issue_uid):
                    issue_data = GithubParser.parse_github_issue(
                        self.repo.get_issue(issue)
                    )
                    unit = MemoryUnit(uid=issue_uid, raw_data=issue_data)
                    self.sgraph.add_unit(unit)
                self.sgraph.add_unit(pr)
                self.sgraph.add_explicit_edge(
                    src_uid=pr_uid,
                    tgt_uid=issue_uid,
                    rel_type="fixes_issue",
                    metadata={
                        "merge_commit": pull_request.merge_commit_sha,
                        "merged_at": pull_request.merged_at,
                    },
                )
            # 提交者
            user_uid = f"github_contributor_{pr.raw_data['user']}"
            print(f"user {user_uid} -> pr {pr_uid}")
            if self.sgraph.semantic_map.get_unit(user_uid):
                self.sgraph.add_unit(pr)
                self.sgraph.add_explicit_edge(
                    src_uid=user_uid,
                    tgt_uid=pr_uid,
                    rel_type="authored_pr",
                )
            # 审查者
            for reviewer in pr.raw_data["reviewers"]:
                user_uid = f"github_contributor_{reviewer}"
                print(f"reviewer {user_uid} -> pr {pr_uid}")
                if self.sgraph.semantic_map.get_unit(user_uid):
                    self.sgraph.add_unit(pr)
                    self.sgraph.add_explicit_edge(
                        src_uid=user_uid,
                        tgt_uid=pr_uid,
                        rel_type="reviewed_pr",
                    )
            # Pr commit：小子链
            pre_commit = None
            for commit_sha in pr.raw_data["commits"]:
                commit_uid = f"github_commit_{commit_sha}"
                if not self.sgraph.semantic_map.get_unit(commit_uid):
                    repo_commit = self.repo.get_commit(commit_sha)
                    data = GithubParser.parse_github_commit(repo_commit)
                    unit = MemoryUnit(uid=commit_uid, raw_data=data)
                    self.sgraph.add_unit(unit)
                self.sgraph.add_unit(pr)
                self.sgraph.add_explicit_edge(
                    src_uid=pr_uid,
                    tgt_uid=commit_uid,
                    rel_type="including_commit",
                )
                # 创建子链
                if pre_commit:
                    pre_commit_id = f"github_commit_{pre_commit}"
                    print(f"commit {pre_commit_id} -> commit {commit_uid}")
                    self.sgraph.add_unit(pr)
                    self.sgraph.add_explicit_edge(
                        src_uid=pre_commit_id,
                        tgt_uid=commit_uid,
                        rel_type="next_commit",
                    )
                pre_commit = commit_sha
            # PR与代码文件的关系
            for git_file in pull_request.get_files():
                file_uid = f"github_code_{git_file.filename}"
                print(f"pr {pr_uid} -> file {file_uid}")
                if self.sgraph.semantic_map.get_unit(file_uid):
                    self.sgraph.add_unit(pr)
                    self.sgraph.add_explicit_edge(
                        src_uid=pr_uid,
                        tgt_uid=file_uid,
                        rel_type="modifies_file",
                        metadata={
                            "changes": {
                                "additions": git_file.additions,
                                "deletions": git_file.deletions,
                            }
                        },
                    )
        # PR修复Issue的关系还可以从以关闭Issue的评论中提取
        issues = self.ms_issues.get_all_units()
        for issue in issues:
            try:
                i = self.repo.get_issue(int(issue.uid[13:]))
            except GithubException as e:
                if e.status == 404:
                    logging.warning(f"issue {issue.uid} not found")
                    continue
                else:
                    raise e
            if (
                i.state == "closed"
                and getattr(i, "state_reason", None) != "not_planned"
            ):
                comments = i.get_comments()
                pattern = (
                    r"https://github\\.com/microsoft/graphrag/pull/(\\d+)(?![\\/\\d])"
                )
                for comment in comments:
                    match = re.search(pattern, comment.body)
                    if match:
                        pr_num = match.group(1)
                        pr = self.repo.get_pull(int(pr_num))
                        pr_uid = f"github_pr_{pr_num}"
                        print(f"pr {pr_uid} -> issue {issue.uid}")
                        if not self.sgraph.semantic_map.get_unit(pr_uid):
                            pr_data = GithubParser.parse_github_pr(pr)
                            unit = MemoryUnit(uid=pr_uid, raw_data=pr_data)
                            self.sgraph.add_unit(unit)
                        self.sgraph.add_unit(issue)
                        self.sgraph.add_explicit_edge(
                            src_uid=pr_uid,
                            tgt_uid=issue.uid,
                            rel_type="fixes_issue",
                            metadata={
                                "merge_commit": pr.merge_commit_sha,
                                "merged_at": pr.merged_at,
                            },
                        )


# 初始化GitHub客户端
def get_github_client(token: str | None = None):
    """创建GitHub API客户端"""
    from dotenv import load_dotenv
    import os

    load_dotenv()  # 从.env文件加载变量
    token = os.getenv("GITHUB_TOKEN")
    if token:
        return Github(token)
    else:  # 匿名访问（有频率限制）
        return Github()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GitHub Analyzer")
    parser.add_argument(
        "--import",
        dest="do_import",
        action="store_true",
        help="从GitHub仓库导入数据并覆盖本地MemorySpace",
    )
    parser.add_argument(
        "--load",
        dest="do_import",
        action="store_false",
        help="仅从本地文件加载MemorySpace（默认）",
    )
    parser.set_defaults(do_import=False)
    args = parser.parse_args()

    github_client = get_github_client()
    repo = github_client.get_repo("microsoft/graphrag")

    graph_dir = "data/issue_manager"
    ms_issues_path = os.path.join(graph_dir, "ms_issues.pkl")
    ms_prs_path = os.path.join(graph_dir, "ms_prs.pkl")
    ms_code_files_path = os.path.join(graph_dir, "ms_code_files.pkl")
    ms_contributors_path = os.path.join(graph_dir, "ms_contributors.pkl")
    ms_commits_path = os.path.join(graph_dir, "ms_commits.pkl")
    ms_reviews_path = os.path.join(graph_dir, "ms_reviews.pkl")
    ms_comments_path = os.path.join(graph_dir, "ms_comments.pkl")

    if args.do_import:
        print("[模式] 从GitHub仓库导入数据...")
        smap = SemanticMap(
            image_embedding_model_name="/mnt/data1/home/guozy/gzy/models/clip-ViT-B-32",
            text_embedding_model_name="/mnt/data1/home/guozy/gzy/models/clip-ViT-B-32-multilingual-v1",
            update_interval=None,
        )
        # 传入所有一级MemorySpace
        sgraph = SemanticGraph(
            smap,
            memory_spaces=[
                importer.ms_issues,
                importer.ms_prs,
                importer.ms_code_files,
                importer.ms_contributors,
                importer.ms_commits,
                importer.ms_reviews,
                importer.ms_comments,
            ],
        )
        importer = GitHubImporter(smap, sgraph, repo)
        importer.import_repo(repo)
        importer._build_relationships()
        # 保存所有MemorySpace和sgraph
        importer.ms_issues.save(ms_issues_path)
        importer.ms_prs.save(ms_prs_path)
        importer.ms_code_files.save(ms_code_files_path)
        importer.ms_contributors.save(ms_contributors_path)
        importer.ms_commits.save(ms_commits_path)
        importer.ms_reviews.save(ms_reviews_path)
        importer.ms_comments.save(ms_comments_path)
        sgraph.save_graph(graph_dir, ms_root=importer.ms_repo)
        ms_issues = importer.ms_issues
        ms_prs = importer.ms_prs
        ms_code_files = importer.ms_code_files
        ms_contributors = importer.ms_contributors
        ms_commits = importer.ms_commits
        ms_reviews = importer.ms_reviews
        ms_comments = importer.ms_comments
    else:
        print("[模式] 仅从本地文件加载数据...")
        # 直接加载sgraph和MemorySpace
        sgraph, ms_root = SemanticGraph.load_graph(graph_dir)
        smap = sgraph.semantic_map

        # 兼容原有变量名
        def get_space_by_name(ms_root, name):
            if ms_root is None:
                return None
            if ms_root.name == name:
                return ms_root
            for member in ms_root._members.values():
                if isinstance(member, MemorySpace):
                    found = get_space_by_name(member, name)
                    if found:
                        return found
            return None

        ms_issues = get_space_by_name(ms_root, "github_issues")
        ms_prs = get_space_by_name(ms_root, "github_prs")
        ms_code_files = get_space_by_name(ms_root, "github_code_files")
        ms_contributors = get_space_by_name(ms_root, "github_contributors")
        ms_commits = get_space_by_name(ms_root, "github_commits")
        ms_reviews = get_space_by_name(ms_root, "github_reviews")
        ms_comments = get_space_by_name(ms_root, "github_comments")
        # 构造importer对象
        importer = GitHubImporter(smap, sgraph, repo)
        if ms_issues is not None:
            importer.ms_issues = ms_issues
        if ms_prs is not None:
            importer.ms_prs = ms_prs
        if ms_code_files is not None:
            importer.ms_code_files = ms_code_files
        if ms_contributors is not None:
            importer.ms_contributors = ms_contributors
        if ms_commits is not None:
            importer.ms_commits = ms_commits
        if ms_reviews is not None:
            importer.ms_reviews = ms_reviews
        if ms_comments is not None:
            importer.ms_comments = ms_comments
        # 传入所有一级MemorySpace
        sgraph.memory_spaces = [
            importer.ms_issues,
            importer.ms_prs,
            importer.ms_code_files,
            importer.ms_contributors,
            importer.ms_commits,
            importer.ms_reviews,
            importer.ms_comments,
        ]
    # 健壮性检查
    for attr in [
        "ms_issues",
        "ms_prs",
        "ms_code_files",
        "ms_contributors",
        "ms_commits",
        "ms_reviews",
        "ms_comments",
    ]:
        ms = getattr(importer, attr, None)
        if ms is None:
            print(f"[警告] 未找到 {attr}，后续统计和操作可能异常。")

    print(smap)
    # sgraph.build_index()

    # 打印嵌套结构
    def print_space(space, indent=0):
        if space is None:
            return
        prefix = "  " * indent
        print(f"{prefix}{space.name} ({len(space.list_members())} members)")
        for k in space.list_members():
            member = space.get(k)
            if isinstance(member, MemorySpace):
                print_space(member, indent + 1)

    print("\n[MemorySpace嵌套结构]")
    for ms in [
        importer.ms_issues,
        importer.ms_prs,
        importer.ms_code_files,
        importer.ms_contributors,
        importer.ms_commits,
        importer.ms_reviews,
        importer.ms_comments,
    ]:
        if ms is not None:
            print_space(ms)

    print("\n[统计信息]")
    print(f"SemanticMap 全局unit数: {len(smap.get_all_units())}")
    print(
        f"SemanticGraph 节点数: {sgraph.nx_graph.number_of_nodes()}，边数: {sgraph.nx_graph.number_of_edges()}"
    )

    for ms in [
        importer.ms_issues,
        importer.ms_prs,
        importer.ms_code_files,
        importer.ms_contributors,
        importer.ms_commits,
        importer.ms_reviews,
        importer.ms_comments,
    ]:
        if ms is not None:
            print(f"{ms.name}: {len(ms.get_all_units())} units")
            ms.build_index()
    print(f"用户空间数: {len(importer.user_spaces)}")

    # 测试语义检索
    query = "如何处理HTTP 404错误"
    results = smap.search_similarity_units_by_text(query_text=query, top_k=3)
    print(f"与「{query}」相关的GitHub项目记忆：")
    for unit, score in results:
        print(f"[相似度 {score:.5f}] {unit.uid}")
        print(f"原始数据摘要：{unit.raw_data.get('text_content', '')[:50]}...\n")

    # === 保存各MemorySpace ===
    for ms, path in [
        (ms_issues, ms_issues_path),
        (ms_prs, ms_prs_path),
        (ms_code_files, ms_code_files_path),
        (ms_contributors, ms_contributors_path),
        (ms_commits, ms_commits_path),
        (ms_reviews, ms_reviews_path),
        (ms_comments, ms_comments_path),
    ]:
        if ms is not None:
            ms.save(path)
