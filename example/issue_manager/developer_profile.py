import logging
from collections import defaultdict
import os
import pickle
from typing import List, Set, Dict
import datetime

from IssueManagerSemanticMap import MemoryUnit, SemanticMap
from IssueManagerSemanticGraph import SemanticGraph
from util import get_github_client, query_llm
from high_prompt import *

from github import GithubException


global smap, sgraph, repo
smap = SemanticMap()
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


def extract_keywords_from_summaries(summaries, max_keywords=30):
    """
    从文件自然语言总结中提取关键技术词汇和概念

    参数:
    - summaries: list, 文件自然语言总结的列表
    - max_keywords: int, 返回的最大关键词数量

    返回:
    - dict: 关键词及其权重的字典，按权重降序排列
    """
    print(f"Processing extract_keywords_from_summaries...")

    # 合并所有总结为一个文本
    combined_text = " ".join(summaries)

    # 使用TF-IDF提取关键词
    from sklearn.feature_extraction.text import TfidfVectorizer

    # 创建TF-IDF向量化器，排除停用词
    vectorizer = TfidfVectorizer(
        max_features=max_keywords,
        stop_words="english",
        ngram_range=(1, 2),  # 提取单词和双词组合
    )

    # 应用向量化器
    try:
        tfidf_matrix = vectorizer.fit_transform([combined_text])

        # 获取特征名称(词汇)
        feature_names = vectorizer.get_feature_names_out()

        # 获取每个词的TF-IDF分数
        tfidf_scores = tfidf_matrix.toarray()[0]

        # 创建词汇-分数对，并按分数降序排序
        keywords = {
            feature_names[i]: float(tfidf_scores[i]) for i in range(len(feature_names))
        }

        # 按权重降序排序
        keywords = dict(sorted(keywords.items(), key=lambda x: x[1], reverse=True))

    except ValueError:
        # 如果文本太短或其他问题，返回空字典
        keywords = {}

    return keywords


def calculate_average_commit_size(commits: List[MemoryUnit]) -> float:
    """
    计算开发者提交的平均大小(增加/删除的代码行数)

    参数:
    - commits: list, commit节点列表

    返回:
    - float: 平均修改行数
    """
    return 724.96
    print(f"Processing calculate_average_commit_size...")

    if not commits:
        return 0

    total_changes = 0
    for commit in commits:
        # 获取commit的修改行数信息
        c = repo.get_commit(commit.uid[14:])
        additions, deletions = 0, 0
        for f in c.files:
            additions += f.additions
            deletions += f.deletions

        # 总修改行数 = 添加 + 删除
        total_changes += additions + deletions

    # 计算平均值
    result = round(total_changes / len(commits), 2)
    print(f"Average commit size: {result} lines")
    return result


def analyze_commit_timing(commits: List[MemoryUnit]) -> dict:
    """
    分析开发者提交代码的时间模式

    参数:
    - commits: list, commit节点列表

    返回:
    - dict: 时间模式分析结果
    """
    return {
        "peak_hours": [23, 19, 0],
        "peak_days": ["周二", "周三", "周五"],
        "time_preference": "强烈倾向于非工作时间",
        "day_preference": "几乎只在工作日提交",
        "details": {
            "hour_distribution": {
                19: 10,
                18: 6,
                0: 9,
                21: 8,
                22: 7,
                17: 8,
                23: 11,
                1: 3,
                20: 7,
                2: 1,
                3: 1,
                16: 1,
                15: 1,
            },
            "day_distribution": {
                "周二": 20,
                "周三": 15,
                "周一": 11,
                "周五": 15,
                "周四": 11,
                "周六": 1,
            },
        },
    }
    print(f"Processing analyze_commit_timing...")

    if not commits:
        return {"pattern": "无足够数据", "details": {}}

    from collections import Counter
    import datetime

    # 按小时和工作日统计提交次数
    hour_counts = Counter()
    day_counts = Counter()

    for commit in commits:
        commit_date = commit.raw_data["timestamp"]

        # 统计小时
        hour_counts[commit_date.hour] += 1

        # 统计星期几 (0=周一, 6=周日)
        day_counts[commit_date.weekday()] += 1

    # 确定高峰时段
    peak_hours = [hour for hour, count in hour_counts.most_common(3)]
    peak_days = [day for day, count in day_counts.most_common(3)]

    # 将星期几转换为可读格式
    day_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    readable_peak_days = [day_names[day] for day in peak_days]

    # 判断是否倾向于工作时间或非工作时间
    work_hours_count = sum(hour_counts[h] for h in range(9, 18))  # 9am - 6pm
    non_work_hours_count = sum(hour_counts[h] for h in range(18, 24)) + sum(
        hour_counts[h] for h in range(0, 9)
    )

    weekday_count = sum(day_counts[d] for d in range(0, 5))  # 周一至周五
    weekend_count = sum(day_counts[d] for d in range(5, 7))  # 周六和周日

    # 确定时间偏好
    if work_hours_count > non_work_hours_count * 2:
        time_preference = "强烈倾向于工作时间"
    elif work_hours_count > non_work_hours_count:
        time_preference = "倾向于工作时间"
    elif non_work_hours_count > work_hours_count * 2:
        time_preference = "强烈倾向于非工作时间"
    elif non_work_hours_count > work_hours_count:
        time_preference = "倾向于非工作时间"
    else:
        time_preference = "无明显时间偏好"

    # 确定工作日偏好
    if weekday_count > weekend_count * 5:  # 考虑到工作日比周末多
        day_preference = "几乎只在工作日提交"
    elif weekday_count > weekend_count * 2.5:
        day_preference = "主要在工作日提交"
    elif weekend_count > weekday_count / 2.5:
        day_preference = "工作日和周末均有大量提交"
    else:
        day_preference = "主要在工作日提交，偶尔周末提交"

    result = {
        "peak_hours": peak_hours,
        "peak_days": readable_peak_days,
        "time_preference": time_preference,
        "day_preference": day_preference,
        "details": {
            "hour_distribution": dict(hour_counts),
            "day_distribution": {
                day_names[day]: count for day, count in day_counts.items()
            },
        },
    }

    print(f"Commit timing analysis: {result}")

    return result


def calculate_approval_rate(reviewed_prs: List[MemoryUnit]):
    """
    计算开发者审查的PR的通过率

    参数:
    - reviewed_prs: list, 开发者审查过的PR节点列表

    返回:
    - float: 通过率百分比
    """
    print(f"Processing calculate_approval_rate...")

    if not reviewed_prs:
        return 0

    approved_count = 0
    for pr in reviewed_prs:
        pull_request = repo.get_pull(int(pr.uid[10:]))
        # 检查PR的审查状态
        if pull_request.merged:
            approved_count += 1

    # 计算通过率
    approval_rate = (approved_count / len(reviewed_prs)) * 100

    result = round(approval_rate, 2)

    print(f"Approval rate: {result}%")

    return result


def calculate_contribution_frequency(developer_node: MemoryUnit, file_node: MemoryUnit):
    """
    计算开发者对特定文件的贡献频率

    参数:
    - developer_node: 开发者节点
    - file_node: 文件节点

    返回:
    - str: 贡献频率描述
    """
    print(f"Processing calculate_contribution_frequency...")

    # 获取文件的所有修改记录
    all_commits_to_file = [
        sgraph.smap.get_unit_by_uid(r["source"])
        for r in sgraph.find_relations(target=file_node.uid, rel_type="modified")
    ]

    if not all_commits_to_file:
        return "无修改记录"

    # 获取开发者提交的所有commit
    developer_commits = [
        sgraph.smap.get_unit_by_uid(r["target"])
        for r in sgraph.find_relations(source=developer_node.uid, rel_type="authored")
    ]

    # 查找开发者对该文件的修改
    developer_commits_to_file: List[MemoryUnit] = []
    for commit in all_commits_to_file:
        if any(dc.uid == commit.uid for dc in developer_commits):
            developer_commits_to_file.append(commit)

    # 计算开发者修改占比
    contribution_ratio = len(developer_commits_to_file) / len(all_commits_to_file)

    # 根据最后修改时间判断是否是最近活跃
    is_recent_contributor = False
    if developer_commits_to_file:
        # 获取最近的commit日期
        last_commit_date = max(
            commit.raw_data["timestamp"] for commit in developer_commits_to_file
        )

        # 检查是否在过去90天内有提交
        now = datetime.datetime.now(datetime.timezone.utc)
        is_recent_contributor = (now - last_commit_date).days <= 90

    # 确定贡献频率描述
    if contribution_ratio > 0.5 and is_recent_contributor:
        return "主要维护者(贡献>50%，近期活跃)"
    elif contribution_ratio > 0.5:
        return "历史主要贡献者(贡献>50%)"
    elif contribution_ratio > 0.25 and is_recent_contributor:
        return "活跃贡献者(贡献>25%，近期活跃)"
    elif contribution_ratio > 0.25:
        return "重要历史贡献者(贡献>25%)"
    elif contribution_ratio > 0.1 and is_recent_contributor:
        return "近期参与者(贡献>10%，近期活跃)"
    elif contribution_ratio > 0.1:
        return "偶尔贡献者(贡献>10%)"
    elif is_recent_contributor:
        return "最近有少量贡献"
    else:
        return "历史上有少量贡献"


def get_top_files(modified_files: List[MemoryUnit], n=10):
    """
    获取开发者贡献最多的前N个文件

    参数:
    - modified_files: set, 开发者修改过的文件节点集合
    - n: int, 返回的文件数量

    返回:
    - list: 按贡献度排序的文件节点列表
    """
    print(f"Processing get_top_files...")

    # 如果文件数量不足n，直接返回所有文件
    if len(modified_files) <= n:
        return list(modified_files)

    # 计算每个文件的修改频率
    file_modification_count = {}
    for f in modified_files:
        # 获取修改该文件的所有commits
        modifying_commits = [
            sgraph.smap.get_unit_by_uid(r["source"])
            for r in sgraph.find_relations(target=f.uid, rel_type="modified")
        ]
        modifying_commits = [commit for commit in modifying_commits if commit]
        file_modification_count[f] = len(modifying_commits)

    # 根据修改频率排序
    sorted_files = sorted(
        modified_files, key=lambda f: file_modification_count.get(f, 0), reverse=True
    )

    result = sorted_files[:n]

    print(f"Top {n} files: {result}")

    return result


def analyze_collaboration_patterns(
    developer_node: MemoryUnit, profile: DeveloperProfile
):
    """
    分析开发者的协作模式

    参数:
    - developer_node: 开发者节点
    - sgraph: 语义图实例

    返回:
    - list: 协作模式描述列表
    """
    print(f"Processing analyze_collaboration_patterns...")
    patterns = []

    # 获取开发者的PRs
    submitted_prs: List[MemoryUnit] = [
        sgraph.smap.get_unit_by_uid(r["target"])
        for r in sgraph.find_relations(source=developer_node.uid, rel_type="authored")
        if r and r["target"].startswith("github_pr_")
    ]

    # 获取开发者审查的PRs
    reviewed_prs: List[MemoryUnit] = [
        sgraph.smap.get_unit_by_uid(r["target"])
        for r in sgraph.find_relations(source=developer_node.uid, rel_type="reviewed")
        if r and r["target"].startswith("github_pr_")
    ]

    # 分析PR大小偏好
    pr_sizes = []
    for pr in submitted_prs:
        # 获取PR包含的commit数量
        commit_count = len(pr.raw_data["commits"])

        # 获取修改的文件数量
        file_count = len(pr.raw_data["files_changed"])

        # 简单的PR大小计算
        size = commit_count * 0.3 + file_count * 0.7
        pr_sizes.append(size)

    # 确定PR大小偏好
    if pr_sizes:
        avg_size = sum(pr_sizes) / len(pr_sizes)
        if avg_size < 3:
            patterns.append({"description": "偏好小型、聚焦的PR", "frequency": "常见"})
        elif avg_size < 7:
            patterns.append({"description": "偏好中等大小的PR", "frequency": "常见"})
        else:
            patterns.append({"description": "偏好大型、综合性PR", "frequency": "常见"})
    print(f"PR size preference: {patterns}")

    # 分析审查活跃度
    if len(reviewed_prs) > len(submitted_prs) * 2:
        patterns.append({"description": "非常活跃的代码审查者", "frequency": "高"})
    elif len(reviewed_prs) > len(submitted_prs):
        patterns.append({"description": "积极参与代码审查", "frequency": "中等"})
    print(f"Review activity: {patterns}")

    # 分析与其他开发者的协作
    collaborators = set()

    # 查找其PR的审查者
    for pr in submitted_prs:
        reviewers = [
            sgraph.smap.get_unit_by_uid(f"github_contributor_{c}")
            for c in pr.raw_data["reviewers"]
            if c
        ]
        collaborators.update([r for r in reviewers if r.uid != developer_node.uid])

    # 查找他审查的PR的提交者
    for pr in reviewed_prs:
        author = sgraph.smap.get_unit_by_uid(
            f"github_contributor_{pr.raw_data['user']}"
        )
        collaborators.update([author])

    # 协作广度
    if len(collaborators) > 10:
        patterns.append({"description": "广泛的团队协作", "frequency": "高"})
    elif len(collaborators) > 5:
        patterns.append({"description": "适度的团队协作", "frequency": "中等"})
    elif len(collaborators) > 0:
        patterns.append({"description": "与少数开发者紧密协作", "frequency": "中等"})
    else:
        patterns.append({"description": "独立工作者", "frequency": "高"})
    print(f"Collaboration breadth: {patterns}")

    # 分析是否解决issues
    fixed_issues: Set[MemoryUnit] = set()
    for pr in submitted_prs:
        issues = [
            sgraph.smap.get_unit_by_uid(r["target"])
            for r in sgraph.find_relations(target=pr.uid, rel_type="fixes")
        ]
        issues = [i for i in issues if i]
        fixed_issues.update(issues)

    if len(fixed_issues) > len(submitted_prs) * 0.8:
        patterns.append({"description": "高度关注问题修复", "frequency": "高"})
    elif len(fixed_issues) > len(submitted_prs) * 0.5:
        patterns.append({"description": "积极参与问题修复", "frequency": "中等"})
    elif len(fixed_issues) > 0:
        patterns.append({"description": "偶尔解决问题", "frequency": "低"})
    print(f"Issue resolution: {patterns}")

    # 分析代码文件类型偏好
    file_types = {}
    file_types = profile.profile["technical_expertise"]["proficient_languages"]

    # 找出最常用的文件类型
    if file_types:
        top_type, top_count = max(file_types.items(), key=lambda x: x[1])
        total_files = sum(file_types.values())
        if top_count > total_files * 0.6:
            patterns.append(
                {"description": f"专注于{top_type}文件开发", "frequency": "高"}
            )
    print(f"File type preference: {patterns}")

    return patterns


def build_developer_expertise_profile(
    developer_node: MemoryUnit, developer_profile: DeveloperProfile
):
    # 1. 显式结构边分析 - 贡献活动与代码修改
    authored_commits: List[MemoryUnit] = [
        sgraph.smap.get_unit_by_uid(r["target"])
        for r in sgraph.find_relations(source=developer_node.uid, rel_type="authored")
        if r and r["target"].startswith("github_commit_")
    ]
    authored_commits = [commit for commit in authored_commits if commit]
    authored_prs: List[MemoryUnit] = [
        sgraph.smap.get_unit_by_uid(r["target"])
        for r in sgraph.find_relations(source=developer_node.uid, rel_type="authored")
        if r and r["target"].startswith("github_pr_")
    ]
    authored_prs = [pr for pr in authored_prs if pr]
    reviewed_prs: List[MemoryUnit] = [
        sgraph.smap.get_unit_by_uid(r["target"])
        for r in sgraph.find_relations(source=developer_node.uid, rel_type="reviewed")
        if r
    ]
    reviewed_prs = [pr for pr in reviewed_prs if pr]
    print(f"Authored commits: {len(authored_commits)}")
    print(f"Authored PRs: {len(authored_prs)}")
    print(f"Reviewed PRs: {len(reviewed_prs)}")

    # 2. 获取修改的代码文件及其自然语言总结
    modified_files: List[MemoryUnit] = []
    for commit in authored_commits:
        files: List[MemoryUnit] = [
            sgraph.smap.get_unit_by_uid(r["target"])
            for r in sgraph.find_relations(source=commit.uid, rel_type="modified")
            if r
        ]
        files = [f for f in files if f]
        modified_files.extend(files)

    # 添加PR修改的文件
    for pr in reviewed_prs:
        files: List[MemoryUnit] = [
            sgraph.smap.get_unit_by_uid(r["target"])
            for r in sgraph.find_relations(source=pr.uid, rel_type="modifies")
            if r
        ]
        files = [f for f in files if f]
        modified_files.extend(files)
    modified_files = list(set(modified_files))  # 去重
    print(f"Modified files: {len(modified_files)}")

    # 3. 隐式语义边分析 - 技能关联与知识图谱
    # 通过隐式语义边找出与开发者提交内容相似的其他内容
    semantic_connections: List[MemoryUnit] = []
    for f in modified_files:
        # 查找与该文件语义相似的其他文件
        similar_nodes: List[MemoryUnit] = sgraph.smap.find_similar_units(
            query_text=f.raw_data["description"], ms_names=["github_code"], top_k=3
        )
        semantic_connections.extend(similar_nodes)
    semantic_connections = list(set(semantic_connections))
    print(f"Semantic connections: {len(semantic_connections)}")

    # 4. 结构与语义的融合分析
    # 提取开发者修改的文件类型分布
    file_types: Dict = developer_profile.profile["technical_expertise"][
        "proficient_languages"
    ]
    # 将文件类型计数转换为百分比
    total_count = sum(file_types.values())
    for file_type, count in file_types.items():
        percentage = round((count / total_count) * 100, 2)
        file_types[file_type] = percentage
    print(f"File types: {file_types}")

    # # 从文件自然语言总结中提取技能关键词
    # expertise_keywords = extract_keywords_from_summaries(
    #     [f.raw_data["description"] for f in modified_files]
    # )

    # # 基于隐式语义边拓展技能关联网络
    # extended_keywords = extract_keywords_from_summaries(
    #     [node.raw_data["description"] for node in semantic_connections]
    # )

    # 5. 使用LLM生成开发者画像
    profile = developer_profile_prompt(
        developer=developer_node,
        commit_stats={
            "authored": len(authored_commits),
            "typical_size": calculate_average_commit_size(authored_commits),
            "preferred_times": analyze_commit_timing(authored_commits),
        },
        pr_stats={
            "submitted": len(reviewed_prs),
            "reviewed": len(authored_prs),
            "approval_rate": calculate_approval_rate(authored_prs),
        },
        core_files=[
            {
                "path": f.raw_data["filename"],
                "summary": f.raw_data["description"],
                "contribution_frequency": calculate_contribution_frequency(
                    developer_node, f
                ),
            }
            for f in get_top_files(modified_files, n=10)
        ],
        # primary_expertise=expertise_keywords,
        # extended_expertise=extended_keywords,
        file_type_distribution=file_types,
        collaboration_patterns=analyze_collaboration_patterns(
            developer_node, developer_profile
        ),
    )

    return profile


if __name__ == "__main__":
    # summary_developer_profile()

    # 初始化，加载开发者profile
    profiles = {}
    with open(
        "/mnt/data1/home/guozy/gzy/SDS/multimodal_semantic_map_dev/issue_manager/data/developer_profiles.pkl",
        "rb",
    ) as f:
        profiles = pickle.load(f)

    # 高阶记忆：开发者画像
    name = "natoverse"
    contributor = smap.get_unit_by_uid(f"github_contributor_{name}")
    profile = profiles.get(name)
    build_developer_expertise_profile(contributor, profile)
