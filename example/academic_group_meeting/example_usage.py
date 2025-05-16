from semantic_map_case.academic_group_meeting.review_generator import generate_comprehensive_review
import datetime

def run_example():
    # 模拟学生数据
    student_data = [
        {"name": "张三", "id": "S001", "performance": "优秀", "attendance": 95},
        {"name": "李四", "id": "S002", "performance": "良好", "attendance": 85},
        {"name": "王五", "id": "S003", "performance": "中等", "attendance": 75}
    ]
    
    # 模拟会议数据
    meeting_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "participants": ["张三", "李四", "王五", "导师"],
        "topics": ["项目进度", "技术难点", "下一步计划"],
        "decisions": ["加快开发进度", "解决UI设计问题", "准备下周演示"],
        "duration": "1小时"
    }
    
    # 生成综合评审，并指定当前阶段为"开发阶段"
    result = generate_comprehensive_review(student_data, meeting_data, "开发阶段")
    
    print("\n生成的评审内容:")
    print(result["review"])

if __name__ == "__main__":
    run_example()
