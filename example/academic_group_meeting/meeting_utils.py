import datetime

class MeetingUtils:
    @staticmethod
    def generate_meeting_summary(meeting_data, discussion_points, decisions):
        """生成会议纪要"""
        now = datetime.datetime.now()
        summary = {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "participants": meeting_data.get("participants", []),
            "discussion_points": discussion_points,
            "decisions": decisions,
            "duration": meeting_data.get("duration", "N/A")
        }
        return summary
    
    @staticmethod
    def generate_tasks(students, meeting_summary):
        """为学生生成并分配任务"""
        tasks = []
        decisions = meeting_summary.get("decisions", [])
        
        # 根据会议决定生成任务
        for i, decision in enumerate(decisions):
            # 选择学生分配任务，简单地循环分配
            student = students[i % len(students)] if students else "未分配"
            
            task = {
                "id": f"TASK-{len(tasks) + 1}",
                "description": f"基于决定 '{decision}' 的任务",
                "assigned_to": student,
                "due_date": (datetime.datetime.now() + datetime.timedelta(days=7)).strftime("%Y-%m-%d"),
                "status": "待处理"
            }
            tasks.append(task)
            
        return tasks
    
    @staticmethod
    def print_meeting_summary(summary):
        """打印会议纪要"""
        print("\n" + "="*50)
        print("会议纪要".center(48))
        print("="*50)
        print(f"日期: {summary['date']}")
        print(f"时间: {summary['time']}")
        print(f"时长: {summary['duration']}")
        print(f"参与者: {', '.join(summary['participants'])}")
        print("\n讨论要点:")
        for i, point in enumerate(summary['discussion_points'], 1):
            print(f"  {i}. {point}")
        print("\n决定:")
        for i, decision in enumerate(summary['decisions'], 1):
            print(f"  {i}. {decision}")
        print("="*50)
    
    @staticmethod
    def print_tasks(tasks):
        """打印任务列表"""
        print("\n" + "="*50)
        print("任务分配".center(48))
        print("="*50)
        for task in tasks:
            print(f"任务ID: {task['id']}")
            print(f"描述: {task['description']}")
            print(f"分配给: {task['assigned_to']}")
            print(f"截止日期: {task['due_date']}")
            print(f"状态: {task['status']}")
            print("-"*30)
        print("="*50)
