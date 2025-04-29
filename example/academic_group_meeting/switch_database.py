#!/usr/bin/env python3
import os
import sys
import subprocess
import re
import argparse

def switch_neo4j_database(database_name):
    """
    修改Neo4j的默认数据库并重启服务
    
    Args:
        database_name: 新的默认数据库名称
    """
    config_file = "/etc/neo4j/neo4j.conf"
    sudo_password = "20031117"  # 固定的sudo密码
    
    print(f"准备将Neo4j默认数据库切换为: {database_name}")
    
    # 检查配置文件是否存在
    if not os.path.exists(config_file):
        print(f"错误：配置文件 {config_file} 不存在")
        return False
    
    try:
        # 读取配置文件内容
        cmd = ["sudo", "-S", "cat", config_file]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        content, err = p.communicate(input=sudo_password + "\n")
        
        if p.returncode != 0:
            print(f"读取配置文件失败: {err}")
            return False
        
        # 查找并替换数据库设置行
        pattern = r'(#?\s*dbms\.default_database\s*=\s*).*'
        
        # 检查是否存在该配置项
        if re.search(pattern, content):
            # 替换现有配置
            new_content = re.sub(pattern, f"\\1{database_name}", content)
            print("找到现有配置项并更新")
        else:
            # 配置项不存在，添加新配置
            new_content = content + f"\n# 设置默认数据库\ndbms.default_database={database_name}\n"
            print("未找到配置项，添加新配置")
        
        # 将新配置写入临时文件
        temp_file = "/tmp/neo4j_temp.conf"
        with open(temp_file, 'w') as f:
            f.write(new_content)
        
        print("配置文件已准备，正在应用更改...")
        
        # 使用sudo复制临时文件到目标位置
        cp_cmd = ["sudo", "-S", "cp", temp_file, config_file]
        p = subprocess.Popen(cp_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        _, err = p.communicate(input=sudo_password + "\n")
        
        if p.returncode != 0:
            print(f"更新配置文件失败: {err}")
            return False
        
        print("配置已更新，正在重启Neo4j服务...")
        
        # 重启Neo4j服务
        restart_cmd = ["sudo", "-S", "systemctl", "restart", "neo4j"]
        p = subprocess.Popen(restart_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        _, err = p.communicate(input=sudo_password + "\n")
        
        if p.returncode != 0:
            print(f"重启Neo4j服务失败: {err}")
            return False
        
        print(f"成功将Neo4j默认数据库切换到 {database_name} 并重启服务")
        print("服务重启可能需要几秒钟才能完全生效")
        return True
            
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='切换Neo4j默认数据库')
    parser.add_argument('--database', help='Neo4j默认数据库名称')
    args = parser.parse_args()
    
    switch_neo4j_database(args.database)

if __name__ == "__main__":
    main()