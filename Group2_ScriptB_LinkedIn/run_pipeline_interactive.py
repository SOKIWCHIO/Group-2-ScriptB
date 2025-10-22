"""
========================================
Group 2 - Uncertainty Estimation Clustering
交互式Pipeline控制器 | Interactive Pipeline Controller
========================================

运行后会显示菜单，让你选择要执行的步骤
"""

import os
import sys
import time
from datetime import datetime

# ============================================================================
# 配置
# ============================================================================
class Config:
    SCRIPTS = {
        '0': ('pipeline_0_linkedin_cookie.py', 'LinkedIn Cookie Authentication'),
        '1': ('pipeline_1_crawler_linkedin.py', 'LinkedIn Post Crawler'),
        '2': ('pipeline_1_crawler_Reddit.py', 'Reddit/StackExchange Crawler'),
        '3': ('pipeline_2_semantic_filter.py', 'Semantic Filtering'),
        '4': ('pipeline_3_cluster.py', 'Clustering Analysis'),
        '5': ('pipeline_3_cluster_print.py', 'Multi-dimensional Visualization')
    }
    
    FOLDERS = {
        'linkedin_posts': 'linkedin_posts',
        'Reddit_posts': 'Reddit_posts',
        'filtered_posts': 'filtered_posts',
        'cluster_output': 'cluster_output',
        'cluster_visualizations': 'cluster_visualizations',
        'filter_stats': 'filter_stats'
    }

# ============================================================================
# 工具函数
# ============================================================================
def count_files(folder, ext=".txt"):
    """统计文件数量"""
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith(ext)])

def check_folder_status(folder):
    """检查文件夹状态"""
    if not os.path.exists(folder):
        return "❌ Not found", 0
    count = count_files(folder)
    if count == 0:
        return "⚠️  Empty", 0
    return "✅ Ready", count

def run_script(script_path, description):
    """运行脚本"""
    print(f"\n{'='*80}")
    print(f"🚀 Running: {description}")
    print(f"   Script: {script_path}")
    print(f"{'='*80}\n")
    
    start = time.time()
    exit_code = os.system(f'python "{script_path}"')
    elapsed = time.time() - start
    
    if exit_code == 0:
        print(f"\n✅ Completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n❌ Failed with exit code {exit_code}")
        return False

# ============================================================================
# 显示状态
# ============================================================================
def show_status():
    """显示当前项目状态"""
    config = Config()
    
    print("\n" + "="*80)
    print("📊 Current Project Status")
    print("="*80)
    
    print("\n📁 Data Folders:")
    print(f"{'Folder':<30} {'Status':<15} {'Files':<10}")
    print("-" * 60)
    
    for name, folder in config.FOLDERS.items():
        status, count = check_folder_status(folder)
        print(f"{folder:<30} {status:<15} {count:<10}")
    
    print("\n📜 Available Scripts:")
    print(f"{'ID':<5} {'Script':<40} {'Status':<10}")
    print("-" * 60)
    
    for script_id, (script_path, desc) in config.SCRIPTS.items():
        status = "✅ Found" if os.path.exists(script_path) else "❌ Missing"
        print(f"{script_id:<5} {script_path:<40} {status:<10}")
    
    print("\n" + "="*80)

# ============================================================================
# 显示菜单
# ============================================================================
def show_menu():
    """显示主菜单"""
    config = Config()
    
    print("\n" + "="*80)
    print("🎯 Pipeline Menu - Select Steps to Run")
    print("="*80)
    
    for script_id, (script_path, desc) in config.SCRIPTS.items():
        exists = "✅" if os.path.exists(script_path) else "❌"
        print(f"  [{script_id}] {exists} {desc}")
    
    print("\n" + "-"*80)
    print("  [A] Run ALL steps (complete pipeline)")
    print("  [S] Show current status")
    print("  [Q] Quit")
    print("="*80)

# ============================================================================
# 主函数
# ============================================================================
def main():
    """交互式主函数"""
    config = Config()
    
    # 欢迎信息
    print("\n" + "="*80)
    print("  Group 2 - Uncertainty Estimation Clustering Pipeline")
    print("  Interactive Mode | AAI 6610 Fall 2025")
    print("="*80)
    
    # 首次显示状态
    show_status()
    
    while True:
        show_menu()
        
        # 获取用户输入
        choice = input("\n👉 Enter your choice (e.g., '1,3,4' or 'A' or 'S'): ").strip().upper()
        
        # 退出
        if choice == 'Q':
            print("\n👋 Goodbye!")
            break
        
        # 显示状态
        if choice == 'S':
            show_status()
            continue
        
        # 运行所有步骤
        if choice == 'A':
            print("\n🚀 Running ALL steps in sequence...")
            for script_id in sorted(config.SCRIPTS.keys()):
                script_path, desc = config.SCRIPTS[script_id]
                if not run_script(script_path, desc):
                    print(f"\n❌ Pipeline stopped at step {script_id}")
                    break
            else:
                print("\n" + "="*80)
                print("✅ ALL STEPS COMPLETED!")
                print("="*80)
            
            show_status()
            continue
        
        # 运行选定的步骤
        if choice:
            # 解析选择（支持 "1,3,4" 或 "1 3 4" 或 "134"）
            selected = []
            for char in choice.replace(',', ' ').replace('-', ' '):
                if char.strip() in config.SCRIPTS:
                    selected.append(char.strip())
            
            if not selected:
                print("❌ Invalid choice. Please try again.")
                continue
            
            # 按顺序运行
            selected = sorted(set(selected))
            print(f"\n🎯 Will run steps: {', '.join(selected)}")
            
            confirm = input("Continue? (Y/n): ").strip().lower()
            if confirm and confirm != 'y':
                print("Cancelled.")
                continue
            
            # 执行
            for script_id in selected:
                script_path, desc = config.SCRIPTS[script_id]
                
                if not os.path.exists(script_path):
                    print(f"\n❌ Script not found: {script_path}")
                    continue
                
                if not run_script(script_path, desc):
                    response = input("\n⚠️  Step failed. Continue to next step? (y/N): ").strip().lower()
                    if response != 'y':
                        break
            
            # 显示更新后的状态
            show_status()

# ============================================================================
# 命令行模式（非交互）
# ============================================================================
def run_command_line():
    """命令行模式"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run pipeline steps')
    parser.add_argument('steps', nargs='*', help='Step IDs to run (e.g., 1 3 4)')
    parser.add_argument('--all', '-a', action='store_true', help='Run all steps')
    parser.add_argument('--status', '-s', action='store_true', help='Show status only')
    args = parser.parse_args()
    
    config = Config()
    
    # 只显示状态
    if args.status:
        show_status()
        return
    
    # 运行所有
    if args.all:
        for script_id in sorted(config.SCRIPTS.keys()):
            script_path, desc = config.SCRIPTS[script_id]
            run_script(script_path, desc)
        return
    
    # 运行指定步骤
    if args.steps:
        for step_id in args.steps:
            if step_id in config.SCRIPTS:
                script_path, desc = config.SCRIPTS[step_id]
                run_script(script_path, desc)
            else:
                print(f"❌ Invalid step ID: {step_id}")
        return
    
    # 如果没有参数，进入交互模式
    main()

# ============================================================================
# 入口
# ============================================================================
if __name__ == "__main__":
    # 如果有命令行参数，使用命令行模式
    if len(sys.argv) > 1:
        run_command_line()
    else:
        # 否则进入交互模式
        main()
