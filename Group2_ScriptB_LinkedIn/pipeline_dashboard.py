"""
Group 2 - Uncertainty Estimation Clustering Dashboard
可视化控制面板 | Visual Control Panel

Usage:
    streamlit run pipeline_dashboard.py
"""

import streamlit as st
import os
import subprocess
import time
from datetime import datetime
import pandas as pd
import json
from pathlib import Path

# ============================================================================
# 页面配置
# ============================================================================
st.set_page_config(
    page_title="Group 2 Pipeline Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 设置工作目录（确保在正确位置）
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# ============================================================================
# 工具函数
# ============================================================================
def count_files(folder, ext=None):
    """统计文件数量（改进版）"""
    if not os.path.exists(folder):
        return 0
    
    try:
        files = os.listdir(folder)
        
        # 如果指定了扩展名
        if ext:
            return len([f for f in files if f.endswith(ext)])
        
        # 统计所有文件（排除文件夹和隐藏文件）
        return len([f for f in files 
                   if os.path.isfile(os.path.join(folder, f)) 
                   and not f.startswith('.')])
    except:
        return 0

def get_folder_status():
    """获取所有文件夹状态"""
    folders = {
        'LinkedIn Posts': ('linkedin_posts', '.txt'),
        'Reddit Posts': ('Reddit_posts', '.txt'),
        'Filtered Posts': ('filtered_posts', '.txt'),
        'Cluster Output': ('cluster_output', None),
        'Visualizations': ('cluster_visualizations', None),
        'Filter Stats': ('filter_stats', None)
    }
    
    status = {}
    for name, (folder, ext) in folders.items():
        count = count_files(folder, ext)
        exists = os.path.exists(folder)
        status[name] = {
            'folder': folder,
            'count': count,
            'exists': exists,
            'status': '✅' if exists and count > 0 else '⚠️' if exists else '❌'
        }
    return status

def run_pipeline_step(script_path, step_name):
    """运行pipeline步骤"""
    if not os.path.exists(script_path):
        st.error(f"❌ Script not found: {script_path}")
        return False
    
    with st.spinner(f'🚀 Running {step_name}...'):
        start = time.time()
        result = subprocess.run(
            ['python', script_path], 
            capture_output=True, 
            text=True,
            cwd=SCRIPT_DIR
        )
        elapsed = time.time() - start
        
        if result.returncode == 0:
            st.success(f'✅ {step_name} completed in {elapsed:.1f}s')
            if result.stdout:
                with st.expander("📋 View Output"):
                    st.code(result.stdout[-2000:], language='text')  # 只显示最后2000字符
            return True
        else:
            st.error(f'❌ {step_name} failed')
            if result.stderr:
                with st.expander("🐛 View Error"):
                    st.code(result.stderr[-2000:], language='text')
            return False

# ============================================================================
# 主界面
# ============================================================================
def main():
    # 标题
    st.title("🎯 Group 2 - Uncertainty Estimation Clustering")
    st.markdown("**Course**: AAI 6610, Fall 2025 | **Institution**: Northeastern University")
    st.markdown("**Repository**: SOKIWCHIO/Group-2-ScriptB")
    
    # 侧边栏
    with st.sidebar:
        st.header("📊 Project Status")
        
        # 当前工作目录
        st.caption(f"📁 Working Dir: `{os.getcwd()}`")
        
        # 刷新按钮
        if st.button("🔄 Refresh Status", width='stretch'):
            st.rerun()
        
        st.divider()
        
        # 显示文件夹状态
        status = get_folder_status()
        for name, info in status.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{info['status']} **{name}**")
            with col2:
                st.write(f"`{info['count']}`")
        
        st.divider()
        
        # 快捷操作
        st.subheader("⚡ Quick Actions")
        
        if st.button("📂 Open Results", width='stretch'):
            folder = os.path.abspath('cluster_output')
            if os.path.exists(folder):
                os.startfile(folder)
            else:
                st.warning("Folder not found")
        
        if st.button("📊 Open Visualizations", width='stretch'):
            folder = os.path.abspath('cluster_visualizations')
            if os.path.exists(folder):
                os.startfile(folder)
            else:
                st.warning("Folder not found")
        
        st.divider()
        
        # 数据统计
        st.subheader("📈 Data Statistics")
        total_raw = status['LinkedIn Posts']['count'] + status['Reddit Posts']['count']
        total_filtered = status['Filtered Posts']['count']
        
        if total_raw > 0:
            retention = (total_filtered / total_raw * 100) if total_raw > 0 else 0
            st.metric("Total Raw", f"{total_raw}")
            st.metric("Filtered", f"{total_filtered}")
            st.metric("Retention", f"{retention:.1f}%")
    
    # ============================================================================
    # 主内容区 - Tabs
    # ============================================================================
    tab1, tab2, tab3, tab4 = st.tabs(["🎮 Run Pipeline", "📊 Results", "📈 Visualizations", "📝 Info"])
    
    # ============================================================================
    # Tab 1: 运行Pipeline
    # ============================================================================
    with tab1:
        st.header("🎮 Pipeline Control Panel")
        
        st.info("💡 Tip: Click any button to run that step. Steps already completed will be skipped automatically.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Data Collection")
            
            if st.button("🔐 Step 0: LinkedIn Auth", 
                        key="step0", width='stretch',
                        help="Get LinkedIn cookies for authentication"):
                run_pipeline_step('pipeline_0_linkedin_cookie.py', 'LinkedIn Auth')
            
            if st.button("📱 Step 1: LinkedIn Crawler", 
                        key="step1", width='stretch',
                        help="Collect ~450 posts from LinkedIn"):
                run_pipeline_step('pipeline_1_crawler_linkedin.py', 'LinkedIn Crawler')
            
            if st.button("🌐 Step 2: Reddit Crawler", 
                        key="step2", width='stretch',
                        help="Collect ~1000 posts from Reddit/StackExchange"):
                run_pipeline_step('pipeline_1_crawler_Reddit.py', 'Reddit Crawler')
        
        with col2:
            st.subheader("🔬 Analysis & Visualization")
            
            if st.button("🔍 Step 3: Semantic Filter", 
                        key="step3", width='stretch',
                        help="Filter relevant posts (threshold=30)"):
                run_pipeline_step('pipeline_2_semantic_filter.py', 'Semantic Filter')
            
            if st.button("🎯 Step 4: Clustering", 
                        key="step4", width='stretch',
                        help="Cluster posts into topics (HDBSCAN + KMeans)"):
                run_pipeline_step('pipeline_3_cluster.py', 'Clustering')
            
            if st.button("📊 Step 5: Visualization", 
                        key="step5", width='stretch',
                        help="Generate PCA, t-SNE, UMAP plots"):
                run_pipeline_step('pipeline_3_cluster_print.py', 'Visualization')
        
        st.divider()
        
        # 批量操作
        st.subheader("⚡ Batch Operations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Run ALL Steps", type="primary", width='stretch'):
                steps = [
                    ('pipeline_1_crawler_linkedin.py', 'LinkedIn Crawler'),
                    ('pipeline_1_crawler_Reddit.py', 'Reddit Crawler'),
                    ('pipeline_2_semantic_filter.py', 'Semantic Filter'),
                    ('pipeline_3_cluster.py', 'Clustering'),
                    ('pipeline_3_cluster_print.py', 'Visualization')
                ]
                
                progress_bar = st.progress(0)
                for i, (script, name) in enumerate(steps):
                    progress_bar.progress((i) / len(steps), f"Running {name}...")
                    if not run_pipeline_step(script, name):
                        st.error(f"Pipeline stopped at {name}")
                        break
                else:
                    progress_bar.progress(1.0, "All steps completed!")
                    st.balloons()
                    st.success("🎉 All steps completed successfully!")
        
        with col2:
            if st.button("🔄 Re-run Analysis", width='stretch',
                        help="Re-run filtering, clustering, visualization"):
                steps = [
                    ('pipeline_2_semantic_filter.py', 'Semantic Filter'),
                    ('pipeline_3_cluster.py', 'Clustering'),
                    ('pipeline_3_cluster_print.py', 'Visualization')
                ]
                for script, name in steps:
                    run_pipeline_step(script, name)
                st.rerun()
        
        with col3:
            if st.button("📊 Viz Only", width='stretch',
                        help="Re-generate visualizations only"):
                if run_pipeline_step('pipeline_3_cluster_print.py', 'Visualization'):
                    st.rerun()
    
    # ============================================================================
    # Tab 2: 结果展示
    # ============================================================================
    with tab2:
        st.header("📊 Clustering Results")
        
        # 读取聚类结果
        clusters_file = 'cluster_output/HDBSCAN_clusters.csv'
        reps_file = 'cluster_output/HDBSCAN_representatives.txt'
        
        if os.path.exists(clusters_file):
            df = pd.read_csv(clusters_file)
            
            # 统计信息
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📄 Total Documents", len(df))
            
            with col2:
                n_clusters = len(df['cluster'].unique()) - (1 if -1 in df['cluster'].values else 0)
                st.metric("🎯 Clusters", n_clusters)
            
            with col3:
                noise = len(df[df['cluster'] == -1]) if -1 in df['cluster'].values else 0
                st.metric("🔇 Noise Points", f"{noise} ({noise/len(df)*100:.1f}%)")
            
            with col4:
                if len(df) > 0:
                    largest = df[df['cluster'] != -1]['cluster'].value_counts().iloc[0] if len(df[df['cluster'] != -1]) > 0 else 0
                    st.metric("📦 Largest Cluster", largest)
            
            st.divider()
            
            # 簇分布图
            st.subheader("📈 Cluster Distribution")
            cluster_counts = df[df['cluster'] != -1]['cluster'].value_counts().sort_index()
            st.bar_chart(cluster_counts)
            
            # 数据表格
            st.subheader("📋 Cluster Breakdown")
            cluster_summary = df.groupby('cluster').size().reset_index(name='count')
            cluster_summary = cluster_summary[cluster_summary['cluster'] != -1]
            cluster_summary['percentage'] = (cluster_summary['count'] / len(df) * 100).round(1)
            st.dataframe(cluster_summary, hide_index=True, width=600)
            
            st.divider()
            
            # 显示代表文本
            if os.path.exists(reps_file):
                st.subheader("📝 Cluster Summaries")
                
                with open(reps_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 按簇分割
                clusters_text = content.split('\n\n')
                
                # 显示前15个簇
                for cluster_text in clusters_text[:15]:
                    if cluster_text.strip():
                        lines = cluster_text.split('\n')
                        if lines:
                            header = lines[0]  # Cluster X (Size: Y texts)
                            with st.expander(f"🔍 {header}"):
                                st.text(cluster_text)
            else:
                st.info("⚠️ Representatives file not found. Run Step 4 (Clustering) first.")
        
        else:
            st.info("⚠️ No clustering results yet. Please run Step 4 (Clustering) first.")
            st.markdown("""
            **To get started**:
            1. If you don't have raw data, run Steps 1-2 (Crawlers)
            2. Run Step 3 (Semantic Filter)
            3. Run Step 4 (Clustering)
            4. Results will appear here!
            """)
    
    # ============================================================================
    # Tab 3: 可视化展示
    # ============================================================================
    with tab3:
        st.header("📈 Cluster Visualizations")
        
        viz_folder = 'cluster_visualizations'
        
        if os.path.exists(viz_folder):
            images = sorted([f for f in os.listdir(viz_folder) if f.endswith('.png')])
            
            if images:
                # 按聚类方法分组
                clustering_methods = {}
                for img in images:
                    if 'HDBSCAN' in img:
                        method = 'HDBSCAN'
                    elif 'KMeans' in img:
                        method = 'KMeans'
                    else:
                        method = 'Other'
                    
                    if method not in clustering_methods:
                        clustering_methods[method] = []
                    clustering_methods[method].append(img)
                
                # 显示每种聚类方法的可视化
                for method, imgs in clustering_methods.items():
                    st.subheader(f"{method} Visualizations")
                    
                    cols = st.columns(min(3, len(imgs)))
                    for i, img in enumerate(imgs):
                        with cols[i % 3]:
                            img_path = os.path.join(viz_folder, img)
                            caption = img.replace('.png', '').replace('_', ' ').upper()
                            st.image(img_path, caption=caption, width=400)
                    
                    st.divider()
                
            else:
                st.info("⚠️ No visualization images found. Run Step 5 first.")
        else:
            st.info("⚠️ Visualization folder not found. Run Step 5 first.")
        
        # 过滤统计图
        st.subheader("🔍 Filtering Statistics")
        filter_viz = 'filter_stats/filtering_analysis.png'
        if os.path.exists(filter_viz):
            st.image(filter_viz, width=800)
        else:
            st.info("⚠️ Filtering statistics not found. Run Step 3 first.")
    
    # ============================================================================
    # Tab 4: 项目信息
    # ============================================================================
    with tab4:
        st.header("📝 Project Information")
        
        # 项目概览
        st.subheader("🎯 Project Overview")
        st.markdown("""
        **Topic**: Uncertainty Estimation in Machine Learning
        
        **Objective**: Collect and cluster social media discussions about ML uncertainty from LinkedIn and Reddit
        
        **Pipeline**:
        1. Data Collection (LinkedIn + Reddit) → 1,519 posts
        2. Semantic Filtering (threshold=30) → 532 posts (35% retention)
        3. Clustering (HDBSCAN + KMeans) → 13-16 topics
        4. Multi-dimensional Visualization (PCA, t-SNE, UMAP)
        """)
        
        st.divider()
        
        # 配置信息
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚙️ Filter Configuration")
            st.code("""
THRESHOLD = 30.0
MODEL = 'all-MiniLM-L6-v2'
BATCH_SIZE = 32

Queries:
- uncertainty estimation
- Bayesian neural networks
- epistemic/aleatoric uncertainty
- confidence calibration
- etc. (10 queries total)
            """, language='python')
        
        with col2:
            st.subheader("⚙️ Clustering Configuration")
            st.code("""
HDBSCAN:
  min_cluster_size = 10
  min_samples = 3
  metric = 'euclidean'

KMeans:
  K range = 2-10 (auto-select)
  metric = silhouette score

Embeddings:
  OpenAI text-embedding-3-large
  Dimensions: 3072 → 50 (UMAP)
            """, language='python')
        
        st.divider()
        
        # 读取filter summary
        st.subheader("📊 Detailed Statistics")
        summary_file = 'filter_stats/filter_summary.json'
        if os.path.exists(summary_file):
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Input", summary.get('total_input', 0))
            with col2:
                st.metric("Filtered Output", summary.get('total_filtered', 0))
            with col3:
                st.metric("Retention Rate", f"{summary.get('filter_rate', 0):.1f}%")
            
            with st.expander("📄 View Full Summary JSON"):
                st.json(summary)
        else:
            st.info("⚠️ Run Step 3 (Semantic Filter) to generate statistics")
        
        st.divider()
        
        # 脚本列表
        st.subheader("📜 Available Scripts")
        scripts = {
            'pipeline_0_linkedin_cookie.py': 'LinkedIn Authentication',
            'pipeline_1_crawler_linkedin.py': 'LinkedIn Crawler',
            'pipeline_1_crawler_Reddit.py': 'Reddit Crawler',
            'pipeline_2_semantic_filter.py': 'Semantic Filter',
            'pipeline_3_cluster.py': 'Clustering Analysis',
            'pipeline_3_cluster_print.py': 'Visualization'
        }
        
        for script, desc in scripts.items():
            exists = os.path.exists(script)
            icon = "✅" if exists else "❌"
            size = f"{os.path.getsize(script)/1024:.1f} KB" if exists else "N/A"
            st.write(f"{icon} **{script}** - {desc} ({size})")

# ============================================================================
# 运行
# ============================================================================
if __name__ == "__main__":
    main()
