import streamlit as st
import numpy as np
import pandas as pd

# 页面配置
st.set_page_config(page_title="高反隔热涂层 | 工业级正向测算平台", layout="wide")

# ==========================================
# 1. 升级版材料数据库 (包含真实物理密度 g/cm³)
# ==========================================
MATERIAL_DB = {
    '无 (不添加)': {'n': 1.0, 'uv_abs': 0.0, 'density': 1.0},
    '硫酸钡 (BaSO4)': {'n': 1.64, 'uv_abs': 0.0, 'density': 4.5},
    '氧化铝 (Al2O3)': {'n': 1.76, 'uv_abs': 0.0, 'density': 3.9},
    '氧化锆 (ZrO2)': {'n': 2.15, 'uv_abs': 0.0, 'density': 5.7},
    '氧化镧 (La2O3)': {'n': 1.95, 'uv_abs': 0.0, 'density': 6.5},
    '金红石钛白粉 (TiO2)': {'n': 2.70, 'uv_abs': 0.85, 'density': 4.1},
    '高岭土 (Kaolin)': {'n': 1.56, 'uv_abs': 0.0, 'density': 2.6},
    '碳酸钙/重钙 (CaCO3)': {'n': 1.59, 'uv_abs': 0.0, 'density': 2.7},
    '二氧化硅 (SiO2)': {'n': 1.46, 'uv_abs': 0.0, 'density': 2.2}
}

# ==========================================
# 2. 核心物理测算引擎 (包含自动体积换算与 K-M 方程)
# ==========================================
def predict_coating_performance_from_mass(resin_n, resin_mass, resin_solid, resin_density, active_fillers, thickness, porosity):
    # --- A. 质量转体积与真实 PVC 计算 ---
    # 1. 算干树脂真实体积
    v_resin = (resin_mass * (resin_solid / 100.0)) / resin_density
    
    # 2. 算各粉体真实体积
    total_filler_volume = 0.0
    for f in active_fillers:
        rho_i = MATERIAL_DB[f['mat']]['density']
        v_i = f['mass'] / rho_i 
        f['true_volume'] = v_i
        total_filler_volume += v_i

    # 3. 如果完全没有加粉体
    if total_filler_volume == 0:
        base_E = 0.85 + (thickness/500.0)*0.08 + (porosity/100.0)*0.02
        return 5.0, min(0.99, base_E), 0.0, []

    # 4. 自动计算系统真实的干膜颜料体积浓度 (PVC)
    calculated_pvc = (total_filler_volume / (v_resin + total_filler_volume)) * 100.0


    # --- B. 多组分 Mie 散射物理池化 ---
    SCATTERING_MULTIPLIER = 12.0  
    THICKNESS_SCALE = 65.0        
    PENALTY_FACTOR = 0.15         

    n_host = resin_n * (1.0 - porosity/100.0) + 1.0 * (porosity/100.0)
    tir = 1.0 + (porosity/100.0 * 1.5) 
    
    pooled_scattering = 0.0
    pooled_uv_abs = 0.0
    diagnostics = []

    for f in active_fillers:
        # 该粉体在所有粉体包中的体积占比
        vol_ratio = f['true_volume'] / total_filler_volume 
        
        n_i = MATERIAL_DB[f['mat']]['n']
        uv_abs_i = MATERIAL_DB[f['mat']]['uv_abs']
        size_i = f['size']
        
        delta_n_i = max(0.01, n_i - n_host)
        opt_size_i = 0.5 / (2 * delta_n_i)
        size_efficiency_i = max(0.1, 1.0 - abs(size_i - opt_size_i) / opt_size_i * 0.6)
        
        scatter_i = (delta_n_i * SCATTERING_MULTIPLIER) * size_efficiency_i * tir
        
        pooled_scattering += scatter_i * vol_ratio
        pooled_uv_abs += uv_abs_i * vol_ratio
        
        diagnostics.append({
            '材料': f['mat'].split(' ')[0],
            '实际质量(份)': f['mass'],
            '体积占比': f"{vol_ratio*100:.1f}%",
            '实际粒径': f"{size_i} μm",
            '最佳理论粒径': f"{opt_size_i:.2f} μm",
            '散射贡献度': scatter_i * vol_ratio
        })

    # --- C. K-M 渐进方程推演最终光学指标 ---
    optical_thickness = pooled_scattering * (calculated_pvc / 100.0) * (thickness / THICKNESS_SCALE)
    
    base_fresnel = 5.0
    solar_uv_fraction = 5.0
    max_r = 100.0 - (solar_uv_fraction * pooled_uv_abs) 
    
    # 太阳光反射比预测
    R_pred = base_fresnel + (max_r - base_fresnel) * (1 - np.exp(-optical_thickness))
    
    # 高浓度干涉惩罚 (CPVC 修正)
    penalty = max(0, (calculated_pvc - 45.0) * PENALTY_FACTOR)
    R_pred = max(5.0, R_pred - penalty)
    
    # 大气窗口发射率预测
    E_pred = 0.85 + (thickness/500.0)*0.08 + (calculated_pvc/100.0)*0.03 + (porosity/100.0)*0.02
    E_pred = min(0.99, E_pred)
    
    return R_pred, E_pred, calculated_pvc, diagnostics

# ==========================================
# 3. 网页交互界面
# ==========================================
st.title("🔬 高反隔热涂层 | 工业级正向测算平台")
st.markdown("完全贴合化工厂**打板称重习惯（质量份）**。系统自动进行密度-体积折算，并依据 K-M 物理方程输出光学性能。")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.header("📝 第一步：输入车间配方表")
    
    with st.container(border=True):
        st.subheader("1. 连续相 (成膜物质)")
        c1, c2, c3 = st.columns(3)
        with c1:
            resin_mass = st.number_input("乳液添加量 (质量份)", min_value=0.0, value=20.0, step=1.0)
        with c2:
            resin_solid = st.number_input("乳液固含 (%)", min_value=10.0, max_value=100.0, value=48.0, step=1.0)
        with c3:
            resin_density = st.number_input("干树脂密度 (g/cm³)", min_value=0.8, max_value=2.0, value=1.05, step=0.01)
            
        c4, c5 = st.columns(2)
        with c4:
            resin_n = st.slider("干树脂折射率 (n)", 1.40, 1.60, 1.50, 0.01)
        with c5:
            porosity = st.slider("成膜微观孔隙率 (%)", 0.0, 50.0, 5.0, 1.0)

    with st.container(border=True):
        st.subheader("2. 离散相 (填料加料单 - 最多5种)")
        
        tabs = st.tabs(["填料 1", "填料 2", "填料 3", "填料 4", "填料 5"])
        active_fillers = []
        
        for i, tab in enumerate(tabs):
            with tab:
                default_mat_idx = 1 if i == 0 else 0
                mat = st.selectbox("选择填料材质", list(MATERIAL_DB.keys()), index=default_mat_idx, key=f"mat_{i}")
                
                if mat != '无 (不添加)':
                    col_a, col_b = st.columns(2)
                    with col_a:
                        # 输入纯质量份
                        mass = st.number_input("添加量 (质量份)", 0.0, 500.0, 50.0 if i==0 else 10.0, 1.0, key=f"mass_{i}")
                    with col_b:
                        size = st.slider("主体粒径 (μm)", 0.1, 20.0, 1.0, 0.1, key=f"size_{i}")
                    
                    if mass > 0:
                        active_fillers.append({'mat': mat, 'size': size, 'mass': mass})

    with st.container(border=True):
        st.subheader("3. 施工控制")
        thickness = st.slider("目标干膜厚度 (μm)", 50, 600, 200, 10)

with col2:
    st.header("📊 第二步：测算结果与体系诊断")
    
    # 执行测算
    R_pred, E_pred, calculated_pvc, diagnostics = predict_coating_performance_from_mass(
        resin_n, resin_mass, resin_solid, resin_density, active_fillers, thickness, porosity
    )
    
    # 顶部核心指标卡片
    st.markdown("### 🎯 终端光学与热力学指标")
    m1, m2 = st.columns(2)
    with m1:
        st.metric(label="预估太阳光反射比 (R_sol)", value=f"{R_pred:.2f} %")
    with m2:
        st.metric(label="预估大气窗口发射率 (ε)", value=f"{E_pred:.3f}")
        
    st.divider()
    
    st.markdown("### 🧪 配方体系内在物理参数诊断")
    
    # 醒目展示后台算出的真实 PVC
    if calculated_pvc > 60.0:
        st.error(f"**真实干膜颜料体积浓度 (PVC)**：{calculated_pvc:.1f}% (⚠️ 已处于超临界状态，成膜可能发脆，建议引入适量孔隙率预设)")
    elif calculated_pvc > 40.0:
        st.warning(f"**真实干膜颜料体积浓度 (PVC)**：{calculated_pvc:.1f}% (接近临界点 CPVC)")
    else:
        st.success(f"**真实干膜颜料体积浓度 (PVC)**：{calculated_pvc:.1f}% (体系健康，成膜致密)")
        
    if not diagnostics:
        st.info("尚未添加填料，系统目前评估为纯树脂清漆状态。")
    else:
        df_diag = pd.DataFrame(diagnostics)
        best_filler = df_diag.loc[df_diag['散射贡献度'].idxmax()]['材料']
        
        st.info(f"**光学贡献分析**：在此配比下，**{best_filler}** 提供了最核心的光学遮盖力。各粉体在干膜内的实际体积排布如下：")
        
        df_display = df_diag[['材料', '实际质量(份)', '体积占比', '实际粒径', '最佳理论粒径']]
        st.dataframe(df_display, use_container_width=True, hide_index=True)
