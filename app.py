import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="高反隔热涂层 | 工业级正向测算平台", layout="wide")

# ==========================================
# 1. 材料数据库 (回归纯净版，剔除白度变量)
# ==========================================
MATERIAL_DB = {
    '无 (不添加)': {'n': 1.0, 'uv_abs': 0.0, 'density': 1.0, 'oa': 0.0},
    '硫酸钡 (BaSO4)': {'n': 1.64, 'uv_abs': 0.0, 'density': 4.5, 'oa': 12.0},
    '氧化铝 (Al2O3)': {'n': 1.76, 'uv_abs': 0.0, 'density': 3.9, 'oa': 20.0},
    '氧化锆 (ZrO2)': {'n': 2.15, 'uv_abs': 0.0, 'density': 5.7, 'oa': 15.0},
    '氧化镧 (La2O3)': {'n': 1.95, 'uv_abs': 0.0, 'density': 6.5, 'oa': 18.0},
    '金红石钛白粉 (TiO2)': {'n': 2.70, 'uv_abs': 0.85, 'density': 4.1, 'oa': 18.0},
    '高岭土 (Kaolin)': {'n': 1.56, 'uv_abs': 0.0, 'density': 2.6, 'oa': 40.0}, 
    '碳酸钙/重钙 (CaCO3)': {'n': 1.59, 'uv_abs': 0.0, 'density': 2.7, 'oa': 15.0},
    '二氧化硅 (SiO2)': {'n': 1.46, 'uv_abs': 0.0, 'density': 2.2, 'oa': 25.0}
}

# ==========================================
# 2. 核心物理测算引擎
# ==========================================
def predict_coating_performance_from_mass(resin_n, resin_mass, resin_solid, resin_density, active_fillers, thickness):
    v_resin = (resin_mass * (resin_solid / 100.0)) / resin_density
    
    total_filler_volume = 0.0
    total_oil_volume_needed = 0.0 
    
    for f in active_fillers:
        rho_i = MATERIAL_DB[f['mat']]['density']
        base_oa_i = MATERIAL_DB[f['mat']]['oa']
        mass_i = f['mass']
        size_i = f['size']
        
        v_i = mass_i / rho_i 
        f['true_volume'] = v_i
        total_filler_volume += v_i
        
        dynamic_oa_i = base_oa_i * (0.8 + (0.2 / max(size_i, 0.05)))
        f['dynamic_oa'] = dynamic_oa_i
        
        oil_vol_i = (mass_i * (dynamic_oa_i / 100.0)) / 0.935
        total_oil_volume_needed += oil_vol_i

    if total_filler_volume == 0:
        base_E = 0.85 + (thickness/500.0)*0.08
        return 5.0, min(0.99, base_E), 0.0, 45.0, 0.0, 0.0, []

    calculated_pvc = (total_filler_volume / (v_resin + total_filler_volume)) * 100.0
    dynamic_cpvc = (total_filler_volume / (total_filler_volume + total_oil_volume_needed)) * 100.0

    effective_porosity = 0.0
    if calculated_pvc > dynamic_cpvc:
        effective_porosity = min(55.0, (calculated_pvc - dynamic_cpvc) * 0.7)  

    spacer_volume = sum(f['true_volume'] for f in active_fillers if f['size'] >= 3.0)
    spacer_ratio = spacer_volume / max(0.0001, total_filler_volume)

    # --- 多组分 Mie 散射物理池化 ---
    THICKNESS_SCALE = 90.0       

    n_host = resin_n * (1.0 - effective_porosity/100.0) + 1.0 * (effective_porosity/100.0)
    tir = 1.0 + (effective_porosity/100.0 * 2.5) 
    
    pooled_scattering = 0.0
    pooled_uv_abs = 0.0
    diagnostics = []

    for f in active_fillers:
        vol_ratio = f['true_volume'] / total_filler_volume 
        n_i = MATERIAL_DB[f['mat']]['n']
        uv_abs_i = MATERIAL_DB[f['mat']]['uv_abs']
        size_i = f['size']
        
        # 🌟 计算每种填料在当前背景下的理论最佳粒径
        delta_n_i = max(0.01, n_i - n_host)
        opt_size_i = 0.5 / (2 * delta_n_i)
        
        if size_i <= opt_size_i:
            size_efficiency_i = max(0.1, size_i / opt_size_i)
        else:
            size_efficiency_i = (opt_size_i / size_i)**0.8 + 0.2
            
        base_physical_scattering = 0.4
        scatter_i = (base_physical_scattering + delta_n_i * 15.0 * size_efficiency_i) * tir
        
        pooled_scattering += scatter_i * vol_ratio
        pooled_uv_abs += uv_abs_i * vol_ratio
        
        # 🌟 在输出诊断表中新增“理论最佳粒径”列
        diagnostics.append({
            '材料': f['mat'].split(' ')[0],
            '质量(份)': f['mass'],
            '实际粒径': f"{size_i} μm",
            '理论最佳粒径': f"{opt_size_i:.2f} μm",
            '动态吸油率': f"{f['dynamic_oa']:.1f}",
            '综合散射贡献': f"{scatter_i*vol_ratio:.2f}",
            '粉体包占比': f"{vol_ratio*100:.1f}%"
        })

    # --- K-M 渐进方程推演 (剥离了白度惩罚) ---
    optical_thickness = pooled_scattering * (calculated_pvc / 100.0) * (thickness / THICKNESS_SCALE)
    
    base_fresnel = 5.0
    solar_uv_fraction = 5.0
    max_r = 100.0 - (solar_uv_fraction * pooled_uv_abs) 
    
    R_pred = base_fresnel + (max_r - base_fresnel) * (1 - np.exp(-optical_thickness))
    
    dynamic_crowding_threshold = 40.0 + (effective_porosity * 1.5) + (spacer_ratio * 25.0)
    crowding_penalty = max(0, (calculated_pvc - dynamic_crowding_threshold) * 0.3)
    
    R_pred = max(5.0, R_pred - crowding_penalty)
    
    E_pred = 0.85 + (thickness/500.0)*0.08 + (calculated_pvc/100.0)*0.03 + (effective_porosity/100.0)*0.02
    E_pred = min(0.99, E_pred)
    
    return R_pred, E_pred, calculated_pvc, dynamic_cpvc, effective_porosity, spacer_ratio, diagnostics

# ==========================================
# 3. Streamlit 网页交互界面
# ==========================================
st.title("🔬 高反隔热涂层 | 工业级正向测算平台")
st.markdown("系统内置**「动态吸油量耦合」**与**「多分散粒径物理底色」**。自动推算干膜真实微孔与理论最佳粒径。")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.header("📝 第一步：输入车间配方表")
    with st.container(border=True):
        st.subheader("1. 连续相 (成膜物质)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            resin_mass = st.number_input("乳液添加量 (质量份)", min_value=0.0, value=22.0, step=1.0)
        with c2:
            resin_solid = st.number_input("乳液固含 (%)", min_value=10.0, max_value=100.0, value=48.0, step=1.0)
        with c3:
            resin_density = st.number_input("干树脂密度", min_value=0.8, max_value=2.0, value=1.05, step=0.01)
        with c4:
            resin_n = st.number_input("折射率 (n)", min_value=1.40, max_value=1.60, value=1.50, step=0.01)

    with st.container(border=True):
        st.subheader("2. 离散相 (填料加料单 - 最多5种)")
        tabs = st.tabs(["填料 1", "填料 2", "填料 3", "填料 4", "填料 5"])
        active_fillers = []
        for i, tab in enumerate(tabs):
            with tab:
                default_mat_idx = 7 if i == 0 else 0 
                mat = st.selectbox("选择填料材质", list(MATERIAL_DB.keys()), index=default_mat_idx, key=f"mat_{i}")
                if mat != '无 (不添加)':
                    col_a, col_b = st.columns(2)
                    with col_a:
                        mass = st.number_input("添加量 (质量份)", 0.0, 500.0, 30.0 if i==0 else 10.0, 1.0, key=f"mass_{i}")
                    with col_b:
                        size = st.number_input("主体粒径 (μm)", min_value=0.1, max_value=50.0, value=12.3 if i==0 else 1.0, step=0.1, key=f"size_{i}")
                    if mass > 0:
                        active_fillers.append({'mat': mat, 'size': size, 'mass': mass})

    with st.container(border=True):
        st.subheader("3. 施工控制")
        thickness = st.number_input("目标干膜厚度 (μm)", min_value=10, max_value=1000, value=200, step=10)

with col2:
    st.header("📊 第二步：测算结果与体系诊断")
    
    R_pred, E_pred, calculated_pvc, dynamic_cpvc, effective_porosity, spacer_ratio, diagnostics = predict_coating_performance_from_mass(
        resin_n, resin_mass, resin_solid, resin_density, active_fillers, thickness
    )
    
    st.markdown("### 🎯 终端光学与热力学指标")
    m1, m2 = st.columns(2)
    with m1:
        st.metric(label="预估太阳光反射比 (R_sol)", value=f"{R_pred:.2f} %")
    with m2:
        st.metric(label="预估大气窗口发射率 (ε)", value=f"{E_pred:.3f}")
        
    st.divider()
    
    st.markdown("### 🧪 配方结构与微孔自动诊断")
    
    c_a, c_b, c_c = st.columns(3)
    with c_a:
        st.metric(label="系统临界点 (CPVC)", value=f"{dynamic_cpvc:.1f}%")
    with c_b:
        st.metric(label="实际浓度 (PVC)", value=f"{calculated_pvc:.1f}%", 
                  delta="超临界" if calculated_pvc > dynamic_cpvc else "致密", delta_color="inverse" if calculated_pvc > dynamic_cpvc else "normal")
    with c_c:
        st.metric(label="自动推演孔隙率", value=f"{effective_porosity:.1f}%",
                  delta="产生干遮盖微孔" if effective_porosity > 0 else "无微孔")
    
    if calculated_pvc > dynamic_cpvc:
        st.success(f"**💡 干遮盖增强触发**：实际 PVC 超出临界点，自动产生 **{effective_porosity:.1f}%** 的结构微孔，大幅放大散射威力。")
    else:
        st.info("配方处于致密状态，树脂完全包裹粉体，不产生额外结构孔隙。")
        
    if spacer_ratio > 0.05:
        st.success(f"**🛠️ 骨架支撑激活**：配方含 {spacer_ratio*100:.1f}% 的大粒径骨架(≥3μm)，有效缓解了细粉拥挤造成的光学相消。")

    if diagnostics:
        df_diag = pd.DataFrame(diagnostics)
        st.markdown("**级配特征与米氏散射(Mie)匹配表：**")
        st.dataframe(df_diag, use_container_width=True, hide_index=True)
        st.caption("🔍 **提示**：当“实际粒径”越接近“理论最佳粒径”时，该填料对反射率的贡献效率越高。重钙/高岭土等大粒径填料偏离最佳粒径，主要提供骨架支撑和物理底色。")
