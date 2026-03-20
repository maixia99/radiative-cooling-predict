import streamlit as st
import numpy as np
import pandas as pd

# ==========================================
# 页面全局配置
# ==========================================
st.set_page_config(page_title="高反隔热涂层 | 工业级正向测算平台", layout="wide")

# ==========================================
# 1. 终极材料数据库 (包含真实物理密度与 1.0μm 基准吸油量 OA)
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
def predict_coating_performance_from_mass(resin_n, resin_mass, resin_solid, resin_density, active_fillers, thickness, porosity):
    # ---------------------------------------------------------
    # 第一阶段：质量转体积与动态吸油量 (OA) 计算
    # ---------------------------------------------------------
    v_resin = (resin_mass * (resin_solid / 100.0)) / resin_density
    
    total_filler_volume = 0.0
    total_oil_volume_needed = 0.0 
    
    for f in active_fillers:
        rho_i = MATERIAL_DB[f['mat']]['density']
        base_oa_i = MATERIAL_DB[f['mat']]['oa']
        mass_i = f['mass']
        size_i = f['size']
        
        # 算真实体积
        v_i = mass_i / rho_i 
        f['true_volume'] = v_i
        total_filler_volume += v_i
        
        # 动态吸油量：粒径越小，比表面积越大，吸油量呈反比放大
        dynamic_oa_i = base_oa_i * (0.8 + (0.2 / max(size_i, 0.05)))
        f['dynamic_oa'] = dynamic_oa_i
        
        # 润湿所需的油(树脂)体积 (标准亚麻子油密度按 0.935 计)
        oil_vol_i = (mass_i * (dynamic_oa_i / 100.0)) / 0.935
        total_oil_volume_needed += oil_vol_i

    # 如果完全没有添加粉体，直接返回纯树脂基准值
    if total_filler_volume == 0:
        base_E = 0.85 + (thickness/500.0)*0.08 + (porosity/100.0)*0.02
        return 5.0, min(0.99, base_E), 0.0, 45.0, porosity, []

    # ---------------------------------------------------------
    # 第二阶段：真实 PVC、动态 CPVC 与干遮盖效应耦合
    # ---------------------------------------------------------
    # 实际 PVC
    calculated_pvc = (total_filler_volume / (v_resin + total_filler_volume)) * 100.0
    # 动态 CPVC (临界点由粉体包的综合吸油量决定)
    dynamic_cpvc = (total_filler_volume / (total_filler_volume + total_oil_volume_needed)) * 100.0

    auto_porosity = 0.0
    # 如果实际浓度超越了临界点，树脂包不住粉体，强制产生干遮盖气孔
    if calculated_pvc > dynamic_cpvc:
        auto_porosity = (calculated_pvc - dynamic_cpvc) * 0.6  
        
    # 最终参与光子弹射的有效孔隙率 (配方自带 + 施工引入)
    effective_porosity = min(60.0, porosity + auto_porosity)


    # ---------------------------------------------------------
    # 第三阶段：多组分 Mie 散射物理池化
    # ---------------------------------------------------------
    SCATTERING_MULTIPLIER = 8.5  # 散射放大倍数经验值
    THICKNESS_SCALE = 90.0       # 光学厚度基准常数

    # 包含孔隙的连续相背景有效折射率
    n_host = resin_n * (1.0 - effective_porosity/100.0) + 1.0 * (effective_porosity/100.0)
    # 极高孔隙率带来的全内反射(TIR)增益
    tir = 1.0 + (effective_porosity/100.0 * 1.5) 
    
    pooled_scattering = 0.0
    pooled_uv_abs = 0.0
    diagnostics = []

    for f in active_fillers:
        vol_ratio = f['true_volume'] / total_filler_volume 
        n_i = MATERIAL_DB[f['mat']]['n']
        uv_abs_i = MATERIAL_DB[f['mat']]['uv_abs']
        size_i = f['size']
        
        # 相对折射率差
        delta_n_i = max(0.01, n_i - n_host)
        # 黄金匹配粒径
        opt_size_i = 0.5 / (2 * delta_n_i)
        # 尺寸偏离惩罚 (稍微放宽对体质大填料的惩罚)
        size_efficiency_i = max(0.15, 1.0 - abs(size_i - opt_size_i) / opt_size_i * 0.5)
        
        # 该颗粒的散射驱动力
        scatter_i = (delta_n_i * SCATTERING_MULTIPLIER) * size_efficiency_i * tir
        
        # 累加到全局池
        pooled_scattering += scatter_i * vol_ratio
        pooled_uv_abs += uv_abs_i * vol_ratio
        
        diagnostics.append({
            '材料': f['mat'].split(' ')[0],
            '质量(份)': f['mass'],
            '实际粒径': f"{size_i} μm",
            '动态吸油率': f"{f['dynamic_oa']:.1f}",
            '体积占比': f"{vol_ratio*100:.1f}%",
            '最佳理论粒径': f"{opt_size_i:.2f} μm"
        })

    # ---------------------------------------------------------
    # 第四阶段：Kubelka-Munk 方程推演太阳光反射比与发射率
    # ---------------------------------------------------------
    # 光子碰撞总概率 (光学厚度)
    optical_thickness = pooled_scattering * (calculated_pvc / 100.0) * (thickness / THICKNESS_SCALE)
    
    # 物理起点：表面菲涅尔反射
    base_fresnel = 5.0
    # 物理终点：扣除紫外带隙吸收后的理论天花板
    solar_uv_fraction = 5.0
    max_r = 100.0 - (solar_uv_fraction * pooled_uv_abs) 
    
    # K-M 核心指数逼近方程 (太阳光反射比)
    R_pred = base_fresnel + (max_r - base_fresnel) * (1 - np.exp(-optical_thickness))
    
    # 低于动态 CPVC 时，过于拥挤会导致粒子干涉惩罚
    if calculated_pvc <= dynamic_cpvc:
        penalty = max(0, (calculated_pvc - (dynamic_cpvc * 0.8)) * 0.1)
        R_pred = max(5.0, R_pred - penalty)
    
    # 大气窗口发射率推演 (高分子厚度与填料浓度共同决定)
    E_pred = 0.85 + (thickness/500.0)*0.08 + (calculated_pvc/100.0)*0.03 + (effective_porosity/100.0)*0.02
    E_pred = min(0.99, E_pred)
    
    return R_pred, E_pred, calculated_pvc, dynamic_cpvc, effective_porosity, diagnostics

# ==========================================
# 3. Streamlit 网页交互界面
# ==========================================
st.title("🔬 高反隔热涂层 | 工业级正向测算平台")
st.markdown("系统内置 **「比表面积-动态吸油量(OA)」自动耦合引擎**。您只需输入工厂加料质量单，系统将自动锁定体系真实 CPVC，并依据 K-M 物理方程推演终端光学性能。")

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
            porosity = st.slider("体系外加发泡/孔隙率 (%)", 0.0, 50.0, 5.0, 1.0)

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
    
    # 核心测算执行
    R_pred, E_pred, calculated_pvc, dynamic_cpvc, effective_porosity, diagnostics = predict_coating_performance_from_mass(
        resin_n, resin_mass, resin_solid, resin_density, active_fillers, thickness, porosity
    )
    
    st.markdown("### 🎯 终端光学与热力学指标")
    m1, m2 = st.columns(2)
    with m1:
        st.metric(label="预估太阳光反射比 (R_sol)", value=f"{R_pred:.2f} %")
    with m2:
        st.metric(label="预估大气窗口发射率 (ε)", value=f"{E_pred:.3f}")
        
    st.divider()
    
    st.markdown("### 🧪 动态临界点 (CPVC) 与吸油诊断")
    
    st.info(f"**系统计算得出，当前混合粉体包的专属临界点 (CPVC) 为：{dynamic_cpvc:.1f}%**")
    
    if calculated_pvc > dynamic_cpvc:
        st.error(f"**您的实际颜料体积浓度 (PVC)：{calculated_pvc:.1f}% (⚠️ 已超越动态 CPVC)**")
        st.success(f"**💡 干遮盖效应触发：**由于树脂不足以浸润所有粉体，体系自动产生了显著的干遮盖微孔。当前参与光学测算的**最终有效孔隙率为 {effective_porosity:.1f}%**，极大地放大了全内反射（TIR）威力！")
    else:
        st.success(f"**您的实际颜料体积浓度 (PVC)：{calculated_pvc:.1f}% (体系健康致密)**")
        st.info(f"**最终有效孔隙率**：{effective_porosity:.1f}%")
        
    if diagnostics:
        df_diag = pd.DataFrame(diagnostics)
        st.markdown("**配方级配与动态吸油特征拆解表：**")
        st.dataframe(df_diag, use_container_width=True, hide_index=True)
        st.caption("🔍 **提示**：观察上表中的『动态吸油率』，当您在左侧改变粒径时，该数值会根据比表面积自动发生显著变化，并直接影响上方的系统 CPVC。")
