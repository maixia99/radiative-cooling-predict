import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 🌟 修复 Scipy 版本兼容问题：使用 simpson 替代已废弃的 simps
from scipy.integrate import simpson as simps

# ==========================================
# 页面全局配置
# ==========================================
st.set_page_config(page_title="高反隔热涂层 | 终极物理测算平台", layout="wide")

# ==========================================
# 1. 材料数据库 (引入 k0 紫外消光系数)
# ==========================================
MATERIAL_DB = {
    '无 (不添加)': {'n': 1.0, 'k0': 0.0, 'density': 1.0, 'oa': 0.0},
    '硫酸钡 (BaSO4)': {'n': 1.64, 'k0': 0.001, 'density': 4.5, 'oa': 12.0},
    '氧化铝 (Al2O3)': {'n': 1.76, 'k0': 0.001, 'density': 3.9, 'oa': 20.0},
    '氧化锆 (ZrO2)': {'n': 2.15, 'k0': 0.001, 'density': 5.7, 'oa': 15.0},
    '氧化镧 (La2O3)': {'n': 1.95, 'k0': 0.001, 'density': 6.5, 'oa': 18.0},
    '金红石钛白粉 (TiO2)': {'n': 2.70, 'k0': 0.08, 'density': 4.1, 'oa': 18.0}, 
    '高岭土 (Kaolin)': {'n': 1.56, 'k0': 0.005, 'density': 2.6, 'oa': 40.0}, 
    '碳酸钙/重钙 (CaCO3)': {'n': 1.59, 'k0': 0.002, 'density': 2.7, 'oa': 15.0},
    '二氧化硅 (SiO2)': {'n': 1.46, 'k0': 0.001, 'density': 2.2, 'oa': 25.0}
}

# ==========================================
# 2. 底层物理算子库 (RTE + S(q) + Saunderson)
# ==========================================
@st.cache_data
def get_solar_spectrum():
    wl = np.linspace(0.3, 2.5, 150) # 波长节点
    T = 5800 
    h, c, k = 6.626e-34, 3e8, 1.38e-23
    wl_m = wl * 1e-6
    intensity = (2 * h * c**2 / wl_m**5) / (np.exp(h * c / (wl_m * k * T)) - 1)
    absorption = np.ones_like(wl)
    for dip in [0.94, 1.14, 1.38, 1.88]:
        absorption *= (1 - 0.5 * np.exp(-((wl - dip) / 0.05)**2))
    return wl, intensity * absorption

def percus_yevick_Sq(q, d, phi):
    phi_eff = min(phi, 0.45) # 极限稳定截断
    if phi_eff <= 0.001: return np.ones_like(q)
    x = np.where(q * d == 0, 1e-8, q * d) 
    lambda1 = ((1 + 2 * phi_eff) ** 2) / ((1 - phi_eff) ** 4)
    lambda2 = -(1 + 0.5 * phi_eff) ** 2 / ((1 - phi_eff) ** 4)
    sin_x, cos_x = np.sin(x), np.cos(x)
    term1 = lambda1 * (sin_x - x * cos_x) / (x**3)
    term2 = 6 * phi_eff * lambda2 * (2 * x * sin_x - (x**2 - 2) * cos_x - 2) / (x**4)
    term3 = 0.5 * phi_eff * lambda1 * ( (4*x**3 - 24*x)*sin_x - (x**4 - 12*x**2 + 24)*cos_x + 24 ) / (x**6)
    cx = 24 * phi_eff * (term1 + term2 + term3)
    Sq = 1.0 / (1.0 - cx)
    return np.maximum(Sq, 1e-4)

# ==========================================
# 3. 核心大一统引擎 (前台配方单 -> 宏观光谱与热学)
# ==========================================
def calculate_coating_performance(resin_n, resin_mass, resin_solid, resin_density, active_fillers, thickness):
    # --- A. 质量转体积与临界点测算 ---
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
        total_oil_volume_needed += (mass_i * (dynamic_oa_i / 100.0)) / 0.935

    if total_filler_volume == 0:
        base_E = 0.85 + (thickness/500.0)*0.08
        return 5.0, min(0.99, base_E), 0.0, 45.0, 0.0, [], None, None

    calculated_pvc = (total_filler_volume / (v_resin + total_filler_volume)) * 100.0
    dynamic_cpvc = (total_filler_volume / (total_filler_volume + total_oil_volume_needed)) * 100.0

    effective_porosity = 0.0
    if calculated_pvc > dynamic_cpvc:
        effective_porosity = min(55.0, (calculated_pvc - dynamic_cpvc) * 0.7)  

    # --- B. 准备代入 RTE 的物理参数 ---
    v_total_solid = v_resin + total_filler_volume
    v_air = v_total_solid * (effective_porosity / 100.0) / (1 - effective_porosity / 100.0) if effective_porosity < 100 else 0
    v_total_film = v_total_solid + v_air
    
    n_host_eff = resin_n * (1.0 - effective_porosity/100.0) + 1.0 * (effective_porosity/100.0)
    
    fillers_for_rte = []
    diagnostics = []
    for f in active_fillers:
        mat_key = f['mat']
        phi_i = f['true_volume'] / v_total_film 
        fillers_for_rte.append({
            'd': f['size'], 'phi': phi_i, 
            'n_p': MATERIAL_DB[mat_key]['n'], 'k0': MATERIAL_DB[mat_key]['k0']
        })
        diagnostics.append({
            '材料': mat_key.split(' ')[0],
            '质量': f['mass'],
            '粒径': f"{f['size']} μm",
            '吸油率': f"{f['dynamic_oa']:.1f}",
            '膜内真体积分数': f"{phi_i*100:.1f}%"
        })

    # --- C. 执行顶级的 RTE 物理求解器 ---
    wl_array, I_sun = get_solar_spectrum()
    R_spectrum = []
    
    theta = np.linspace(0, np.pi, 180) 
    solid_angle_element = 2 * np.pi * np.sin(theta)
    
    phi_total_rte = sum(f['phi'] for f in fillers_for_rte)
    d_eff_rte = sum(f['d'] * f['phi'] for f in fillers_for_rte) / max(1e-5, phi_total_rte)
    d_large = max(f['d'] for f in fillers_for_rte)
    
    # 动态 Saunderson 界面 k1
    k1 = ((n_host_eff - 1.0) / (n_host_eff + 1.0))**2
    theta_c = np.arcsin(1.0 / n_host_eff)
    mask_tir = theta > theta_c

    for wl in wl_array:
        Sigma_s_total, Sigma_a_total, g_weighted_sum = 0.0, 0.0, 0.0
        global_p_mod_sum = np.zeros_like(theta)
        
        for f in fillers_for_rte:
            d, phi, n_p, k0 = f['d'], f['phi'], f['n_p'], f['k0']
            if phi <= 0: continue
            
            r = d / 2.0
            
            # 🌟 核心量纲与光学截面修复区 🌟
            delta_n_ratio = abs(n_p - n_host_eff) / n_host_eff
            x_param = 2 * np.pi * r * n_host_eff / wl
            x_eff = x_param * delta_n_ratio
            
            # 1. 经验 Mie 共振饱和模型 (防止大颗粒散射截面无脑发散)
            Q_sca = 3.0 * (x_eff**2) / (1.0 + x_eff**2)
            
            # 2. 超级高斯带隙衰减 (四次方)，彻底斩断紫外吸收向可见光的致命“拖尾”
            k_lambda = k0 * np.exp(- (max(0.0, wl - 0.38) / 0.05)**4)
            
            # 3. 连续介质物理学的严密体积截面换算
            Sigma_s_base = (3.0 * phi * Q_sca) / (4.0 * r)
            Sigma_a_i = (4.0 * np.pi * k_lambda / wl) * phi
            
            # --- S(q) 多体解耦与方向重塑 ---
            q = (4 * np.pi * n_host_eff / wl) * np.sin(theta / 2.0)
            Sq_eff = percus_yevick_Sq(q, d_eff_rte, phi_total_rte) * np.exp(- (q * d_large)**2 * 0.1)
            
            x_param_hg = np.pi * d / wl
            g0 = min(0.95, max(0.1, 1.0 - 2.0/x_param_hg)) if x_param_hg > 2 else 0.1
            p_theta = (1 - g0**2) / (4 * np.pi * (1 + g0**2 - 2*g0*np.cos(theta))**1.5)
            
            p_mod_raw = p_theta * Sq_eff
            normalization = simps(p_mod_raw * solid_angle_element, theta)
            p_mod = p_mod_raw / max(1e-10, normalization)
            
            g_eff = simps(p_mod * np.cos(theta) * solid_angle_element, theta)
            
            Sigma_s_total += Sigma_s_base
            Sigma_a_total += Sigma_a_i
            g_weighted_sum += g_eff * Sigma_s_base
            global_p_mod_sum += p_mod * Sigma_s_base

        if Sigma_s_total > 0:
            g_total = g_weighted_sum / Sigma_s_total
            global_p_mod = global_p_mod_sum / Sigma_s_total
        else:
            g_total, global_p_mod = 0.0, np.ones_like(theta)/(4*np.pi)
            
        # 动态 Saunderson k2 (捕捉全内反射)
        integral_hemi = simps(global_p_mod * np.sin(theta), theta)
        integral_tir = simps(global_p_mod[mask_tir] * np.sin(theta[mask_tir]), theta[mask_tir])
        k2 = integral_tir / integral_hemi if integral_hemi > 0 else 0.6
        k2 = min(max(k2, 0.1), 0.95)
        
        # RTE 两流降阶 (Kubelka-Munk 形式)
        S_flux = Sigma_s_total * (1 - g_total) 
        K_flux = 2 * Sigma_a_total 
        
        if K_flux == 0:
            R_internal = (S_flux * thickness) / (1 + S_flux * thickness)
        else:
            a = 1 + K_flux / max(1e-10, S_flux)
            b = np.sqrt(max(0, a**2 - 1))
            tanh_arg = min(b * S_flux * thickness, 100.0)
            R_internal = 1 / (a + b * (1 / np.tanh(tanh_arg))) if tanh_arg > 1e-4 else 1.0
            
        # Saunderson 边界耦合折损
        R_measured = k1 + ((1 - k1) * (1 - k2) * R_internal) / max(1e-10, 1 - k2 * R_internal)
        R_spectrum.append(R_measured * 100.0)
        
    R_spectrum = np.array(R_spectrum)
    
    # 积分求算太阳光反射比 R_solar
    R_solar = simps(R_spectrum * I_sun, wl_array) / simps(I_sun, wl_array)
    
    # 热发射率预估 
    E_pred = 0.85 + (thickness/500.0)*0.08 + (calculated_pvc/100.0)*0.03 + (effective_porosity/100.0)*0.02
    E_pred = min(0.99, max(0.0, E_pred))
    
    return R_solar, E_pred, calculated_pvc, dynamic_cpvc, effective_porosity, diagnostics, wl_array, R_spectrum

# ==========================================
# 4. Streamlit 网页交互界面
# ==========================================
st.title("🔬 高反隔热涂层 | 第一性原理物理测算引擎")
st.markdown("内核已搭载 **多波段RTE + Percus-Yevick多体解耦 + 动态Saunderson界面修正**。完全消除数值虚高，还原真实物理世界。")

col1, col2 = st.columns([1.1, 1.3])

with col1:
    st.header("📝 第一步：输入车间配方表")
    with st.container(border=True):
        st.subheader("1. 连续相 (成膜物质)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            resin_mass = st.number_input("乳液(份)", min_value=0.0, value=20.0, step=1.0)
        with c2:
            resin_solid = st.number_input("固含(%)", min_value=10.0, max_value=100.0, value=48.0, step=1.0)
        with c3:
            resin_density = st.number_input("干密度", min_value=0.8, max_value=2.0, value=1.05, step=0.01)
        with c4:
            resin_n = st.number_input("折射率", min_value=1.40, max_value=1.60, value=1.50, step=0.01)

    with st.container(border=True):
        st.subheader("2. 离散相 (填料加料单)")
        tabs = st.tabs(["填料 1", "填料 2", "填料 3", "填料 4", "填料 5"])
        active_fillers = []
        for i, tab in enumerate(tabs):
            with tab:
                default_mat_idx = 5 if i == 0 else 0  # 默认测试高浓度钛白粉
                mat = st.selectbox("选择填料材质", list(MATERIAL_DB.keys()), index=default_mat_idx, key=f"mat_{i}")
                if mat != '无 (不添加)':
                    col_a, col_b = st.columns(2)
                    with col_a:
                        mass = st.number_input("添加量 (质量份)", 0.0, 500.0, 40.0 if i==0 else 10.0, 1.0, key=f"mass_{i}")
                    with col_b:
                        size = st.number_input("主体粒径 (μm)", min_value=0.01, max_value=50.0, value=0.3 if i==0 else 1.0, step=0.05, key=f"size_{i}")
                    if mass > 0:
                        active_fillers.append({'mat': mat, 'size': size, 'mass': mass})

    with st.container(border=True):
        st.subheader("3. 施工控制")
        thickness = st.number_input("目标干膜厚度 (μm)", min_value=10, max_value=1000, value=200, step=10)

with col2:
    st.header("📊 第二步：多维物理诊断与光谱")
    
    with st.spinner("🚀 正在执行高维多体辐射传输积分求解..."):
        R_sol, E_pred, pvc, cpvc, porosity, diagnostics, wl_array, R_spectrum = calculate_coating_performance(
            resin_n, resin_mass, resin_solid, resin_density, active_fillers, thickness
        )
    
    st.markdown("### 🎯 宏观热学指标")
    m1, m2 = st.columns(2)
    with m1:
        st.metric(label="预估太阳光综合反射比 (R_solar)", value=f"{R_sol:.2f} %")
    with m2:
        st.metric(label="预估大气窗口发射率 (ε)", value=f"{E_pred:.3f}")
        
    st.divider()
    
    st.markdown("### 🌈 全波段物理反射光谱 $R(\lambda)$")
    if wl_array is not None:
        fig, ax = plt.subplots(figsize=(8, 3.5), dpi=120)
        ax.plot(wl_array, R_spectrum, color='crimson', linewidth=2, label="Predicted Reflectance")
        _, I_sun = get_solar_spectrum()
        ax.fill_between(wl_array, 0, I_sun/np.max(I_sun)*100, color='gold', alpha=0.15, label='Solar Energy Distribution')
        
        ax.set_xlim(0.3, 2.5)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Wavelength (μm)")
        ax.set_ylabel("Reflectance (%)")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc="lower right", fontsize=8)
        st.pyplot(fig)
    
    st.markdown("### 🧪 配方结构与网络连通性")
    c_a, c_b, c_c = st.columns(3)
    with c_a:
        st.metric(label="系统临界点 (CPVC)", value=f"{cpvc:.1f}%")
    with c_b:
        st.metric(label="实际浓度 (PVC)", value=f"{pvc:.1f}%", 
                  delta="触发干遮盖微孔" if pvc > cpvc else "体系致密", delta_color="inverse" if pvc > cpvc else "normal")
    with c_c:
        st.metric(label="有效结构孔隙率", value=f"{porosity:.1f}%")

    if diagnostics:
        df_diag = pd.DataFrame(diagnostics)
        st.markdown("**干膜内部体积分数（决定相干相消与骨架效应的核心参数）：**")
        st.dataframe(df_diag, use_container_width=True, hide_index=True)
