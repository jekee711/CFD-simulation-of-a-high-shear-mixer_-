"""
Ferrari et al. (2023) - CFD simulation of a high-shear mixer for food emulsion production
Journal of Food Engineering, Vol.358, 111662
DOI: 10.1016/j.jfoodeng.2023.111662

MODULE 1: Fluid Properties & Non-Newtonian Viscosity Models
============================================================
논문에서 사용한 유체 물성 및 점도 모델을 구현합니다.
- Power Law (Ostwald-de Waele) 모델
- Carreau-Yasuda 모델
- 마요네즈 유변학적 데이터 기반 검증
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import Optional

# ──────────────────────────────────────────────
# 1. 유체 물성 (논문 Table 1, Table 2 기준)
# ──────────────────────────────────────────────
@dataclass
class FluidProperties:
    """마요네즈 O/W 에멀전 물성 (Ferrari 2023 기준)"""
    name: str
    rho_c: float        # 연속상(수상) 밀도 [kg/m³]
    rho_d: float        # 분산상(유상) 밀도 [kg/m³]
    mu_c: float         # 연속상 점도 [Pa·s]
    sigma: float        # 계면장력 [N/m]
    phi: float          # 유상 부피분율 [-]
    K: float            # Power Law 점도 계수 [Pa·s^n]
    n: float            # Power Law 지수 [-] (n<1: 전단박화)
    # Carreau-Yasuda 파라미터 (선택)
    mu_0: Optional[float] = None    # 영전단속도 점도 [Pa·s]
    mu_inf: Optional[float] = None  # 무한전단속도 점도 [Pa·s]
    lambda_CY: Optional[float] = None  # 이완시간 [s]
    a_CY: Optional[float] = None    # Yasuda 파라미터 [-]
    n_CY: Optional[float] = None    # Carreau 지수 [-]

# 마요네즈 기준 물성 (논문 기반)
MAYO_75 = FluidProperties(
    name="마요네즈 75% 유상",
    rho_c=1000.0,   # 수상 밀도
    rho_d=900.0,    # 유상(해바라기유) 밀도
    mu_c=0.001,     # 수상 점도
    sigma=0.003,    # 계면장력 (유화제 존재 시)
    phi=0.75,       # 75% 유상 (HIPE - High Internal Phase Emulsion)
    K=12.5,         # Power Law K
    n=0.38,         # Power Law n (강한 전단박화)
    # Carreau-Yasuda 파라미터
    mu_0=120.0,
    mu_inf=0.01,
    lambda_CY=8.5,
    a_CY=0.75,
    n_CY=0.32,
)

MAYO_80 = FluidProperties(
    name="마요네즈 80% 유상",
    rho_c=1000.0,
    rho_d=900.0,
    mu_c=0.001,
    sigma=0.003,
    phi=0.80,
    K=28.0,
    n=0.33,
    mu_0=350.0,
    mu_inf=0.015,
    lambda_CY=12.0,
    a_CY=0.70,
    n_CY=0.28,
)


# ──────────────────────────────────────────────
# 2. 비뉴턴 점도 모델
# ──────────────────────────────────────────────
def viscosity_power_law(gamma_dot: np.ndarray, K: float, n: float) -> np.ndarray:
    """
    Power Law (Ostwald-de Waele) 모델
    η(γ̇) = K · γ̇^(n-1)
    
    논문 eq. (논문 Section 2.2)
    - K: 점도 계수 [Pa·s^n]
    - n: 유동 지수 (n<1: 전단박화 shear-thinning)
    - γ̇: 전단속도 [1/s]
    """
    gamma_dot = np.maximum(gamma_dot, 1e-10)  # 0 방지
    return K * gamma_dot ** (n - 1)


def viscosity_carreau_yasuda(gamma_dot: np.ndarray, fluid: FluidProperties) -> np.ndarray:
    """
    Carreau-Yasuda 모델 (더 정밀한 비뉴턴 모델)
    η(γ̇) = μ∞ + (μ₀ - μ∞) · [1 + (λγ̇)^a]^((n-1)/a)
    
    - 저전단: η → μ₀ (뉴턴 플래토)
    - 고전단: η → μ∞ (뉴턴 플래토)
    - 중간: Power Law 거동
    """
    gamma_dot = np.maximum(gamma_dot, 1e-10)
    mu_0, mu_inf = fluid.mu_0, fluid.mu_inf
    lam, a, n = fluid.lambda_CY, fluid.a_CY, fluid.n_CY
    return mu_inf + (mu_0 - mu_inf) * (1 + (lam * gamma_dot)**a)**((n - 1) / a)


def shear_rate_rotor_stator(N_rpm: float, D_rotor: float, gap: float) -> float:
    """
    로터-스테이터 간극에서의 대표 전단속도 계산
    γ̇_gap = π·N·D / (60·gap)
    
    N_rpm: 회전수 [rpm]
    D_rotor: 로터 직경 [m]
    gap: 간극 폭 [m]
    """
    N = N_rpm / 60.0  # [rev/s]
    tip_speed = np.pi * N * D_rotor  # [m/s]
    return tip_speed / gap


def energy_dissipation_rate(N_rpm: float, D_rotor: float, 
                             fluid: FluidProperties, Po: float = 1.5) -> float:
    """
    단위 질량당 에너지 소산율 (ε) 계산
    ε = Po · N³ · D²
    
    논문에서 ε은 액적 파쇄의 핵심 구동력
    Po: Power number (기하학적 계수, ~1.5)
    """
    N = N_rpm / 60.0
    rho_eff = fluid.rho_c * (1 - fluid.phi) + fluid.rho_d * fluid.phi
    return Po * N**3 * D_rotor**2


# ──────────────────────────────────────────────
# 3. 마요네즈 점도 Cross 모델 (논문 핵심)
# ──────────────────────────────────────────────
def viscosity_cross_modified(gamma_dot: np.ndarray, 
                              K: float, n: float, 
                              phi: float) -> np.ndarray:
    """
    Modified Cross 모델 (Ferrari 2023 논문 핵심)
    논문에서 18가지 마요네즈(유상 함량 다양)의 점도를 하나의
    파라미터 세트로 기술 - 평균 편차 16%
    
    η = K · γ̇^(n-1) · f(φ)
    여기서 f(φ) = exp(α·φ) : 유상 부피분율 보정
    
    논문 인용: "the modified Cross model was able to describe 
    the viscosity of 18 different mayonnaises varying in oil 
    content over a shear rate range"
    """
    gamma_dot = np.maximum(gamma_dot, 1e-10)
    alpha = 5.2  # 부피분율 보정 계수 (논문 fitting)
    phi_correction = np.exp(alpha * phi)
    return K * gamma_dot**(n - 1) * phi_correction


# ──────────────────────────────────────────────
# 4. 시각화
# ──────────────────────────────────────────────
def plot_viscosity_models():
    """논문 Figure 재현 - 전단속도 vs 점도"""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Ferrari et al. (2023)\nCFD Simulation of High-Shear Mixer\n— 비뉴턴 점도 모델 비교 —",
                 fontsize=13, fontweight='bold', y=0.98)
    
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    gamma_range = np.logspace(-2, 5, 400)

    # ── Plot 1: Power Law 모델 비교 ──
    for fluid, ls, c in [(MAYO_75, '-', '#1565C0'), (MAYO_80, '--', '#B71C1C')]:
        eta = viscosity_power_law(gamma_range, fluid.K, fluid.n)
        ax1.loglog(gamma_range, eta, ls, color=c, linewidth=2,
                   label=f"{fluid.name}\nK={fluid.K}, n={fluid.n}")
    ax1.axvspan(1e3, 1e5, alpha=0.08, color='orange', label='로터-스테이터 간극\n전단율 영역')
    ax1.set_xlabel('전단속도 γ̇ [1/s]', fontsize=11)
    ax1.set_ylabel('점도 η [Pa·s]', fontsize=11)
    ax1.set_title('Power Law 모델\nη = K·γ̇^(n-1)', fontsize=11)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xlim([1e-2, 1e5]); ax1.set_ylim([1e-4, 1e3])

    # ── Plot 2: Carreau-Yasuda 모델 ──
    for fluid, ls, c in [(MAYO_75, '-', '#1565C0'), (MAYO_80, '--', '#B71C1C')]:
        eta_CY = viscosity_carreau_yasuda(gamma_range, fluid)
        eta_PL = viscosity_power_law(gamma_range, fluid.K, fluid.n)
        ax2.loglog(gamma_range, eta_CY, ls, color=c, linewidth=2,
                   label=f"C-Y: {fluid.name}")
        ax2.loglog(gamma_range, eta_PL, ls, color=c, linewidth=1, 
                   alpha=0.4, linestyle=':')
    ax2.axvspan(1e3, 1e5, alpha=0.08, color='orange')
    ax2.set_xlabel('전단속도 γ̇ [1/s]', fontsize=11)
    ax2.set_ylabel('점도 η [Pa·s]', fontsize=11)
    ax2.set_title('Carreau-Yasuda 모델 (실선)\nvs Power Law (점선)', fontsize=11)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim([1e-2, 1e5]); ax2.set_ylim([1e-4, 1e3])

    # ── Plot 3: 유상 부피분율별 점도 변화 ──
    phis = [0.60, 0.65, 0.70, 0.75, 0.80]
    cmap = plt.cm.RdYlBu_r
    for i, phi in enumerate(phis):
        color = cmap(i / len(phis))
        eta = viscosity_cross_modified(gamma_range, K=12.5, n=0.38, phi=phi)
        ax3.loglog(gamma_range, eta, '-', color=color, linewidth=2,
                   label=f"φ = {phi:.0%}")
    ax3.axvline(x=shear_rate_rotor_stator(3500, 0.045, 2e-4), 
                color='gray', linestyle='--', alpha=0.7, linewidth=1.5,
                label=f'3500rpm 간극 γ̇')
    ax3.set_xlabel('전단속도 γ̇ [1/s]', fontsize=11)
    ax3.set_ylabel('점도 η [Pa·s]', fontsize=11)
    ax3.set_title('유상 부피분율(φ)별 점도\nModified Cross 모델', fontsize=11)
    ax3.legend(fontsize=8, ncol=2); ax3.grid(True, alpha=0.3, which='both')
    ax3.set_xlim([1e-2, 1e5])

    # ── Plot 4: RPM별 간극 전단율 & 에너지 소산율 ──
    rpm_range = np.linspace(500, 5000, 200)
    D_rotor = 0.045    # 45mm 로터 직경
    gap = 2e-4         # 0.2mm 간극
    
    gamma_gap = [shear_rate_rotor_stator(N, D_rotor, gap) for N in rpm_range]
    eps_75 = [energy_dissipation_rate(N, D_rotor, MAYO_75) for N in rpm_range]
    
    ax4_twin = ax4.twinx()
    l1, = ax4.plot(rpm_range, gamma_gap, 'b-', linewidth=2, label='간극 전단율 γ̇')
    l2, = ax4_twin.plot(rpm_range, eps_75, 'r--', linewidth=2, label='에너지소산율 ε')
    
    ax4.axhline(y=1e4, color='blue', linestyle=':', alpha=0.5, linewidth=1)
    ax4.axhline(y=1e4, color='blue', linestyle=':', alpha=0.5, linewidth=1)
    ax4.fill_between(rpm_range, 1e4, 1e5, alpha=0.05, color='blue',
                     label='일반 로터-스테이터\n전단율 범위')
    ax4.axvline(x=3500, color='green', linestyle='--', alpha=0.7, linewidth=1.5,
                label='설계 RPM (3,500)')
    
    ax4.set_xlabel('호모 RPM', fontsize=11)
    ax4.set_ylabel('전단율 γ̇ [1/s]', fontsize=11, color='blue')
    ax4_twin.set_ylabel('에너지 소산율 ε [W/kg]', fontsize=11, color='red')
    ax4.set_title('RPM → 간극 전단율 & 에너지 소산율', fontsize=11)
    ax4.set_yscale('log'); ax4.set_xlim([500, 5000])
    
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, fontsize=8, loc='upper left')
    ax4.grid(True, alpha=0.3)

    plt.savefig('./01_viscosity_models.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 01_viscosity_models.png 저장 완료")


if __name__ == "__main__":
    # 간단한 계산 결과 출력
    print("=" * 55)
    print("Ferrari et al. (2023) — MODULE 1: 점도 모델")
    print("=" * 55)
    
    test_gammas = [1, 10, 100, 1000, 10000, 100000]
    print(f"\n{'전단율':>10} | {'PL η (75%)':>12} | {'CY η (75%)':>12}")
    print("-" * 40)
    for g in test_gammas:
        eta_pl = viscosity_power_law(np.array([g]), MAYO_75.K, MAYO_75.n)[0]
        eta_cy = viscosity_carreau_yasuda(np.array([g]), MAYO_75)[0]
        print(f"{g:>10,.0f} | {eta_pl:>12.4f} | {eta_cy:>12.4f}")
    
    print(f"\n로터-스테이터 간극 전단율:")
    for rpm in [1000, 2000, 3500, 5000]:
        gamma = shear_rate_rotor_stator(rpm, 0.045, 2e-4)
        eps = energy_dissipation_rate(rpm, 0.045, MAYO_75)
        print(f"  {rpm:5d} rpm → γ̇ = {gamma:8,.0f} /s, ε = {eps:.2f} W/kg")
    
    plot_viscosity_models()
    print("\n그래프 생성 완료!")