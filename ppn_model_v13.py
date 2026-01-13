import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import brentq

# --- 1. 頁面基礎設定 ---
st.set_page_config(page_title="2026 台灣 FIA 旗艦定價模型 (V14.0)", layout="wide")

st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Microsoft JhengHei', 'Segoe UI', Arial, sans-serif;
    }
    h1 { font-size: 26px; color: #1E1E1E; border-bottom: 2px solid #A9A9A9; padding-bottom: 10px; }
    h2 { font-size: 20px; color: #004085; margin-top: 30px; margin-bottom: 15px; font-weight: 700; border-left: 5px solid #0056b3; padding-left: 10px; }
    .stApp { background-color: #F8F9FA; }
    div[data-testid="stMetricValue"] { font-size: 22px; font-weight: 700; color: #2E4053; }
    
    .math-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #b3e5fc;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 側邊欄：全域參數設定 ---
st.sidebar.title("參數控制台 (Global)")

page = st.sidebar.radio("選擇分析頁面", 
    ["1. 方案一：參與率型 (Option 1)", 
     "2. 方案二：FIA主流型 (Option 2)", 
     "3. 兩案比較與利潤分析 (Comparison)"])

st.sidebar.markdown("---")

# A. 產品結構
st.sidebar.subheader("1. 產品結構")
T = st.sidebar.number_input("保單/策略年期 (Tenor)", value=3.0, step=1.0, help="設定預算產生的總年期")
sales_load = st.sidebar.number_input("銷售通路佣金 (Sales Load) %", value=2.0, step=0.5, help="一次性從總預算中扣除") / 100

# B. 市場環境
st.sidebar.subheader("2. 市場環境")
r_rf = st.sidebar.number_input("無風險利率 (Risk-Free) %", value=4.20, step=0.01) / 100
div_q = st.sidebar.number_input("標的股利率 (Dividend Yield) %", value=1.50, step=0.01, help="S&P 500 平均約 1.5%") / 100
sigma = st.sidebar.slider("ATM 波動率 (VIX) %", 10.0, 30.0, 16.0, step=0.5) / 100
vol_skew = st.sidebar.slider("波動率偏度 (Vol Skew) %", -5.0, 0.0, -2.0, step=0.5, help="價外 Call 的隱含波動率通常較低。賣出 Cap 時使用 (Sigma + Skew) 定價。") / 100

# C. 資金與成本
st.sidebar.subheader("3. 資金與成本")
bond_yield = st.sidebar.number_input("債券收益率 (Funding Yield) %", value=5.20, step=0.10) / 100
issuer_spread = st.sidebar.number_input("公司目標利差 (Issuer Spread) %", value=1.50, step=0.10) / 100
opt_spread_cost = st.sidebar.number_input("避險交易價差 (Hedging Spread) %", value=0.80, step=0.10) / 100

# --- 3. 核心函數 ---
def bs_price(S, K, T, r, q, sigma, option_type='call'):
    """Black-Scholes 定價模型"""
    try:
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    except:
        return 0.0

S0 = 100 

# 預算計算邏輯 (Budget = Net Spread * T - Sales Load)
net_annual_spread = bond_yield - issuer_spread
option_budget_pct = (net_annual_spread * T) - sales_load
option_budget_amt = S0 * option_budget_pct

# 預算檢查
if option_budget_amt <= 0:
    st.error(f"⚠️ 嚴重警告：預算不足！(虧損 {option_budget_pct:.2%})")
    st.markdown(f"""
    **原因分析：**
    * 總利差收入: {net_annual_spread:.2%} * {T}年 = {net_annual_spread*T:.2%}
    * 銷售費用: {sales_load:.2%}
    * **淨預算:** {option_budget_pct:.2%} (負值無法購買期權)
    """)
    st.stop()

# ==========================================
# PAGE 1: 方案一 (Option 1)
# ==========================================
if page == "1. 方案一：參與率型 (Option 1)":
    st.title("方案一：參與率型 (Fixed Income + Buy Call)")
    st.markdown("### 架構：全額預算買入 ATM Call")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("資金池結構 (Total Tenor)")
        st.metric("總債券收益 (Yield * T)", f"{bond_yield * T:.2%}")
        st.metric("總公司利潤 (Spread * T)", f"-{issuer_spread * T:.2%}")
        st.metric("一次性銷售費用 (Sales Load)", f"-{sales_load:.2%}")
        st.metric("👉 總期權預算", f"${option_budget_amt:.2f} ({option_budget_pct:.2%})")
        
    with col2:
        st.subheader("規格試算")
        call_atm_raw = bs_price(S0, S0, T, r_rf, div_q, sigma, 'call')
        call_atm_ask = call_atm_raw * (1 + opt_spread_cost)
        
        pr_opt1 = option_budget_amt / call_atm_ask
        
        st.metric("買入 Call 成本 (含Spread)", f"${call_atm_ask:.2f}")
        st.metric("✨ 可提供參與率 (PR)", f"{pr_opt1:.2%}", delta="無上限")
        
    # 圖表
    st.markdown("---")
    market_moves = np.linspace(-0.15, 0.30, 400)
    y_opt1 = [max(0, m * pr_opt1) if m > 0 else 0 for m in market_moves]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=market_moves*100, y=market_moves*100, name="S&P 500", line=dict(color='gray', dash='dot')))
    fig.add_trace(go.Scatter(x=market_moves*100, y=np.array(y_opt1)*100, name=f"方案一 (PR={pr_opt1:.0%})", line=dict(color='#2E86C1', width=4)))
    fig.update_layout(title=f"方案一損益模擬 ({T}年期累積)", xaxis_title="指數漲幅 (%)", yaxis_title="客戶收益 (%)", template="plotly_white", height=450)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# PAGE 2: 方案二 (Option 2 - FIA)
# ==========================================
elif page == "2. 方案二：FIA主流型 (Option 2)":
    st.title("方案二：FIA 主流型 (Bull Call Spread)")
    
    with st.expander("📚 定價邏輯揭密 (Pricing Equation) - 點擊展開", expanded=True):
        st.markdown("""
        <div class="math-box">
        <b>為什麼 Cap 與參與率 (PR) 只能二選一？</b><br>
        因為我們的預算 (Budget) 是固定的。這是數學上的零和遊戲：
        </div>
        """, unsafe_allow_html=True)
        st.latex(r'''Budget = PR \times ( \underbrace{Call_{Buy}}_{買入成本} - \underbrace{Call_{Sell}}_{賣出Cap收入} )''')

    st.markdown("### 🛠️ 設計模式選擇")
    solve_mode = st.radio("請選擇設計邏輯：", 
                          ["模式 A：固定參與率 (100%) ➜ 算出 Cap", 
                           "模式 B：固定 Cap (自訂) ➜ 算出 參與率 (PR)"])
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    call_atm_raw = bs_price(S0, S0, T, r_rf, div_q, sigma, 'call')
    call_atm_ask = call_atm_raw * (1 + opt_spread_cost)
    
    final_cap = 0; final_pr = 0; cap_display = ""
    eq_cost_long = call_atm_ask; eq_rev_short = 0
    
    if solve_mode == "模式 A：固定參與率 (100%) ➜ 算出 Cap":
        with col1:
            st.subheader("模式 A：鎖定 PR = 100%")
            funding_gap = call_atm_ask - option_budget_amt
            st.metric("預算缺口", f"-${funding_gap:.2f}", help="需賣出 Cap 來填補")

        with col2:
            st.subheader("試算結果")
            if funding_gap <= 0:
                final_cap = 9.99; cap_display = "無上限"
                final_pr = 1.0
            else:
                target_short_val = funding_gap / (1 - opt_spread_cost)
                try:
                    vol_adjusted = sigma + vol_skew # 使用 Skew
                    k_cap = brentq(lambda K: bs_price(S0, K, T, r_rf, div_q, vol_adjusted, 'call') - target_short_val, S0, S0*3)
                    final_cap = (k_cap / S0) - 1
                    cap_display = f"{final_cap:.2%}"
                    
                    call_short_raw = bs_price(S0, k_cap, T, r_rf, div_q, vol_adjusted, 'call')
                    eq_rev_short = call_short_raw * (1 - opt_spread_cost)
                except:
                    final_cap = 0; cap_display = "無法計算 (預算過低)"
                final_pr = 1.0
            
            st.metric("參與率 (PR)", "100%")
            st.metric("✨ 推算獲利上限 (Cap)", cap_display, delta="考慮 Skew 後")

    else: # 模式 B
        with col1:
            st.subheader("模式 B：鎖定 Cap (競品對標)")
            target_cap_input = st.slider("請設定目標 Cap %", 5.0, 30.0, 15.0, step=0.5) / 100
            
        with col2:
            st.subheader("試算結果")
            k_cap_target = S0 * (1 + target_cap_input)
            vol_adjusted = sigma + vol_skew # 使用 Skew
            call_short_val = bs_price(S0, k_cap_target, T, r_rf, div_q, vol_adjusted, 'call')
            eq_rev_short = call_short_val * (1 - opt_spread_cost)
            unit_spread_cost = call_atm_ask - eq_rev_short
            
            final_pr = option_budget_amt / unit_spread_cost
            final_cap = target_cap_input
            cap_display = f"{final_cap:.2%}"
            
            st.metric("設定獲利上限 (Cap)", cap_display)
            delta_color = "normal" if final_pr >= 0.8 else "inverse"
            st.metric("✨ 可提供參與率 (PR)", f"{final_pr:.2%}", delta="考慮 Skew 後", delta_color=delta_color)

    st.markdown("---")
    market_moves = np.linspace(-0.15, 0.30, 400)
    y_opt2 = [min(m * final_pr, final_cap) if m > 0 else 0 for m in market_moves]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=market_moves*100, y=market_moves*100, name="S&P 500", line=dict(color='gray', dash='dot')))
    fig.add_trace(go.Scatter(x=market_moves*100, y=np.array(y_opt2)*100, name=f"方案二 (Cap={cap_display})", line=dict(color='#C0392B', width=4)))
    fig.add_annotation(x=15, y=final_cap*100, text=f"獲利封頂 {cap_display}", showarrow=True, arrowhead=1, ax=0, ay=-40, font=dict(color="#C0392B"))
    fig.update_layout(title=f"方案二損益模擬 ({T}年期累積)", xaxis_title="指數漲幅 (%)", yaxis_title="客戶收益 (%)", template="plotly_white", height=450)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# PAGE 3: 兩案比較與利潤分析 (Final Enhanced)
# ==========================================
elif page == "3. 兩案比較與利潤分析 (Comparison)":
    st.title("兩案比較與利潤結構分析")
    
    # 快速重算 (Backend Recalculation)
    call_atm_raw = bs_price(S0, S0, T, r_rf, div_q, sigma, 'call')
    call_atm_ask = call_atm_raw * (1 + opt_spread_cost)
    pr_opt1 = option_budget_amt / call_atm_ask
    
    # Option 2: Default to PR 100% mode for comparison base
    vol_adjusted = sigma + vol_skew
    gap = call_atm_ask - option_budget_amt
    if gap <= 0: final_cap_o2 = 9.99
    else:
        try:
            k = brentq(lambda K: bs_price(S0, K, T, r_rf, div_q, vol_adjusted, 'call') * (1 - opt_spread_cost) - gap, S0, S0*3)
            final_cap_o2 = (k / S0) - 1
        except: final_cap_o2 = 0
    final_pr_o2 = 1.0

    # --- Part 1: 資金分配概覽 ---
    st.header("1. 資金分配概覽 (Profitability Allocation)")
    val_margin = issuer_spread * T
    val_cost = option_budget_pct * (opt_spread_cost / (1 + opt_spread_cost))
    val_client = option_budget_pct - val_cost
    
    fig_profit = go.Figure()
    fig_profit.add_trace(go.Bar(y=['資金分配'], x=[val_client*100], name='客戶期權價值', orientation='h', marker=dict(color='#3498DB'), text=[f"{val_client:.2%}"], textposition='auto'))
    fig_profit.add_trace(go.Bar(y=['資金分配'], x=[val_cost*100], name='避險交易成本', orientation='h', marker=dict(color='#E74C3C'), text=[f"{val_cost:.2%}"], textposition='auto'))
    fig_profit.add_trace(go.Bar(y=['資金分配'], x=[val_margin*100], name='公司總利潤 (Spread)', orientation='h', marker=dict(color='#2ECC71'), text=[f"{val_margin:.2%}"], textposition='auto'))
    fig_profit.update_layout(barmode='stack', title=f"{T}年期總債券收益分配", xaxis_title="佔本金百分比 (%)", height=180, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_profit, use_container_width=True)
    
    # --- Part 2: ICS 2.0 資本結構 ---
    st.markdown("---")
    st.header("2. ICS 2.0 資本結構拆解")
    risk_margin_ratio = 0.40 
    net_profit_ratio = 0.60 
    val_risk_margin = val_margin * risk_margin_ratio
    val_net_profit = val_margin * net_profit_ratio
    
    fig_ics = go.Figure()
    fig_ics.add_trace(go.Bar(y=['ICS 2.0 結構'], x=[val_client*100], name='客戶權益', orientation='h', marker=dict(color='#AED6F1')))
    fig_ics.add_trace(go.Bar(y=['ICS 2.0 結構'], x=[val_cost*100], name='交易成本', orientation='h', marker=dict(color='#F1948A')))
    fig_ics.add_trace(go.Bar(y=['ICS 2.0 結構'], x=[val_risk_margin*100], name='風險邊際 (RM & CoC)', orientation='h', marker=dict(color='#F39C12'), text=[f"{val_risk_margin:.2%}"], textposition='auto'))
    fig_ics.add_trace(go.Bar(y=['ICS 2.0 結構'], x=[val_net_profit*100], name='股東淨利 (Net Profit)', orientation='h', marker=dict(color='#27AE60'), text=[f"{val_net_profit:.2%}"], textposition='auto'))
    fig_ics.update_layout(barmode='stack', title="Issuer Spread 深度拆解", xaxis_title="佔本金百分比 (%)", height=180, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_ics, use_container_width=True)

    # --- Part 3: 效益整合 (Green Zone Added) ---
    st.markdown("---")
    st.header("3. 效益整合分析：紅藍對決")
    market_moves = np.linspace(0.0, 0.45, 300) # 拉長X軸以顯示交叉
    y_o1 = market_moves * pr_opt1
    y_o2 = [min(m * final_pr_o2, final_cap_o2) for m in market_moves]
    
    fig_comp = go.Figure()
    # 方案一
    fig_comp.add_trace(go.Scatter(x=market_moves*100, y=y_o1*100, name=f'方案一: 參與率型 (PR={pr_opt1:.0%})', line=dict(color='#2E86C1', width=3)))
    # 方案二
    fig_comp.add_trace(go.Scatter(x=market_moves*100, y=np.array(y_o2)*100, name=f'方案二: FIA型 (PR=100% / Cap={final_cap_o2:.1%})', line=dict(color='#C0392B', width=4)))
    
    # [VISUAL UPGRADE] 計算交叉點與優勢區間
    if final_pr_o2 > pr_opt1: # 方案二斜率較陡 (100% vs <100%)
        cross_point = final_cap_o2 / pr_opt1
        
        # 繪製綠色優勢區間 (Rect)
        fig_comp.add_vrect(
            x0=0, x1=cross_point*100,
            fillcolor="rgba(46, 204, 113, 0.15)", # 半透明綠色
            layer="below", line_width=0
        )
        
        # 標註文字
        fig_comp.add_annotation(
            x=cross_point*40, y=final_cap_o2*105, # 文字位置微調
            text="<b>Bull Call Spread 優勢區間</b><br>(累積獲利更快)", 
            showarrow=False, 
            font=dict(color="#1D8348", size=14)
        )
        
        # 標註交叉點
        if cross_point < 0.45:
             fig_comp.add_annotation(x=cross_point*100, y=final_cap_o2*100, text=f"黃金交叉: {cross_point:.1%}", showarrow=True, arrowhead=2, ax=40, ay=-40)

    fig_comp.update_layout(
        title="客戶收益比較：方案一 vs 方案二", 
        xaxis_title="指數漲幅 (%)", yaxis_title="客戶收益 (%)", 
        template="plotly_white", height=500,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)")
    )
    st.plotly_chart(fig_comp, use_container_width=True)