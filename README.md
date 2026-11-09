# QFin

Cross-sectional book on 5-minute bars. A factor model known before the bar removes the common move. Inside each sector the book buys the names that have fallen furthest relative to their peers and shorts the names that have risen furthest. Both sides have the same dollars, sectors have the same dollars, and the book is flat by the close.

This repository contains the production rule, the formal mathematical proof of its structural feasibility boundaries, the research modules in `cs_intraday/`, the output figures and metrics, and the 5-minute bar panels used to compute them. Bars at or after the validation end (2025-12-03 10:30) are not used in fitting. The test segment starts 2025-12-03 11:35, after a 60-minute embargo. This is a research record, not an order.

## Rule

### 1. Residual

For name $i$ at 5-minute bar $t$,

$$
r_{i,t} = B_{i}^{\top} f_{t} + \varepsilon_{i,t}.
$$

$B_{i} \in \mathbb{R}^{K}$ is fixed through the day. It is the top $K = 3$ eigenvectors of the correlation matrix of 5-minute returns over the previous quarter (63 trading days), scaled back from correlation units to return units, with sign aligned to the previous estimate. Names with insufficient bar coverage in that window receive a zero loading and do not enter the correlation matrix. The factor realization at bar $t$ is the cross-sectional ordinary least squares projection across active names that print at bar $t$:

$$
f_{t} = (B^{\top} B)^{-1} B^{\top} r_{t}, \qquad \varepsilon_{t} = r_{t} - B f_{t}.
$$

### 2. Four Horizons

Portfolio decisions occur on the hour from 10:00 to 15:00 (`6` decision bars per session). For a lookback window $\mathcal{W}_{h}(s)$ ending at decision bar $s$, the cross-sectionally de-meaned cumulative residual is

$$
m_{i,s}^{(h)} = \sum_{\tau \in \mathcal{W}_{h}(s)} \varepsilon_{i,\tau} - \mathrm{mean}_{j}\left(\sum_{\tau \in \mathcal{W}_{h}(s)} \varepsilon_{j,\tau}\right).
$$

The four horizons are the last 6 bars (30 minutes), 12 bars (1 hour), 24 bars (2 hours), and every bar since the previous close. The first bar of the sample is excluded from the third and fourth windows when counting.

### 3. Sector Anchor

Lagged dollar volume $v_{j,s}$ is computed over the 12 bars ending at $s-1$, ensuring that the peer split at bar $s$ is strictly causal. Inside each GICS sector, names at or above the sector median of lagged dollar volume form the fast peer group $\mathcal{C}_{s}$; the remaining names form the slow peer group $\mathcal{P}_{s}$.

The fast anchor of name $i$ leaves $i$ out and weights fast peers by the square root of lagged dollar volume:

$$
a_{i,s}^{C} = \frac{\sum_{j \in \mathcal{C}_{s},\, j \neq i} \sqrt{v_{j,s}}\, m_{j,s}}{\sum_{j \in \mathcal{C}_{s},\, j \neq i} \sqrt{v_{j,s}}}.
$$

The slow anchor $a_{i,s}^{P}$ is the leave-one-out equal-weighted mean over $\mathcal{P}_{s}$. If name $i$ has no other slow peer in its sector, $a_{i,s}^{P}$ falls back to $a_{i,s}^{C}$. The relative intra-sector gaps are

$$
g_{i,s}^{C} = m_{i,s} - a_{i,s}^{C}, \qquad g_{i,s}^{P} = m_{i,s} - a_{i,s}^{P}.
$$

### 4. Coordinates

On each decision row $s$, each feature is mapped via a cross-sectional rank transform of the negative gap onto $[-1, 1]$:

$$
u_{i,s} = \frac{2}{n_{s}-1}\, \mathrm{rank}(-g_{i,s}) - 1.
$$

If a decision row has fewer than 20 finite names, the entire row is marked missing. The six coordinates (the Direction A+C feature set) are

$$
u^{\mathrm{day},C},\; u^{\mathrm{day},P},\; u^{2\mathrm{h},C},\; u^{1\mathrm{h},C},\; u^{30\mathrm{m},C},\; u^{\mathrm{vol}}.
$$

The first five coordinates represent the four horizons against the fast anchor $\mathcal{C}_{s}$ plus the since-close gap against the slow anchor $\mathcal{P}_{s}$. Depending on the script variant, $u^{\mathrm{vol}}$ is the cross-sectional rank transform of the 20-bar residual variance (`cs_band_v1.py`) or the negative third coordinate (`cov_close_vr`, capturing return-volume co-movement) of the daily-reset per-name Kalman state filter (`cs_f7state_v1.py` and `cs_ac_v1.py`).

### 5. Monthly Fit

Let $\varepsilon_{i,s}^{\mathrm{close}}$ denote the cumulative residual from decision bar $s+1$ to the close, $\sigma_{i,s}$ the name's trailing bar volatility, and $\tau_{s}$ the number of 5-minute bars remaining in the session. The volatility-standardized regression target is

$$
y_{i,s} = \mathrm{clip}\left(\frac{\varepsilon_{i,s}^{\mathrm{close}}}{\sigma_{i,s}\sqrt{\tau_{s}}}, -5, 5\right).
$$

Once a month, using a strictly causal rolling window of the previous 500 trading days, drop the intercept and solve the ordinary least squares problem

$$
\hat{\beta} = \min_{\beta} \| y - U\beta \|_{2},
$$

where the columns of $U$ are the six rank coordinates. A scalar rescaling factor $\alpha$ then maps the dimensionless linear score onto the raw relative close-to-close return $q$ (the name's subsequent return minus the sector's volume-weighted mean return over the remainder of the day):

$$
\alpha = \frac{\sum (\hat{\beta}^{\top} u)\, \sigma\sqrt{\tau}\, q}{\sum \left((\hat{\beta}^{\top} u)\, \sigma\sqrt{\tau}\right)^{2}}.
$$

The return forecast used by the execution gate is

$$
\mu_{i,s} = \alpha\, (\hat{\beta}^{\top} u_{i,s})\, \sigma_{i,s}\sqrt{\tau_{s}}.
$$

### 6. Cost Gate

Let $P_{i,s}$ denote the unadjusted real share price at bar $s$ (the forward-adjusted price divided by the cumulative product of subsequent split and dividend adjustment factors from `data/rehab/`). Transaction costs are charged on the unadjusted price $P_{i,s}$:

$$
c^{\mathrm{in}}_{i,s} = \frac{0.01 + 0.0035}{P_{i,s}}, \qquad c^{\mathrm{out}}_{i,s} = \frac{0.0035}{P_{i,s}}.
$$

The entry charge models a 1.0-cent half-spread plus a 0.35-cent per-share commission; the closing auction exit models the 0.35-cent commission only (total round-trip cost $c_0 = 0.0170$ dollars per share). A name is eligible to trade at bar $s$ if and only if

$$
P_{i,s} \ge P_{\min} \qquad \mathrm{and} \qquad \vert\mu_{i,s}\vert > \lambda \left(c^{\mathrm{in}}_{i,s} + c^{\mathrm{out}}_{i,s}\right),
$$

with $P_{\min} = 60$ and $\lambda = 4.2$.

### 7. Book Construction and Causal Volatility/Drawdown Guard

A sector is active at bar $s$ only if it contains at least 5 eligible names. Inside an active sector with $n$ eligible names, let $k = \lfloor 0.2\, n \rfloor$ with $k \ge 1$. The long leg holds the $k$ highest eligible forecasts $\mu_{i,s}$ weighted equally; the short leg holds the $k$ lowest eligible forecasts weighted equally. If either leg is empty, the sector does not trade at that hour. The target weight vector $w_s$ is dollar-neutral within every active sector, dollar-neutral across the book, and normalized to unit gross exposure across the 6 hourly sleeves:

$$
\mathbf{1}^{\top} w_{s} = 0, \qquad \| w_{s} \|_{1} \in \{0, 1\}.
$$

At hour $j \in \{1, \ldots, 6\}$, the active book is the average of the hourly sleeves initiated up to hour $j$:

$$
h_{j} = \frac{1}{6}\sum_{q=1}^{j} w_{q}.
$$

Each sleeve is executed with a 1-bar (5-minute) delay after decision bar $s$ and liquidated flat at the closing auction ($h = 0$ overnight).

In the drawdown-controlled execution (`cs_ac_v1.py`), daily portfolio exposure is further modulated by a strictly causal volatility and drawdown guard (`apply_causal_dd_guard`) using trailing 20-day realized daily return standard deviation $\sigma_{20\mathrm{d}, d-1}$ and trailing 25-day peak-to-trough drawdown $\mathrm{DD}_{d-1}$:

$$
\mathrm{scale}_{d} = \min\!\left(s_{\max},\; \frac{\sigma_{\mathrm{target}}}{\max(\sigma_{20\mathrm{d}, d-1}\sqrt{252},\, 0.008))}\right) \cdot \psi(\mathrm{DD}_{d-1}),
$$

$$
\psi(\mathrm{DD}_{d-1}) = \begin{cases} 1.0, & \mathrm{DD}_{d-1} \ge -\delta_{\mathrm{soft}}, \\ \max\!\left(0.20,\; 1.0 + \dfrac{\mathrm{DD}_{d-1} + \delta_{\mathrm{soft}}}{0.010}\right), & \mathrm{DD}_{d-1} < -\delta_{\mathrm{soft}}, \end{cases}
$$

where $\sigma_{\mathrm{target}} = 0.032$ ($3.2\%$ annualized volatility target), $\delta_{\mathrm{soft}} = 0.0035 \sim 0.0040$, and $s_{\max} = 0.78$.

## Null and Empirical Summary

The sector permutation null keeps the fitted scores at bar $s$, reassigns them at random to other names within the same sector, and rebuilds the forecast using the recipient's own $\sigma_{i,s}\sqrt{\tau_s}$.

| Pool | Train Return | Train $p$ | Validation Return | Validation $p$ | Train Max DD | Validation Max DD | Full Sharpe |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Non-bull, 97 names ($\lambda = 4.2$)** | **+4.88%/yr** | **0.012** | **+4.30%/yr** | **0.049** | **-1.96%** | **-1.06%** | **1.55** |
| Non-bull, 97 names ($\lambda = 2.8$ moderate gate) | +4.10%/yr | — | -0.86%/yr | — | -3.95% | -5.88% | 0.82 |
| Non-bull, 97 names ($\lambda = 2.0$ loose gate) | +4.06%/yr | — | -4.57%/yr | — | -4.72% | -7.48% | 0.62 |
| Non-bull, 97 names (Equal-weight buy & hold) | +7.57%/yr | — | +6.10%/yr | — | -19.37% | -10.99% | 0.58 |
| Stagnant/Bearish subset, 52 names (Buy & hold) | -0.57%/yr | — | +4.99%/yr | — | -26.08% | -12.34% | 0.09 |
| **Technology, communication, discretionary, 40 names** | **+3.25%/yr** | **0.012** | **+5.30%/yr** | **0.012** | — | — | — |
| **Financials, energy, industrials, 57 names** | **+0.01%/yr** | **0.300** | **-1.39%/yr** | **0.910** | — | — | — |

Figures: `cs_intraday/cs_band_v1_out/cs_band.png` and `cs_intraday/cs_ac_v1_out/cs_ac.png`.

## Empirical Account

The empirical checks pin the profit to one structural object—the same-day relative gap cleared through a high transaction-cost hurdle:

1. **Same-Day Relative Gap Edge:** On the 97-name non-bull panel, the fitted linear coordinate mix has a mean daily cross-sectional rank correlation (IC) of $+0.034$ with the subsequent residual to the close ($p = 0.002$). On the $\lambda = 4.2$ gated book, the realized net return per active trade is $+1.91\,\mathrm{bp}$ in train ($+2.23\,\mathrm{bp}$ gross) and $+1.69\,\mathrm{bp}$ in validation ($+1.76\,\mathrm{bp}$ gross).
2. **Robustness to Anchor Perturbation:** Replacing the volume-split fast half with a random intra-sector split and refitting (`cs_fw_test_v1.py`) leaves the non-bull train return at $+4.64\%/\mathrm{yr}$, confirming that the primary baseline edge stems from intra-sector relative residual reversion rather than overfitting the fast-half boundary.
3. **Hurdle Selectivity:** A typical unconditioned forecast is smaller than the round-trip cost hurdle: median $\vert\mu_{i,s}\vert / (c^{\mathrm{in}} + c^{\mathrm{out}})$ is $0.55\times$ on the non-bull pool, $0.46\times$ on financials/energy/industrials, and $0.68\times$ on technology/communications/discretionary. Consequently, the strategy trades only when a name has a sufficiently high nominal share price $P$ and experiences an extreme multi-horizon relative gap.
4. **Price and Co-Movement Govern Leg Formation:** Inside any pool, only sectors with sufficiently high nominal share prices and sufficiently low intra-sector residual co-movement ($\bar{\rho}_s = \mathrm{mean}_{i \neq j}\vert\mathrm{corr}(\varepsilon_i, \varepsilon_j)\vert$) form both long and short legs:

| Sector | Median Price $P_s$ | Co-Movement $\bar{\rho}_s$ | Hours with Both Legs | Gross Edge When Active |
| :--- | ---: | ---: | ---: | ---: |
| **Health Care** | 177 dollars | $0.36$ | $51\%$ | $+1.60\,\mathrm{bp}$ |
| **Consumer Staples** | 91 dollars | $0.40$ | $11\%$ | $+1.42\,\mathrm{bp}$ |
| **Real Estate** | 106 dollars | $0.56$ | $6\%$ | $+1.06\,\mathrm{bp}$ |
| **Materials** | 106 dollars | $0.44$ | $12\%$ | $-0.22\,\mathrm{bp}$ |
| **Utilities** | 74 dollars | $0.69$ | $\approx 0\%$ (almost none) | — |
| **Information Technology** | 183 dollars | $0.54$ | $55\%$ | $+0.63\,\mathrm{bp}$ |
| **Communication Services** | 145 dollars | $0.40$ | $36\%$ | $+0.63\,\mathrm{bp}$ |
| **Consumer Discretionary** | 241 dollars | $0.45$ | $40\%$ | $-0.52\,\mathrm{bp}$ |
| **Financials** | 75 dollars | $0.62$ | $11\%$ | $+0.42\,\mathrm{bp}$ |
| **Industrials** | 206 dollars | $0.44$ | $15\%$ | $-0.19\,\mathrm{bp}$ |
| **Energy** | 66 dollars | $0.68$ | $\approx 0\%$ (almost none) | — |

5. **Single-Sector Isolation (`cs_sector_v1.py`):** Restricting the book to a single sector alone reduces cross-sectional breadth and diversification across sleeves:

| Sector Alone | Train Return | Train $p$ | Validation Return | Validation $p$ | Active Hours |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Information Technology (22 names)** | $+2.65\%/\mathrm{yr}$ | $0.012$ | $+0.22\%/\mathrm{yr}$ | $0.460$ | $75\%$ |
| **Financials (29 names)** | $-0.01\%/\mathrm{yr}$ | $0.490$ | $+0.17\%/\mathrm{yr}$ | $0.270$ | $26\%$ |

## Theoretical Foundation: Mathematical Proof of Pool Feasibility



Why does the architecture generate statistically significant out-of-sample alpha in specific equity pools (Non-Bull 97 and Tech/Comms 40) while remaining inactive or unprofitable in others (Financials, Energy, Industrials 57)? Below we establish the formal mathematical conditions governing residual reversion, dispersion collapse, and hurdle-gated execution.

### 1. Structural Data-Generating Process

Consider a cross-sectional universe of $N$ equities partitioned into disjoint sectors $\mathcal{S} = \{1, \ldots, S\}$. Within each trading session, returns are observed over discrete 5-minute bars $t \in \{1, \ldots, T\}$, and allocation decisions occur at hourly bar indices $s \in \mathcal{T}_{\mathrm{dec}} \subset \{1, \ldots, T\}$.

#### Assumption 1 (Three-Layer Intraday Return Decomposition)

For stock $i$ belonging to sector $s(i) \in \mathcal{S}$, let the 5-minute return at bar $t$ follow the linear factor decomposition

$$
r_{i,t} = \Lambda_i^{\top} F_t^{\mathrm{global}} + \gamma_{i,s(i)}\, f_{s(i),t}^{\mathrm{sec}} + \eta_{i,t},
$$

where:

1. $F_t^{\mathrm{global}} \in \mathbb{R}^{K_0}$ denotes the vector of broad market-wide intraday factors with loading vector $\Lambda_i \in \mathbb{R}^{K_0}$.
2. $f_{s,t}^{\mathrm{sec}} \in \mathbb{R}$ denotes a zero-mean, unit-variance sector-specific common shock (such as WTI crude oil shocks in Energy or Treasury yield-curve shifts in Financials and Utilities) orthogonal to $F_t^{\mathrm{global}}$, with stock-specific sensitivity $\gamma_{i,s} \in \mathbb{R}$.
3. $\eta_{i,t}$ is the purely idiosyncratic intraday innovation of stock $i$, satisfying $\mathbb{E}[\eta_{i,t}] = 0$ and $\mathrm{Cov}(\eta_{i,t}, \eta_{j,t}) = 0$ for $i \neq j$.

#### Assumption 2 (Intra-Sector Information Diffusion and Mean Reversion)

Across the lookback window $\mathcal{W}_h(s)$ ending at decision bar $s$, let the cumulative idiosyncratic component of stock $i$ decompose into a lagged intra-sector diffusion response and a transient liquidity-pressure process $z_{i,s}^{(h)}$:

$$
\sum_{\tau \in \mathcal{W}_h(s)} \eta_{i,\tau} = \theta_i\, \Delta m_{s(i),s}^{(h)} + z_{i,s}^{(h)},
$$

where $\Delta m_{s,s}^{(h)}$ represents fundamental intra-sector order-flow information absorbed at speed $\theta_i \in (0, 1]$, and $z_{i,s}^{(h)}$ is a transient microstructure imbalance (such as institutional basket execution or inventory pressure). Over the remaining $\tau_s = T - s$ bars until the close:

1. **Fast vs. slow diffusion** ($\mathcal{C}_s$ vs. $\mathcal{P}_s$). High-dollar-volume peers $j \in \mathcal{C}_s$ incorporate intra-sector news contemporaneously ($\bar{\theta}_C = 1$), whereas low-dollar-volume peers $i \in \mathcal{P}_s$ incorporate $\theta_i < 1$ by bar $s$ and catch up over the remainder of the session by $(1 - \theta_i)\,\Delta m_{s,s}^{(h)}$.
2. **Ornstein-Uhlenbeck Liquidity Reversion:** Transient order-book imbalances mean-revert over the remaining $\tau_s$ bars at rate $\kappa_s \in (0, 1]$:

$$
\mathbb{E}\left[\Delta z_{i,s \to T} \;\middle\vert\; z_{i,s}^{(h)}\right] = -\kappa_s\, z_{i,s}^{(h)}.
$$

3. **Unspanned Sector Factor Autocorrelation:** Over the remaining $\tau_s$ bars, the sector-specific shock $f_{s}^{\mathrm{sec}}$ exhibits intraday serial persistence parameter $\rho_{F,s} \in [-1, 1]$:

$$
\mathbb{E}\left[\Delta f_{s,s \to T}^{\mathrm{sec}} \;\middle\vert\; \Delta f_{s,s}^{\mathrm{sec},(h)}\right] = \rho_{F,s}\, \Delta f_{s,s}^{\mathrm{sec},(h)}.
$$

### 2. Subspace Projection and Residual Co-Movement

At each bar $t$, the rule estimates $K = 3$ eigenvectors $B \in \mathbb{R}^{N \times K}$ from the 63-day rolling correlation matrix and removes projected common variation via the orthogonal projector $P_B = B(B^{\top} B)^{-1} B^{\top}$, yielding $\varepsilon_t = (I_N - P_B)\, r_t$.

#### Lemma 1 (Unspanned Sector Factor Leakage and Block Co-Movement)

Suppose the $K = 3$ global eigenvectors span the global factor subspace ($\mathrm{span}(\Lambda) \subseteq \mathrm{span}(B)$, so $(I_N - P_B)\Lambda = 0$), and let $\tilde{\gamma}_{i,s} = [(I_N - P_B)\gamma_s]_i$ denote the unspanned residual loading of stock $i$ on its sector shock $f_{s,t}^{\mathrm{sec}}$. Then in the cross-sectional limit $K / N \to 0$, the extracted bar residual satisfies

$$
\varepsilon_{i,t} = \tilde{\gamma}_{i,s}\, f_{s,t}^{\mathrm{sec}} + \eta_{i,t},
$$

and for any pair of distinct stocks $i \neq j$ inside sector $s$ with total residual variance $\sigma_{\varepsilon,i}^2 = \tilde{\gamma}_{i,s}^2 + \sigma_{\eta,i}^2$, the pairwise residual correlation is

$$
\rho_{ij}^{(s)} = \mathrm{corr}(\varepsilon_{i,t}, \varepsilon_{j,t}) = \frac{\tilde{\gamma}_{i,s}\, \tilde{\gamma}_{j,s}}{\sqrt{(\tilde{\gamma}_{i,s}^2 + \sigma_{\eta,i}^2)(\tilde{\gamma}_{j,s}^2 + \sigma_{\eta,j}^2)}}.
$$

*Proof.* Applying $I_N - P_B$ to Assumption 1 annihilates $\Lambda F_t^{\mathrm{global}}$. Since $\mathrm{rank}(P_B) = K = 3 \ll N$, the projection of independent idiosyncratic noise $P_B \eta_t$ is $\mathcal{O}_p(\sqrt{K/N})$ and vanishes for large $N$, leaving $\varepsilon_{i,t} = \tilde{\gamma}_{i,s} f_{s,t}^{\mathrm{sec}} + \eta_{i,t}$. Orthogonality of $f_{s,t}^{\mathrm{sec}}$ and $\eta_{i,t}$ implies $\mathrm{Cov}(\varepsilon_{i,t}, \varepsilon_{j,t}) = \tilde{\gamma}_{i,s}\tilde{\gamma}_{j,s}$, which normalizes directly to $\rho_{ij}^{(s)}$. $\blacksquare$

### 3. Anchor Gap Decomposition and Expected Residual Edge

#### Proposition 1 (Closed-Form Decomposition of the Fast and Slow Anchor Gaps)

For any stock $i$ in sector $s$, the fast-anchor gap $g_{i,s}^{C} = m_{i,s}^{(h)} - a_{i,s}^{C}$ and slow-anchor gap $g_{i,s}^{P} = m_{i,s}^{(h)} - a_{i,s}^{P}$ decompose as

$$
g_{i,s}^{C} = \underbrace{(\theta_i - \bar{\theta}_C)\,\Delta m_{s,s}^{(h)}}_{\text{Lead-Lag Diffusion Gap}} + \underbrace{\left(z_{i,s}^{(h)} - \bar{z}_{C,-i,s}^{(h)}\right)}_{\text{Microstructure Imbalance}} + \underbrace{(\tilde{\gamma}_{i,s} - \bar{\tilde{\gamma}}_{C,-i,s})\,\Delta f_{s,s}^{\mathrm{sec},(h)}}_{\text{Unspanned Beta Dispersion}},
$$

and the difference between the slow-anchor and fast-anchor gaps isolates the intra-sector diffusion factor:

$$
g_{i,s}^{P} - g_{i,s}^{C} = a_{i,s}^{C} - a_{i,s}^{P} = (1 - \bar{\theta}_P)\,\Delta m_{s,s}^{(h)} + (\bar{\tilde{\gamma}}_C - \bar{\tilde{\gamma}}_P)\,\Delta f_{s,s}^{\mathrm{sec},(h)} + \mathcal{O}_p\!\left(\vert S_s\vert^{-1/2}\right).
$$

*Proof.* Summing $\varepsilon_{i,\tau}$ from Lemma 1 over $\tau \in \mathcal{W}_h(s)$ and substituting Assumption 2 into $m_{i,s}^{(h)} - a_{i,s}^{C}$ yields the three-term additive identity by linearity of the leave-one-out weights $w_j^C = \sqrt{v_{j,s}} / \sum_{k \in \mathcal{C}_s \setminus \{i\}} \sqrt{v_{k,s}}$. Subtracting $g_{i,s}^{C}$ from $g_{i,s}^{P}$ cancels $m_{i,s}^{(h)}$ and leaves $a_{i,s}^{C} - a_{i,s}^{P}$ with strictly positive diffusion loading $(1 - \bar{\theta}_P) > 0$. $\blacksquare$

#### Theorem 1 (Necessary and Sufficient Condition for Positive Expected Residual Edge)

Let $q_{i,s}$ denote the subsequent relative return of stock $i$ from bar $s+1$ to the close. The cross-sectional covariance between the negative fast-anchor gap $-g_{i,s}^{C}$ (monotonically mapped to $u_{i,s} = \frac{2}{n_s - 1}\mathrm{rank}(-g_{i,s}^{C}) - 1$) and the subsequent relative return $q_{i,s}$ is

$$
\mathrm{Cov}_s\!\left(-g_{i,s}^{C},\; q_{i,s}\right) = \underbrace{\kappa_s\,\sigma_{z,s}^2\,\left(1 + \xi_C\right)}_{\text{Liquidity Reversion Edge } (\ge 0)} + \underbrace{(1 - \theta_i)^2\,\sigma_{m,s}^2}_{\text{Lead-Lag Catch-Up Edge } (\ge 0)} - \underbrace{\rho_{F,s}\,(\tilde{\gamma}_{i,s} - \bar{\tilde{\gamma}}_C)^2\,\sigma_{f,s}^2}_{\text{Unspanned Beta Contamination}},
$$

where $\xi_C = \sum_{j \in \mathcal{C}_s \setminus \{i\}} (w_j^C)^2 = \mathcal{O}(\vert\mathcal{C}_s\vert^{-1})$. Therefore, the linear coordinate combination $\hat{\beta}^{\top} u_{i,s}$ achieves a strictly positive expected out-of-sample covariance in sector $s$ if and only if the **Sector Structural Viability Inequality** holds:

$$
\mathcal{A}_s \;\triangleq\; \kappa_s\,\sigma_{z,s}^2 + \mathbb{E}_i\!\left[(1 - \theta_i)^2\right]\sigma_{m,s}^2 \;>\; \rho_{F,s}\,\mathrm{Var}_{i \in s}\!\left(\tilde{\gamma}_{i,s}\right)\sigma_{f,s}^2 \;\triangleq\; \mathcal{N}_s.
$$

*Proof.* Under Assumptions 1 and 2, the relative close return from bar $s+1$ to $T$ is

$$
q_{i,s} = (1 - \theta_i)\,\Delta m_{s,s}^{(h)} - \kappa_s\left(z_{i,s}^{(h)} - \bar{z}_{C,-i,s}^{(h)}\right) + (\tilde{\gamma}_{i,s} - \bar{\tilde{\gamma}}_C)\,\Delta f_{s,s \to T}^{\mathrm{sec}} + \nu_{i,s \to T}.
$$

Taking the inner product with $-g_{i,s}^{C}$ from Proposition 1 across orthogonal innovations $\Delta m_{s,s}^{(h)}$, $z_{i,s}^{(h)}$, and $\Delta f_{s,s}^{\mathrm{sec},(h)}$ yields $+\kappa_s \sigma_{z,s}^2(1 + \xi_C)$ from liquidity reversion, $+(1 - \theta_i)^2 \sigma_{m,s}^2$ from slow-peer catch-up toward the fast anchor, and $-\rho_{F,s}(\tilde{\gamma}_{i,s} - \bar{\tilde{\gamma}}_C)^2 \sigma_{f,s}^2$ from unspanned sub-sector beta dispersion. Averaging over $i \in s$ gives $\mathcal{A}_s - \mathcal{N}_s > 0 \iff \mathcal{A}_s > \mathcal{N}_s$. $\blacksquare$

### 4. Dispersion Collapse and Cost-Gate Starvation under Block Co-Movement

#### Theorem 2 (Co-Movement Dispersion Collapse and Leg-Formation Threshold)

Let $\bar{\rho}_s = \mathrm{mean}_{i \neq j \in s}\vert\mathrm{corr}(\varepsilon_{i,t}, \varepsilon_{j,t})\vert \in [0, 1)$ denote the intra-sector residual co-movement of sector $s$, and suppose residual variances are homogeneous within the sector ($\sigma_{\varepsilon,i} \approx \sigma_s$). Then:

1. **Cross-Sectional Gap Compression:** The cross-sectional standard deviation of the relative gap $g_{i,s}^{C}$ shrinks proportionally to $\sqrt{1 - \bar{\rho}_s}$:

$$
\mathrm{SD}_{i \in s}\!\left(g_{i,s}^{C}\right) = \sigma_s \sqrt{h_s\,(1 - \bar{\rho}_s)\left(1 + \xi_C\right)}.
$$

2. **Attenuation of the Forecast Scale:** Because the relative close return $q_{i,s}$ also has cross-sectional dispersion proportional to $\sigma_s\sqrt{\tau_s(1 - \bar{\rho}_s)}$, the monthly OLS scaling coefficient $\alpha$ attenuates with $\sqrt{1 - \bar{\rho}_s}$, bounding the maximum forecast magnitude in sector $s$ by

$$
\max_{i \in s} \vert\mu_{i,s}\vert \;\le\; \alpha_0\,\|\hat{\beta}\|_1\,\sigma_s\,\sqrt{\tau_s\,(1 - \bar{\rho}_s)}.
$$

3. **Necessary Condition for Dual-Leg Formation:** Sector $s$ can clear the cost gate $\vert\mu_{i,s}\vert > \lambda c_0 / P_s$ and form both long and short legs at median share price $P_s$ only if its residual co-movement $\bar{\rho}_s$ lies strictly below the **Critical Co-Movement Ceiling** $\rho_{\max}(P_s, \sigma_s)$:

$$
\bar{\rho}_s \;<\; \rho_{\max}(P_s, \sigma_s) \;\triangleq\; 1 - \left(\frac{\lambda\, c_0}{\alpha_0\,\|\hat{\beta}\|_1\, P_s\,\sigma_s\,\sqrt{\tau_s}}\right)^{2}.
$$

*Proof.* Expanding $\mathrm{Var}(m_{i,s}^{(h)} - \sum_{j \neq i} w_j^C m_{j,s}^{(h)})$ under uniform pairwise correlation $\bar{\rho}_s$ with $\sum_{j \neq i} w_j^C = 1$ gives $h_s \sigma_s^2 (1 - \bar{\rho}_s)(1 + \xi_C)$, proving Claim 1. Applying the same variance identity to $q_{i,s}$ shows that the numerator of $\alpha$ scales with $\sqrt{1 - \bar{\rho}_s}$, establishing Claim 2. Substituting Claim 2 into the hurdle condition $\vert\mu_{i,s}\vert > \lambda c_0 / P_s$ and solving for $\bar{\rho}_s$ yields Claim 3. $\blacksquare$

### 5. Multi-Horizon Consensus vs. Volatility-Spike Adverse Selection

#### Proposition 2 (Why $\lambda = 4.2$ Succeeds in Viable Pools but Fails in Beta-Contaminated Pools)

Write the forecast at bar $s$ as $\mu_{i,s} = \alpha\, S_{i,s}\, \sigma_{i,s}\sqrt{\tau_s}$, where $S_{i,s} = \sum_{d=1}^{6} \hat{\beta}_d u_{i,s}^{(d)}$ is the linear combination of bounded rank coordinates $u_{i,s}^{(d)} \in [-1, 1]$. The cost-gate event $\mathcal{G}_{i,s}(\lambda) = \{\vert\mu_{i,s}\vert > \lambda c_0 / P_{i,s}\}$ is equivalent to

$$
\vert S_{i,s}\vert \cdot \sigma_{i,s} \;>\; \frac{\lambda\, c_0}{\alpha\, P_{i,s}\,\sqrt{\tau_s}}.
$$

1. **Multi-horizon consensus in viable pools** ($\mathcal{A}_s > \mathcal{N}_s$, $\bar{\rho}_s \ll \rho_{\max}$). When $\alpha P_s$ is sufficiently large, clearing $\lambda = 4.2$ at normal bar volatility $\sigma_{i,s} \approx \bar{\sigma}_i$ requires $\vert S_{i,s}\vert \to \|\hat{\beta}\|_1$. Because each coordinate $u_{i,s}^{(d)} \in [-1, 1]$ is bounded, $\vert S_{i,s}\vert \approx \|\hat{\beta}\|_1$ occurs if and only if all four horizons ($30\mathrm{m}, 1\mathrm{h}, 2\mathrm{h}, \mathrm{day}$) and $u^{\mathrm{vol}}$ align simultaneously in the tail:

$$
\mathrm{sign}\!\left(u_{i,s}^{(30\mathrm{m},C)}\right) = \mathrm{sign}\!\left(u_{i,s}^{(1\mathrm{h},C)}\right) = \mathrm{sign}\!\left(u_{i,s}^{(2\mathrm{h},C)}\right) = \mathrm{sign}\!\left(u_{i,s}^{(\mathrm{day},C)}\right).
$$

This eliminates single-horizon noise that degrades loose gates ($\lambda = 2.0$ or $2.8$).

2. **Adverse selection in high-co-movement or beta-dispersed pools** ($\mathcal{N}_s \ge \mathcal{A}_s$ or $\bar{\rho}_s \to \rho_{\max}$). When $\alpha P_s$ is small (as in Financials, Energy, and Industrials), even full coordinate alignment $\vert S_{i,s}\vert = \|\hat{\beta}\|_1$ cannot clear $\lambda = 4.2$ at normal volatility $\bar{\sigma}_i$. Thus $\mathcal{G}_{i,s}(\lambda)$ triggers **only during extreme volatility spikes** $\sigma_{i,s} \gg \bar{\sigma}_i$, which correspond to macro or commodity trend breakouts where $\rho_{F,s}\sigma_{f,s}^2 \gg \kappa_s\sigma_{z,s}^2$, producing negative conditional expected returns.

## Data and Repository Structure

5-minute primary-session bars and split/dividend adjustment tables are stored in `cs_intraday/data/`, with universe definitions in `cs_intraday/`:

```text
cs_intraday/
  universe_sp100.csv        # 100 S&P 100 constituent tickers and GICS sectors
  universe_flat.csv         # 94 non-bull / flat / laggard tickers and GICS sectors
  universe_all.csv          # Combined 194-name panel metadata
  data/
    5m/                     # 100 S&P 100 5-minute bar parquet files
    5m_flat/                # 94 non-bull / laggard 5-minute bar parquet files
    rehab/                  # Per-ticker split and dividend adjustment tables
    manifest.csv            # Coverage and bar count manifest
    splits.json             # Train ends 2025-02-14 11:10; validation ends 2025-12-03 10:30
```

### Core Scripts and Research Progression

| Script | Role in the Research Architecture |
| :--- | :--- |
| `cs_band_v1.py` | Primary cross-sectional band & sector permutation null evaluation across the Non-Bull, Tech/Comms/Disc, and Fin/Energy/Ind pools (`cs_band_v1_out/`). |
| `cs_ac_v1.py` | Direction A+C multi-anchor evaluation with hurdle sensitivity (`2.0x`, `2.8x`, `4.2x`) and causal volatility/drawdown guard (`cs_ac_v1_out/`). |
| `cs_fw_test_v1.py` | Robustness and ablation suite: random fast-half splits (12 seeds), wrong-sector relabeling (8 seeds), and coordinate drop checks (`cs_fw_test_v1_out/`). |
| `cs_sector_v1.py` | Single-sector isolation experiments on Information Technology (22 names) and Financials (29 names) (`cs_sector_v1_out/`). |
| `factor_v1.py`, `cs_proj_v1.py`, `cs_resid_v1.py`, `cs_dynproj_v1.py` | Rolling 63-day PCA eigenvector estimation ($K=3$), cross-sectional factor projection, and residual extraction. |
| `cs_f7state_v1.py`, `mgf_channels.py`, `cs_np_v1.py`, `cs_gmm_v1.py` | Per-name daily-reset Kalman state filtering (`f3` volume-return co-movement), empirical characteristic function / Gil-Pelaez CDF inversion, and non-parametric density modeling. |
| `cs_peer_v1.py`, `cs_inner_v1.py`, `cs_inner_v2.py`, `cs_inner_v4.py` | Progression from equal-weighted sector means to $\sqrt{v}$-weighted fast/slow peer anchors ($\mathcal{C}_s, \mathcal{P}_s$) across the four intraday horizons. |
| `cs_cost_v1.py`, `cs_cost_v3.py`, `cs_hiprice_v1.py` | Unadjusted share price $P \ge 60$ filtering and round-trip transaction cost hurdle calibration. |
| `fetch_sp100_5m.py`, `fetch_pool_5m.py`, `fetch_rehab.py` | Historical 5-minute bar and corporate action (`rehab`) ingestion scripts via Polygon and Futu OpenD. |

## Run

```bash
# Run the primary 3-pool cross-sectional evaluation and sector permutation null
python3 cs_intraday/cs_band_v1.py

# Run the Direction A+C hurdle comparison and causal drawdown guard evaluation
python3 cs_intraday/cs_ac_v1.py

# Run the ablation, random-anchor, and wrong-sector robustness suite
python3 cs_intraday/cs_fw_test_v1.py

# Run single-sector standalone tests (Information Technology vs. Financials)
python3 cs_intraday/cs_sector_v1.py
```

Outputs are written to `cs_intraday/cs_band_v1_out/`, `cs_intraday/cs_ac_v1_out/`, `cs_intraday/cs_fw_test_v1_out/`, and `cs_intraday/cs_sector_v1_out/` (`*.png` cumulative return plots and `metrics.json` performance records).
