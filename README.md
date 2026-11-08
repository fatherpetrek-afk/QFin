# QFin

Cross-sectional book on 5-minute bars. A factor model known before the bar removes the common move. Inside each sector the book buys the names that have fallen furthest relative to their peers and shorts the names that have risen furthest. Both sides have the same dollars, sectors have the same dollars, and the book is flat by the close. The market's own drift is not the position.

This repository is the rule below, the code it imports, the figures, and the bars used to compute them. Bars after the validation cut are not used in the reported numbers. This is a research record, not an order.

## Rule

### Residual

For name $i$ at bar $t$,

$$
r_{i,t} = B_{i}^{\top} f_{t} + \varepsilon_{i,t}.
$$

$B_{i} \in \mathbb{R}^{K}$ is fixed through the day. It is the top $K = 3$ eigenvectors of the correlation of 5-minute returns over the previous quarter (63 trading days), scaled back from correlation units to return units, with sign aligned to the previous estimate. Names with too few bars in that window get a zero loading and do not enter the correlation. The factor at the bar is the cross-sectional projection on the names that actually print,

$$
f_{t} = (B^{\top} B)^{-1} B^{\top} r_{t}, \qquad \varepsilon_{t} = r_{t} - B f_{t}.
$$

### Four horizons

Decisions are on the hour from 10:00 to 15:00. For a window $\mathcal{W}_{h}(s)$ ending at the decision bar $s$,

$$
m_{i,s}^{(h)} = \sum_{\tau \in \mathcal{W}_{h}(s)} \varepsilon_{i,\tau} - \mathrm{mean}_{j}\left(\sum_{\tau \in \mathcal{W}_{h}(s)} \varepsilon_{j,\tau}\right).
$$

The four windows are the last 6 bars (30 minutes), 12 bars (1 hour), 24 bars (2 hours), and every bar since the previous close. The first bar of the sample is left missing. The mean is across names at that bar.

### Sector anchor

Lagged dollar volume uses the 12 bars ending at $s-1$, so the split does not see bar $s$. Inside a sector, names at or above the median of that lagged volume are the fast half $C_{s}$. Names below it are the slow half $P_{s}$. A sector with fewer than 4 names is skipped.

The fast anchor of name $i$ leaves $i$ out and weights the others by the square root of lagged dollar volume $v$:

$$
a_{i,s}^{C} = \frac{\sum_{j \in C_{s},\, j \neq i} \sqrt{v_{j,s}}\, m_{j,s}}{\sum_{j \in C_{s},\, j \neq i} \sqrt{v_{j,s}}}.
$$

The slow anchor is the same leave-one-out mean with equal weights. If name $i$ has no other slow peer, $a_{i,s}^{P}$ falls back to $a_{i,s}^{C}$. The gaps are

$$
g_{i,s}^{C} = m_{i,s} - a_{i,s}^{C}, \qquad g_{i,s}^{P} = m_{i,s} - a_{i,s}^{P}.
$$

### Coordinates

On each decision row, a coordinate is the cross-sectional rank of the negative gap, mapped to $[-1,1]$:

$$
u_{i,s} = \frac{2}{n_{s}-1}\,\mathrm{rank}(-g_{i,s}) - 1.
$$

If the row has fewer than 20 finite names, the whole row is missing. The six coordinates are

$$
u^{\mathrm{day},C},\; u^{\mathrm{day},P},\; u^{2\mathrm{h},C},\; u^{1\mathrm{h},C},\; u^{30\mathrm{m},C},\; u^{\mathrm{vol}}.
$$

The first five are the four horizons against the fast anchor, plus the since-close gap against the slow anchor. $u^{\mathrm{vol}}$ is the same rank transform of the negative volume state (the third coordinate of the per-name Kalman state).

### Monthly fit

Let $\varepsilon_{i,s}^{\mathrm{close}}$ be the residual from the decision to the close, $\sigma_{i,s}$ the name's bar volatility, and $\tau_{s}$ the number of bars left until the close. The regression target, on names that pass the price floor, is

$$
y_{i,s} = \mathrm{clip}\left(\frac{\varepsilon_{i,s}^{\mathrm{close}}}{\sigma_{i,s}\sqrt{\tau_{s}}},\,-5,\,5\right).
$$

Once a month, on the previous 500 trading days only, drop the intercept and solve

$$
\hat\beta = \min_{\beta}\ \| y - U\beta \|_{2},
$$

where the columns of $U$ are the six coordinates. A scalar $\alpha$ then matches the fitted mix to the raw relative close-to-close move $q$ (the name's return to the close, minus the cross-sectional mean) on those same past rows:

$$
\alpha = \frac{\sum (\hat\beta^{\top} u)\, \sigma\sqrt{\tau}\, q}{\sum \left((\hat\beta^{\top} u)\, \sigma\sqrt{\tau}\right)^{2}}.
$$

The forecast used by the book is

$$
\mu_{i,s} = \alpha\, (\hat\beta^{\top} u_{i,s})\, \sigma_{i,s}\sqrt{\tau_{s}}.
$$

### Cost gate

The real price $P$ is the forward-adjusted price divided by the product of later split and dividend factors. Costs are charged on $P$, not on the adjusted price:

$$
c^{\mathrm{in}} = \frac{0.01 + 0.0035}{P}, \qquad c^{\mathrm{out}} = \frac{0.0035}{P}.
$$

The entry charge is a one-cent half-spread plus a 0.35-cent fee. The exit is the closing auction, fee only. A name is eligible when $P \ge P_{\min}$ and

$$
|\mu_{i,s}| > \lambda \left(c^{\mathrm{in}}_{i,s} + c^{\mathrm{out}}_{i,s}\right).
$$

$P_{\min} = 60$ and $\lambda = 4.2$ on the sector pools below. $\lambda$ was kept after looking at the non-bull validation, so that pool is not a locked sample. The other pools did not enter the choice.

### Book

A sector is used only if it has at least 5 names. Inside it, let $k = \lfloor 0.2\, n \rfloor$ with $k \ge 1$. The long leg is the $k$ highest eligible scores that are positive. The short leg is the $k$ lowest eligible scores that are negative. If either leg is empty the sector is flat. Sectors that do trade share one unit of gross capital equally, and each sector's long dollars equal its short dollars. The resulting target $w_{s}$ satisfies

$$
\mathbf{1}^{\top} w_{s} = 0, \qquad \| w_{s} \|_{1} \in \{0, 1\}.
$$

The six hourly targets of a day are stacked into sleeves. At hour $j = 1,\ldots,6$,

$$
h_{j} = \frac{1}{6}\sum_{q=1}^{j} w_{q}.
$$

The sleeve is applied one 5-minute bar after the decision, and $h = 0$ overnight. Daily profit is the sum of sleeve times the subsequent return, minus the turnover cost of $c^{\mathrm{in}}$ on entries and $c^{\mathrm{out}}$ on exits.

## Null

The null keeps the fitted scores and reassigns them, at that bar, to other names in the same sector. The forecast is rebuilt with the recipient's own $\sigma\sqrt{\tau}$, and the same price floor and the same $\lambda$ are applied. Eighty such books are the colored cloud. Blue is the middle of that cloud. Green is the outer edge. The reported $p$-value is $(1 + N_{\mathrm{ge}}) / 81$, where $N_{\mathrm{ge}}$ counts null books whose mean is at least the real book's. Train days and validation days are counted separately.

| Pool | Train | Train $p$ | Validation | Validation $p$ |
| --- | ---: | ---: | ---: | ---: |
| Non-bull, 97 names | +4.88%/yr | 0.012 | +4.30%/yr | 0.049 |
| Financials, energy, industrials, 57 names | +0.01%/yr | 0.30 | −1.39%/yr | 0.91 |
| Technology, communication, discretionary, 40 names | +3.25%/yr | 0.012 | +5.30%/yr | 0.012 |

The rank signal is present in all three. Mean daily rank correlation of the fitted mix with the residual to the close is +0.032, +0.042 and +0.044. Cross-sectional residual dispersion is about the same. The financials pool fails later in the rule: only 56% of financial and energy names clear $P \ge 60$, the 4.2 cost gate passes on 2.8% of name-rows (16.9% in the technology pool, where the median real price is about 230 dollars and the round trip is 0.7 bp), and a sector book exists on only 45% of hours. Gross profit on that pool is already about zero, so the cost is not what removes an otherwise positive book. The non-bull sectors stay thick enough to fill both legs. The technology pool clears the gate often because its round trip is a smaller fraction of price. Figure: `cs_intraday/cs_band_v1_out/cs_band.png`.

Shuffling which names are in the fast half, or shuffling the sector labels, does not remove the non-bull train result. The since-close relative gap and the cost gate are the parts that the book uses. The fast/slow label is not a separate source of profit.

## Data

Bars are 5-minute, regular session, forward-adjusted, one parquet per name, columns `time, open, high, low, close, volume`. Downloaded through Futu OpenD.

| Path | Contents |
| --- | --- |
| `cs_intraday/data/5m/` | S&P 100 names, `universe_sp100.csv` |
| `cs_intraday/data/5m_flat/` | Added sector and laggard names, `universe_flat.csv` |
| `cs_intraday/data/rehab/` | Split and dividend events used to recover $P$ |
| `cs_intraday/data/splits.json` | Train, validation, and test cuts |
| `cs_intraday/data/manifest.csv` | Download record for the S&P side |

The cut is chronological by bar, with a 60-minute embargo between pieces.

- Train ends 2025-02-14 11:10
- Validation: 2025-02-14 12:15 through 2025-12-03 10:30
- Test starts 2025-12-03 11:35. Loaders drop every bar on or after the validation end date.

`data/5m_all/` is not in the repository. The first run hard-links the two bar directories into it and rewrites `universe_all.csv`. Names with train coverage under 90% of the best name are dropped (GEV, PLTR, RTX, SNDK). EQR and AVB have no parquet.

## Run

```bash
pip install -r requirements.txt
cd cs_intraday
python cs_ac_v1.py
python cs_fw_test_v1.py
python cs_band_v1.py
```

Each script rewrites its figure and `metrics.json` under the matching `*_out` directory. OpenD is required only to download bars again.

- `cs_intraday/cs_ac_v1_out/cs_ac.png` — equity of the non-bull pool and of the full 194-name panel
- `cs_intraday/cs_fw_test_v1_out/cs_fw_test.png` — coordinates, anchor, lag, and sign, at the same gate
- `cs_intraday/cs_band_v1_out/cs_band.png` — the reassignment cloud on the three pools
