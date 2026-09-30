# KTZ × PHANGS–MUSE 工作区

将 KTZ（KT18/KT25 金属关联理论 + 螺旋臂非均匀注入 + 源成团）应用于 PHANGS–MUSE 四个星系（NGC 628、1566、4254、4303）的完整、可复现的分析。本目录于 2026-09-29 从 `KTZ_theory_observations_2026-09-19/`（观测部分）和 `claude_ngc628_diagnosis_2026-09-19/`（报告与诊断说明）整理而来，只包含观测应用；理论稿与数值验证留在 `Non_uniform_KT/` 和原快照中。

符号一律采用理论稿 `Non_uniform_KT/dimensionless_KTZ+numerical_validation.tex`：ℓ*（输运尺度）、ε（注入宽度比）、η（臂幅，代码 `eta`）、β_src（均值场强度，代码 `beta`/`beta_src`）、h_R（速率标长；代码 `rate_log_slope` = −1/h_R 为内部坐标，`inv_h_R` 为对外报告量）、β_src𝒜_src 与 ℓ_src（可选成团项）、σ_q（指向零点项，观测扰动参数，不属于 KTZ）。

## 目录结构

```
KTZ_PHANGS/
├── README.md                          本文件
├── KTZ_obs.tex / KTZ_obs_diff.tex     观测稿（干净版 / 红字修订版，后者为当前版本）
├── example.bib, mnras.cls, mnras.bst  编译所需（需 newtxtext/newtxmath/ulem/cancel 宏包）
├── NGC<n>_PHANGS_halpha_map_systematic.ipynb   Stage 1（4 个主 notebook）
├── NGC<n>_PHANGS_exact_window_joint.ipynb      Stage 2（4 个，论文主结果）
├── NGC<n>_PHANGS_halpha_map_fwhm{0p5,1p5}_systematic.ipynb   Stage 1 的平滑尺度对照
├── NGC<n>_PHANGS_implications.ipynb            PHANGS 掩膜对照（也是生成器的源 notebook）
├── NGC<n>_PHANGS_logspiral_systematic.ipynb    Hα 对数螺旋谐波对照
├── NGC628_PHANGS_window_projection_refit.ipynb NGC 628 精确窗口首次实现（历史/审计）
├── NGC628_PHANGS_source_integral_refit.ipynb   NGC 628 全局源积分参考点（历史/审计，缓存指纹只在 Mac 上匹配）
├── scripts/                           全部脚本（见下）
├── data/phangs/                       PHANGS–MUSE DR1 产品（MAPS 立方图、星云目录与掩膜、SDSS r 图、环境掩膜）
├── outputs/implications/              每个 notebook 一个子目录：图、csv/json 表、npz 缓存
├── figs/                              论文图（pdf）
└── docs/                              中文工作报告（2026-09-23）与 NGC 628 指向诊断说明（附录 A–E.10）
```

## 分析流程

**Stage 1 —— `NGC<n>_PHANGS_halpha_map_systematic.ipynb`**（由 `scripts/make_halpha_map_systematic_notebooks.py` 从 `*_implications.ipynb` 生成）
从 PHANGS–MUSE 线图出发：PSF 匹配到 120 pc、60 pc 面向网格、Li21 选择（S/N ≥ 2、BPT、EW(Hα) > 6 Å）、Pilyugin & Grebel (2016) S 标定、0.2 kpc 环带去均值 → 丰度残差像元图；穷举配对统计 ξ_{3,pp}(r)（83 个 0.06 kpc bin，r < 5 kpc）；48 块 jackknife + 线流量 Monte Carlo + 选择系统误差 → 正则化协方差（收缩 0.35、下限 0.01）；1 kpc 平滑的 Hα 源模板；发射器网格；平稳等效环带响应下的七参数 MCMC（保留为对照）。产物：`<g>_muse_scal_metallicity_residual_pixel_map.npz`、`<g>_residual_covariance_diagnostics.csv`、`<g>_fit_covariance.npz`、`<g>_morphology_summary.json`、发射器与链缓存。

**Stage 2 —— `NGC<n>_PHANGS_exact_window_joint.ipynb`**（由 `scripts/make_exact_window_joint_notebooks.py` 生成；论文表 4、图 3、表 5 来自这里）
读取 Stage 1 产物，只改观测算子与数据向量：
1. 指向瓦片：从 Hβ 误差面的噪声底读出 OB 矩形（`scripts/phangs_pointing_tiles.py`，`RECT` 按星系记录）；
2. 像元对拆成同指向 / 跨指向，算 ξ_same、ξ_cross、D(r)=ξ_same−ξ_cross（10 个 0.3 kpc bin）、jackknife 协方差、平移瓦片零检验；
3. 精确窗口响应 P C Pᵀ（`scripts/ktz_exact_window.build_window_response`）：每个 KTZ 成分在实际掩膜上逐对累加，均值场按 (Pm)(Pm)ᵀ 投影，同时累加指向指示量 Σ_p w_p w_pᵀ（w_p = P 1_p）；
4. 联合协方差 [ξ_all(83), D(10)]；
5. 采样：run A（σ_q 自由，U[0,3]）、run R0（σ_q≡0）、精确窗口 KT18 参照；64 走子 × 10⁵ 步；平坦 η 先验；ℓ_src ≥ 0.12 kpc；
6. 可选 `RUN_EVIDENCE=True` 或 `python -m scripts.run_exact_window_evidence NGC628 ...`：dynesty 嵌套模型证据（L0/L0q/L1/L1q/L1n/L1nq/L2/L2q）。

**汇总与检验**
- `scripts/make_phangs_exact_window_figure.py` → `figs/phangs_four_galaxy_exact_window.pdf`、`outputs/implications/phangs_exact_window_posteriors.{json,tex}`
- `scripts/n2s2_pointing_check.py` → `outputs/implications/hbeta_free_calibration_check/`：用无 Hβ 的 Dopita+16 N2S2–Hα 标定重建残差图，比较指向签名与逐块线比均值
- `scripts/make_phangs_four_galaxy_figure.py`、`summarise_halpha_smoothing_sensitivity.py`、`ktz_physical_validation.py`、`ktz_maximum_likelihood_diagnostic.py`、`embed_ktz_physical_validation_notebooks.py`：Stage 1 的论文图、平滑敏感性表、H II 区外部检验与最大似然诊断

## 如何运行

环境：Python ≥ 3.10，numpy、scipy、astropy、matplotlib、dynesty（证据）、nbconvert/nbclient（批量执行）。在本目录（`ROOT = Path.cwd()`）下运行：

```bash
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 NGC628_PHANGS_halpha_map_systematic.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 NGC628_PHANGS_exact_window_joint.ipynb
python scripts/make_phangs_exact_window_figure.py
python -m scripts.n2s2_pointing_check            # 约 3 min/星系
python -m scripts.run_exact_window_evidence NGC628 NGC1566 NGC4254 NGC4303   # 约 15 min/星系
```

所有耗时步骤都带签名缓存（`outputs/implications/...`）：命中时 Stage 1 约 1–2 min、Stage 2 约 3 min；不命中时 Stage 1 的 MCMC 约 10 min（50k 步）、发射器数分钟到数小时（C_NP 发射器最慢），Stage 2 的窗口响应每个子集 6–25 min（随像元数平方增长）、每条链约 7 min。缓存签名包含浮点数据的哈希，换机器（Mac ↔ Linux）会导致链缓存失效并重算；固定随机种子保证重算结果到小数点后三位一致。implications 类 notebook 峰值内存约 5 GB。`NGC628_PHANGS_source_integral_refit.ipynb` 的参考点缓存有指纹检查，只能在生成它的 Mac 上重跑。

## 本目录未包含的大文件

为控制体积（OneDrive 同步），只有 4 个主 notebook 与 4 个 Stage 2 目录带全部缓存（链、发射器）；`fwhm*`、`implications`、`logspiral` 与两个 refit 目录只保留 csv/json/png 和 ≤ 20 MB 的 npz，缺的链 / C_NP 发射器缓存在重跑时自动重建（原件仍在 `KTZ_theory_observations_2026-09-19/outputs/implications/`）。`data/phangs/` 完整（2.2 GB）；如缺失，Stage 1 的第 3 个单元会从 ESO / CANFAR 自动下载并校验 SHA-256。

## GitHub 同步范围

GitHub 仓库保留代码、notebook（含已有输出）、文稿、结果图表和 CSV/JSON 汇总。原始数据 `data/` 与所有 `*.npz` 数值缓存由 `.gitignore` 排除，本地文件仍然保留。首次克隆后，按上面的分析流程依次运行 Stage 1 和 Stage 2；Stage 1 会下载所需原始数据，缺失的数值缓存会在运行时重建。首次运行的耗时因此高于已有缓存的情况。

## 当前主要结果（Stage 2，run A）

| 星系 | ℓ* [kpc] | η | ℓ_src [kpc] | σ_q | f_q | χ²_joint A / σ_q=0 / KT18 | 指向零检验 z |
|---|---|---|---|---|---|---|---|
| NGC 628 | 0.152 [0.134, 0.174] | 0.98 | 0.71 | 0.34 [0.23, 0.42] | 4.9 % | 10.5 / 14.1 / 28.1 | 2.8 |
| NGC 1566 | 0.167 [0.134, 0.208] | 0.16 | 0.56 [0.39, 0.74] | 0.23 [0.09, 0.37] | 2.0 % | 4.9 / 5.6 / 19.2 | 1.0 |
| NGC 4254 | 0.195 [0.146, 0.264] | 0.46 | 0.35 [0.17, 0.59] | 0.12 [0.04, 0.22] | 0.6 % | 7.8 / 7.5 / 7.5 | 0.0 |
| NGC 4303 | 0.244 [0.169, 0.359] | 0.16 | 0.50 [0.25, 0.69] | 0.32 [0.14, 0.47] | 3.4 % | 10.2 / 11.3 / 9.2 | 3.2 |

证据（相对 L0）：NGC 628 需要臂均值场（L1q +6.3）；NGC 1566 需要各向同性成团项（L1n +1.0）；NGC 4254、4303 与 KT18 相容（L0 最优）。KT18 单独下 ℓ* 为 0.37–0.68 kpc，含 kpc 尺度成分的任何模型给 0.15–0.25 kpc。

指向零点：在 NGC 628、4303 显著（σ_q ≈ 0.3，即 ≈ 0.01 dex rms/块），另两个为上限；机制上表现为**加性的逐 OB 线流量残差**（逐块均值跟随 [N II]/Hα，在暗像元中约大 2 倍），并非仅 Hβ 定标；无 Hβ 的 N2S2 标定无法判定偏移是否消失（噪声大 1.5–2 倍）。

## 已知限制与待办

- 协方差偏保守（四星系 χ² 都远低于数据数），形状张力（NGC 1566 的外围正平台、NGC 4303 的负尾）可能比 χ² 显示的更大；Q̂ 只是描述性量。
- t* 在三个星系贴先验下界 0.05 Gyr，κ、t* 单独为标定条件下的量。
- NGC 4303 螺距：Hα 谐波 44°，C_M 零点 ≈ 30°，S4G 3.6 µm 11.3° ± 2.4°（Díaz-García et al. 2019）；示踪物间差异是主要几何系统误差。
- 指向零点的直接验证需要 ESO 归档的逐 OB 立方（未做）。
- 四个星系不足以做 κ / t* 的标度检验（约需 20 个）。

## 历史与出处

- 2026-09-15 至 09-19：Stage 1 与三种源几何对照（原快照）。
- 2026-09-19 至 09-23：NGC 628 指向零点诊断、精确窗口、联合拟合、稳健性（`docs/README_NGC628_pointing_diagnosis.md`、`docs/KTZ_PHANGS_report_2026-09-23.pdf`）。
- 2026-09-25：符号统一到理论稿，全部 notebook 重跑。
- 2026-09-29：诊断管线并入快照成为 Stage 2，N2S2 检验，螺距文献，观测稿更新（`KTZ_obs_diff.tex`）。
