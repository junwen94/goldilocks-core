# goldilocks-core 包走读

本文基于当前工作树整理，目标是回答四个问题：

- 这个包整体想解决什么问题
- 架构如何分层，每一层的边界在哪里
- 每个代码文件负责什么
- 本仓库当前的开发、测试、发布流程是什么

## 一句话概括

`goldilocks-core` 是一个面向 DFT 输入推荐的 Python 包。它从晶体结构出发，做结构事实分析，再结合用户计算意图、启发式规则、伪势元数据和可选 ML 模型，生成 Quantum ESPRESSO 可用的参数建议和 `pw.x` 输入文件。

当前代码已经不只是旧文档里说的 k-mesh 和 UPF 解析两条线。当前主线更接近：

```text
Load -> Analyze -> Advise -> Generate
```

其中：

- `Load`: 读入 `pymatgen.Structure`
- `Analyze`: 提取事实，不做参数建议
- `Advise`: 把事实和用户意图转成参数决策
- `Generate`: 把参数决策写成 QE 输入文件

## 当前状态提醒

本地仓库当前在 `main` 分支，工作树有大量未提交变更，并且本地 `main` 领先 `origin/main` 21 个提交。GitHub CLI 未登录，所以无法读取远端 issue / PR 状态。

这对理解代码有两个影响：

- 当前代码很可能是一次较大的重构中间态。
- `README.md` 和 `docs/architecture.md` 有部分内容落后于源码，例如还提到旧的 `advisors/` 包。

## 核心设计思路

### 1. 事实和建议分离

`analyse/structure.py` 只回答“结构是什么样”：

- 元素、原子数、物种数
- 空间群、点群、晶系、反演对称
- 是否含 d/f 电子、重元素、磁性候选元素
- 维度、是否 slab、是否真空
- 极性、无序、警告

它不直接决定 k 点、smearing、spin 或 cutoff。

真正的参数建议放在 `advise/`：

- `protocol.py` 选协议
- `smearing.py` 选占据展宽
- `kpoints.py` 选 k 点网格
- `spin.py` 选磁性和 SOC 处理
- `pseudo.py` 选伪势 family 和文件
- `basis.py` 选平面波 cutoff
- `pipeline.py` 汇总成 `QEParameterSet`

这个设计让事实层可以复用，也方便将来把 QE 之外的后端接上。

### 2. 决策对象带来源和理由

`advise/types.py` 里的每个 decision 都带：

- `provenance`: `heuristic`, `ML`, `MLIP`, `user_hint`
- `rationale`: 解释为什么这样推荐

CLI 可以把这些解释直接展示给用户。这是本包的一个重要产品设计点：不是只吐参数，而是说明参数从哪里来。

### 3. 启发式优先，可选 ML 增强

当前包的默认路径不依赖重型 ML：

- k 点没有 ML 时走协议 k-distance
- 磁性没有 mMACE 分类器时走元素启发式
- 伪势没有外部目录时优先用 bundled PseudoDojo，找不到再返回占位选择

`ml/loader.py` 采用 `try_load_*` 风格，失败返回 `None`，调用者自动 fallback。

### 4. CLI 薄封装，核心逻辑在包 API

`cli/` 负责参数解析、交互、展示和调用包 API。真正逻辑在 `io`, `analyse`, `advise`, `generate`, `pseudo`, `ml`, `kmesh`。

## 主流程

### 非交互命令 `gl input`

```text
cli.commands.input.run
  -> io.structures.load_structure
  -> analyse.structure.analyze_structure
  -> intent.CalculationIntent
  -> advise.pipeline.build_qe_parameter_set
  -> rich table display
```

它只展示推荐，不自动写输入文件。

### 交互命令 `gl`

```text
cli.main.main
  -> cli.wizard.main.wizard
    -> pre_analysis.run
       -> load_structure
       -> analyze_structure
       -> optional try_load_magnetic_classifier
    -> input_kit.run
       -> collect task / accuracy / hints
       -> optional try_load_kpoints_predictor
       -> build_qe_parameter_set
       -> optional generate.qe.write_qe_inputs
```

这条路径可以生成 `goldilocks_output/goldilocks.in` 和 `goldilocks_output/pseudo/*.upf`。

### 旧入口 `goldilocks-kmesh`

这是一个窄用途 argparse 命令：

```text
cli.cli_kmesh.main
  -> load_structure
  -> analyze_structure
  -> CalculationIntent
  -> build_qe_parameter_set
  -> print kpoints_grid
```

它现在已经不是最完整的用户入口，更像兼容/快速查看 k-mesh 的命令。

## 目录与文件说明

### 根目录

- `AGENTS.md`: 本仓库给 agent 的开发规则。重点是用 `uv`，不要直接 push/merge `main`，PR 必须 close issue，GitHub 评论要声明 agent 代写。
- `CLAUDE.md`: 额外 agent/协作上下文文件，当前未纳入核心包逻辑。
- `README.md`: 用户入口文档，但当前有旧路径内容，部分 API 已落后于源码。
- `pyproject.toml`: 项目元数据、依赖、命令入口、ruff/mypy/pytest 配置。定义 `gl` 和 `goldilocks-kmesh` 两个脚本。
- `uv.lock`: `uv` 锁文件。
- `mkdocs.yml`: MkDocs 文档站配置。
- `Fe_bcc.cif`: 示例结构文件。
- `goldilocks_output/goldilocks.in`: 已生成的 QE 输入文件。
- `goldilocks_output/pseudo/Fe.upf`: 已复制出的伪势文件。
- `plan-code-package-cryptic-reef.md`: 计划/工作记录类文件，不参与包运行。

### `.github/workflows`

- `ci.yml`: GitHub Actions CI。跑 ruff check、ruff format --check、mypy、pytest 3.12/3.13、mkdocs strict build。
- `publish.yml`: tag `v*.*.*` 时构建 wheel/sdist，并通过 PyPI trusted publishing 发布。

### `src/goldilocks_core/__init__.py`

包顶层导出：

- `build_kmesh_entries`
- `infer_features`

这是很窄的公共面。当前完整推荐流水线没有从顶层导出。

### `intent.py`

定义 `CalculationIntent`，是推荐流水线的第一等输入对象。它把结构和用户意图放在一起：

- `structure`
- `code`
- `task`
- `xc`
- `pseudo_family`
- `accuracy`
- `hints`

`hints` 是用户覆盖参数的统一入口。

### `shared/types.py`

共享数据类型：

- `PathLike`
- `StructureInput`
- `StructureFeatureVector`
- `KMeshEntry`

`StructureFeatureVector` 保存 ML 特征值和特征名，避免特征顺序变成隐式约定。`KMeshEntry` 是 k-mesh 扫描结果的结构化记录。

### `io/structures.py`

结构读取入口。接受：

- 已经存在的 `pymatgen.Structure`
- 文件路径字符串
- `Path`

内部用 `Structure.from_file` 读取，并把缺文件、格式不支持、类型错误分成不同异常。

### `io/db_search.py`

通过公开 OPTIMADE 接口搜索材料数据库，不需要 API key。支持：

- Materials Project
- Materials Cloud MC3D
- NOMAD
- JARVIS

核心对象是 `SearchResult`。`search_databases()` 会并发查询四个源，返回 `(results, errors)`。

### `analyse/structure.py`

结构事实分析层。核心是：

- `StructureAnalysis`
- `analyze_structure()`

它使用 pymatgen 的 `SpacegroupAnalyzer` 和 dimensionality 工具，提取 composition、symmetry、electronic、magnetic、SOC、geometry、polarity、disorder、warnings。

重要边界：这个文件只做事实观察，不产生 QE 参数。

### `advise/types.py`

推荐层的数据契约。主要类型：

- `Protocol`
- `SmearingDecision`
- `KPointsDecision`
- `SpinDecision`
- `CutoffDecision`
- `PseudoSelection`
- `QEParameterSet`

这里体现了两层设计：

- code-agnostic decisions: smearing/kpoints/spin/cutoff/pseudo
- QE-specific parameter set: occupations、degauss、kpoints、ecut、nspin、noncolin、lspinorb 等

### `advise/protocol.py`

从 `accuracy` 和结构特征选协议：

- `fast`
- `balanced`
- `stringent`

规则包括：

- 含 f 元素的金属强制 stringent
- 绝缘体不会用 stringent，而是 capped at balanced

### `advise/smearing.py`

决定 occupation broadening。关键规则：

- `metallic`, `likely_metallic`, `unknown` 强制 smearing
- insulating 默认 fixed occupations
- 用户可通过 `smearing_method` 和 `smearing_width_ev` 覆盖
- 对金属/unknown 有 guardrail，不能被覆盖成 fixed

### `advise/kpoints.py`

决定 Monkhorst-Pack 网格和 shift。优先级：

1. `hints["kpoints_grid"]`
2. ML `k_index`
3. ML `k_distance_ml`
4. 协议默认 `k_distance`
5. `hints["k_distance"]` 可覆盖 heuristic k-distance

非周期方向会被 clamp 到 1。

### `advise/spin.py`

决定磁性、非共线和 SOC：

- 用户 `spin_treatment` 最优先
- 如果 `StructureAnalysis` 有 ML magnetic prediction，则使用 ML
- 否则根据磁性元素和重元素启发式 fallback

它还生成初始磁矩和非共线角度。生成 QE 输入时，磁矩会再除以伪势 `z_valence` 转成 QE 的 `starting_magnetization(i)`。

### `advise/pseudo.py`

决定每个元素使用哪个伪势。规则：

- 默认用 `intent.pseudo_family`
- 若结构 SOC relevant 且 family 是 SR，则自动换成 FR
- 如果 hints 里有 `pseudo_family`，优先用户指定
- 如果有 `pseudo_dir`，扫描本地 UPF
- 否则尝试 bundled data
- 如果都没有，返回 path 为 `None` 的占位选择，交给运行时或 aiida-pseudo 解决

它还从 UPF/SSSP 元数据中提取 recommended cutoff，供 `basis.py` 抬高 cutoff floor。

### `advise/basis.py`

决定平面波 cutoff。当前规则：

- accuracy tier 给 norm-conserving 默认 floor
- 若伪势元数据提供更高 cutoff，则用伪势元数据抬高
- `ecutwfc_ev` 和 `ecutrho_ev` 必须同时提供才完全覆盖

输出单位是 eV，`pipeline.py` 再转成 Ry。

### `advise/pipeline.py`

推荐流水线总装入口：

```text
select_protocol
-> advise_smearing
-> advise_kpoints
-> advise_spin
-> advise_pseudos
-> advise_basis
-> QEParameterSet
```

这里还负责把 code-agnostic decision 翻译到 QE 表达：

- smearing method: `marzari_vanderbilt -> mv`
- spin treatment: `non_magnetic -> nspin=1`, `collinear -> nspin=2`, non-collinear -> `nspin=4` 语义
- cutoff: eV -> Ry

### `kmesh.py`

k 点网格基础算法：

- `k_distance_to_mesh()`: reciprocal lattice length / k-distance -> grid
- `generate_candidate_k_distances()`: 从 reciprocal lengths 生成候选 spacing
- `build_k_distance_intervals()`: 生成 spacing 区间和对应 mesh
- `mesh_to_k_line_density_interval()`: 从 mesh 反推 line-density 区间
- `mesh_to_k_pra()`: 计算 k-points per reciprocal atom
- `mesh_to_n_reduced_kpoints()`: 用 symmetry 计算 irreducible k-points 数
- `build_kmesh_entries()`: 汇总成 indexed `KMeshEntry`

这是中立几何/对称性模块，不应该放任务策略。

### `pseudo/metadata.py`

低层伪势元数据对象：

- `PseudoMetadata`: UPF 解析结果
- `PseudoSelection`: 旧/低层选择对象，和 `advise.types.PseudoSelection` 名称相同但语义不同

注意：这里存在同名类型，后续可以考虑收敛，避免 API 混淆。

### `pseudo/parse_upf.py`

UPF 解析核心。支持：

- attribute-style `PP_HEADER`
- text-style `PP_HEADER`
- `PP_INFO` 辅助 relativistic 信息
- SSSP sidecar JSON cutoff 信息

还提供：

- `parse_upf_metadata()`
- `parse_upf_folders()`
- `metadata_to_row()`
- `metadata_list_to_dataframe()`

这是伪势底层解析层，不应掺入推荐策略。

### `pseudo/registry.py`

本地伪势 registry 工具：

- 递归加载 `.upf` / `.UPF`
- 按 element / functional / pseudo_type / relativistic 过滤

### `pseudo/policy.py`

对 `PseudoMetadata` 列表应用过滤策略：

- relativistic mode
- preferred functional
- allowed sources
- allowed pseudo types

当前 `advise/pseudo.py` 没直接使用这个 policy，而是用 registry filter 做 family 解析。

### `data/__init__.py`

访问 bundled 数据：

- `pseudo_dir(family)`
- `model_dir(task, version)`
- `available_pseudo_families()`

数据目录当前包含：

- `data/pseudopotentials/PseudoDojo/0.4/PBEsol/SR/standard/upf/*.upf`
- `data/pseudopotentials/PseudoDojo/0.4/PBEsol/FR/standard/upf/*.upf`
- `data/models/kpoints/1.0/manifest.json`
- `data/models/metallicity/1.0/is_metal.ckpt`
- `data/models/metallicity/1.0/atom_init.json`
- `data/models/magnetic_classifier/1.0/magnetic_clf.pt`

### `ml/features.py`

稳定的公共结构特征接口。提供：

- composition features
- structure features
- lattice features
- reciprocal lattice features
- CSLR 拼接特征
- `infer_features()` 作为 goldilocks-models 期望的公共名字

### `ml/inference.py`

非常薄的 sklearn-like 推理函数：

- 要求 model 有 `predict`
- 把 `StructureFeatureVector.values` reshape 成 `(1, -1)`
- 返回第一个 scalar prediction

### `ml/models.py`

加载传统 ML artifact：

- `ModelManifest`
- `LoadedModel`
- `load_model(manifest_dir)`
- `load_model_from_hf(repo_id, revision, cache_dir)`

本地 manifest 至少需要 `manifest.json` 和模型文件。支持的模型类型目前是 `random_forest`, `xgboost`, `gradient_boosting`。

### `ml/loader.py`

便利加载器，失败时返回 `None`：

- `try_load_kpoints_predictor()`: 加载 CGCNN + QRF k-spacing predictor
- `try_load_magnetic_classifier()`: 找 mMACE backbone，再加载磁性分类器

这个文件承担 graceful fallback 边界。

### `ml/magnetic.py`

磁性分类器。思路是：

- 用 mMACE backbone 提取结构 embedding
- mean-pool 到结构向量
- 用 bundled MLP checkpoint 做二分类

外部 mMACE backbone 不 bundled，需要通过参数、环境变量或 `~/.goldilocks/models/...` 提供。当前输出 label 是 `non_magnetic` 或 `collinear`，类型上预留了 `non_collinear`。

### `ml/kpoints/predictor.py`

k-spacing 预测器。组合：

- bundled CGCNN metallicity model
- matminer / SOAP / lattice structure features
- HuggingFace 上下载的 quantile regression forest

`predict()` 返回 `(kdist, kdist_upper, kdist_lower)`，单位 Å⁻¹。

### `ml/kpoints/features.py`

kpoints 专用特征工程代码，偏研究代码风格。包含：

- formula normalization
- matminer composition features
- matminer structure features
- lattice features
- SOAP features

这里依赖 `dscribe`, `matminer`, `pymatgen`，并会在部分失败路径打印 warning。

### `ml/kpoints/atom_features.py`

CGCNN/ALIGNN 图模型用的原子特征：

- 从 `atom_init.json` 读 atomic embeddings
- 可选拼接 per-atom SOAP
- 返回每个 site 的 node feature

### `ml/kpoints/cgcnn_graph.py`

把 `pymatgen.Structure` 转成 PyTorch Geometric graph：

- radius neighbor graph
- CrystalNN neighbor graph

节点特征来自 `atom_features.py`，边特征是距离。

### `ml/kpoints/cgcnn.py`

CGCNN 的 PyTorch Geometric 实现：

- `Standardize`
- `RBFExpansion`
- `CGCNNConv`
- `CGCNN_PyG`

支持 classification、regression、robust regression、quantile regression，也提供 `extract_crystal_repr()` 给 metallicity feature extraction。

### `ml/kpoints/alignn_graph.py`

构建 ALIGNN 需要的 atomic graph 和 line graph：

- atomic graph 表示 bond
- line graph 表示 bond angle
- angle feature 用 cosine 表达

### `ml/kpoints/alignn.py`

ALIGNN 的 PyTorch Geometric 实现：

- edge-gated message passing
- ALIGNNConv
- RBF expansion
- optional additional compound features
- 多种输出头

当前主 k-spacing predictor 用的是 CGCNN metallicity path，ALIGNN 更像保留的研究模型能力。

### `generate/qe.py`

把 `QEParameterSet` 写成 Quantum ESPRESSO `pw.x` 输入：

- 创建输出目录和 `pseudo/`
- 复制已解析到 path 的 UPF
- 组织 `CONTROL`, `SYSTEM`, `K_POINTS automatic`
- 根据 noncolin / lspinorb / nspin 写 QE spin flags
- 把 μB 初始磁矩除以 UPF `z_valence` 转成 `starting_magnetization(i)`
- 输出 `goldilocks.in`

返回：

```python
{"input_file": Path, "pseudo_dir": Path, "missing_pp": list[str]}
```

### `cli/main.py`

Typer 主入口，脚本名是 `gl`。行为：

- 无参数进入 wizard
- `gl wizard` 进入 wizard
- `gl input` 调用非交互输入推荐

### `cli/commands/input.py`

`gl input` 命令实现。职责：

- 校验 task / accuracy / code / xc
- 解析 `--hints key=value`
- 加载结构
- 分析结构
- 构建 `CalculationIntent`
- 运行 `build_qe_parameter_set`
- 用 Rich 表格展示结构分析、推荐参数、provenance、可选 rationale

### `cli/cli_kmesh.py`

旧的 `goldilocks-kmesh` 命令。只接受 structure 和 accuracy，打印推荐 mesh、shift、protocol rationale。

### `cli/wizard/main.py`

交互 wizard 顶层菜单：

- Pre-Analysis
- Database Search
- Code-agnostic Input Kit
- QE Inputs
- Submit Playground, Results Lab 目前只是 coming soon 文案

### `cli/wizard/_context.py`

`WizardContext`，在 wizard 步骤间传递：

- structure path
- structure
- analysis
- task
- accuracy
- hints

### `cli/wizard/pre_analysis.py`

wizard 的结构预分析和数据库搜索步骤：

- 询问结构文件并加载
- 尝试加载 magnetic classifier
- 调用 `analyze_structure`
- 展示结构事实
- 也可按 formula 或结构文件搜索 MP / MC / NOMAD / JARVIS

### `cli/wizard/input_kit.py`

wizard 的推荐和生成步骤：

- 收集 task
- 收集 accuracy
- 收集 hints
- 尝试加载 kpoints predictor
- 运行 advise pipeline
- 支持 code-agnostic 展示
- 支持 QE 参数展示
- 可选择调用 `write_qe_inputs()` 生成文件

### 空 `__init__.py` 文件

以下文件主要用于标记 package，目前没有运行逻辑：

- `analyse/__init__.py`
- `cli/__init__.py`
- `cli/commands/__init__.py`
- `cli/wizard/__init__.py`
- `generate/__init__.py`
- `io/__init__.py`
- `ml/kpoints/__init__.py`
- `pseudo/__init__.py`
- `shared/__init__.py`

`ml/__init__.py` 只导出 `infer_features` 和 model loader 相关对象。

## 测试文件说明

- `tests/test_cli_kmesh.py`: 测 `goldilocks-kmesh` parser 的必需参数和 accuracy flag。
- `tests/test_kmesh.py`: 测 k-distance 到 mesh、候选 k-distance、区间构建、`KMeshEntry` 构建。
- `tests/test_ml_features.py`: 测 lattice features 和 CSLR 特征拼接顺序。
- `tests/test_ml_inference.py`: 用 dummy sklearn-like model 测 `predict()`。
- `tests/test_ml_models.py`: 测 manifest model loader 的成功和错误路径。
- `tests/test_parse_upf.py`: 用本地 `local_data` 中真实 UPF 测解析器。
- `tests/test_pseudo_policy.py`: 测 `PseudoPolicy` 多条件过滤。
- `tests/test_pseudo_registry.py`: 用临时合成 UPF 测 registry 加载和过滤。
- `tests/test_structures.py`: 测结构加载、缺文件、unsupported XYZ。

注意：`test_parse_upf.py` 引用了 `local_data/pseudopotentials/...`，而 `local_data/` 通常是 gitignored。如果 CI 没有这些文件，这些测试可能需要调整为 bundled fixture 或 synthetic UPF。

## 当前开发流程

### 环境

本项目使用 `uv`，不要用 `pip` 管理开发环境。

常用命令：

```bash
uv sync --group dev
uv run pytest
uv run ruff check src tests
uv run ruff format src tests
uv run pre-commit run --all-files
```

### 代码风格

来自 `AGENTS.md` 和 `pyproject.toml`：

- Python 3.12+
- Ruff: `E`, `F`, `I`
- 公共 API 加类型标注
- dataclass 默认 `slots=True`，不可变值对象用 frozen
- 每个模块顶部使用 `from __future__ import annotations`
- 不建泛型 `helpers/`, `utils/`, `processing/`
- 领域模块边界优先

### GitHub/PR 流程

项目规则：

- 不直接 push 或 merge 到 `main`
- 变更通过 PR
- 每个 PR 必须 close 一个 issue
- agent 写 issue/comment/PR/review comment 时必须声明 `Written by an agent on behalf of <user>.`
- 持续工作开始用 catchup，较大变更先 plan，PR 前 review，结束用 report

当前本地状态与这些规则有张力：本地 `main` 已领先远端，且工作树有大量未提交变更。下一步如果要正式整理，应先建分支、拆分提交、补 issue/PR 记录。

## 软件设计评价

### 已经比较清楚的部分

- `Analyze` 和 `Advise` 分层明确。
- 参数决策带 provenance/rationale，适合科研软件解释性需求。
- `CalculationIntent` 是一个好抽象，把用户目标集中传入流水线。
- `generate/qe.py` 把 code-specific 输出限制在生成层，方向正确。
- `ml/loader.py` 的 graceful fallback 适合可选重依赖场景。

### 当前需要注意的部分

- 文档落后于源码，尤其 README 和 architecture 中旧 `advisors/` 描述。
- `pseudo.metadata.PseudoSelection` 和 `advise.types.PseudoSelection` 同名不同义，容易混淆。
- `ml/kpoints/*` 多个文件缺少统一代码风格，例如 import 顺序、debug print、长行、非 slots dataclass 约定不一致。
- `test_parse_upf.py` 依赖 gitignored `local_data`，不够可移植。
- `write_qe_inputs()` 返回 dict 类型标注是 `dict[str, Path]`，但实际 `missing_pp` 是 list，类型不准确。
- `README.md` 的 quick start 仍引用不存在的 `goldilocks_core.advisors` 和 `ModelSpec`。

## 推荐的后续整理顺序

1. 更新 README 和 `docs/architecture.md`，让它们匹配当前 `advise/analyse/generate` 架构。
2. 给 `advise` 和 `generate` 增加 focused tests，覆盖 `build_qe_parameter_set()` 与 `write_qe_inputs()`。
3. 清理 `ml/kpoints/*` 的 style 和 optional dependency 边界。
4. 消除或重命名重复的 `PseudoSelection`。
5. 让 UPF 解析测试使用 bundled 或 synthetic fixtures，避免依赖 `local_data`。
6. 在正式 PR 前从当前 dirty `main` 分支拆出 feature branch，并补 issue 追踪。
