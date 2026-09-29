# 串行与多进程仿真一致性 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task after the user reviews the plan. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 在相同实验配置、显式随机种子及相同运行环境下，使串行和多进程批量仿真的返回数据与保存数据一致，并交付有复现证据的 GitHub issue 草稿及必要修复。

**Architecture:** 沿用 `SimObj → sim → batch_sim` 和现有 pathos 进程池。先用独立实验配置复现差异，分别检查数值、任务映射和文件保存；只修改证据指向的根因。生产代码的具体修复方案在复现后补充到本计划，不能把源码风险当作已经发生的数值错误。

**Tech Stack:** Python、NumPy、SciPy、pandas、pathos/multiprocess/dill、pytest；当前主机为 Windows。

**Spec:** 用户本次要求及下文“验收约定”；基线提交 `a6f6777586c860457a2a1e6cce6d0bd3fd33d1cc`。

## 验收约定

- 一致性适用于相同 Python 和依赖版本、相同配置、相同显式种子和确定性控制器。`seed=None` 不承诺跨独立运行一致。
- 返回结果按输入任务顺序逐一对应；比较完整时间索引、列、形状、NaN 位置及所有数值列：`BG, CGM, CHO, insulin, LBGI, HBGI, Risk`。
- 同一环境优先要求 `assert_frame_equal(..., check_exact=True)`。若确有浮点差异，记录首个差异、最大绝对/相对误差及根因；未经说明不能放宽容差使测试通过。不承诺不同平台/依赖间逐位一致。
- 保存数据必须能唯一对应每次实验，不能因同名患者、不同场景或并发创建目录而丢失/混合结果。CSV 以解析后内容比较，考虑十进制序列化精度；不要求 PNG、耗时和 PID 一致。
- 必须实际使用多个工作进程，记录 PID。缺少 pathos 后退回串行不能作为并行验证通过。
- 完成标准包括可复现失败、根因证据、必要修复、回归结果、复现命令及英文 issue 草稿。没有复现的风险必须明确标注，不能写成已确认缺陷。

## Global Constraints

- 保持生理模型方程、设备参数、控制器默认行为和奖励定义不变，除非证据表明它们是本任务根因并重新记录理由。
- 优先保持现有公开接口及唯一患者的保存路径兼容；修复若涉及文件命名或输入对象副作用，应明确说明兼容影响。
- 依赖使用隔离环境；优先以仓库 CI 已列出的 Python 3.11 建立基线，记录实际完整版本。当前系统默认 Python 为 3.13，不能据此宣称仓库支持版本已验证。
- 批量测试 `animate=False`、`MPLBACKEND=Agg`；Windows 复现脚本必须有 `if __name__ == '__main__':` 入口。
- 不扩大为线程池、分布式训练、GPU、全库重构或性能优化项目；耗时作为辅助证据，不设置任意加速比门槛。
- issue 只生成可审阅草稿；实际发布须有用户明确授权。阶段状态和证据持续写入本文件及验证报告。

## 已知源码事实与待验证假设

1. `simulation/sim_engine.py`：串行为列表推导，并行为 `ProcessPool.map`；单个仿真内部按时间顺序推进。
2. 现有 `tests/test_sim_engine.py::TestSimEngine::test_batch_sim` 比较两个固定种子任务的 DataFrame，但未验证保存文件完整性、实际进程数和完整实验矩阵。
3. `save_results()` 使用患者名作为文件名；`advanced_tutorial.py` 两个同名患者场景共用路径，存在覆盖风险。实际影响须复现。
4. `save_results()` 先判断目录再创建目录，存在并发竞争窗口；需要确定性复现并核对真实进程运行。
5. `user_interface.simulate()` 为多个环境传入同一个传感器对象；需检查 reset、随机状态和序列化的实际影响，尚不能称为数值不一致根因。
6. 现有 Gym reset 测试明确允许连续 episode 不同。本任务不能为了重复结果而改变该语义。

## Review Focus

- 同名患者、不同餐食或种子保存到同一目录：任务 2 验证两份数据都可追溯，任务 3 固定回归行为。
- 输出根目录不存在：任务 2 验证并发目录创建，任务 3 验证修复后首次运行成功。
- 连续创建/使用进程池与重复运行：任务 2/3 验证不会挂起、复用错误随机状态或读取旧结果。
- 返回 DataFrame、CSV 与输入对象状态的差异：任务 2 分开记录；只按已约定的输出契约判定一致性。
- 未安装 pathos 或单进程回退：任务 1/3 以工作进程证据识别，不能伪报并行成功。

## 文件责任

| 文件 | 责任 |
| --- | --- |
| `examples/verify_batch_consistency.py`（新增） | 独立复现与结果对比入口，固定配置、种子、任务 ID，输出诊断证据 |
| `tests/test_batch_consistency.py`（新增） | 无图形交互、临时目录隔离的回归测试 |
| `simglucose/simulation/sim_engine.py`（按根因修改） | 批量调度与结果保存 |
| `simglucose/simulation/user_interface.py`（仅证据需要时修改） | 高层入口的对象隔离与结果整合 |
| `examples/advanced_tutorial.py`（按修复需要修改） | 展示安全的多场景批量运行方法 |
| `docs/serial-parallel-validation.md`（新增） | 环境、基线失败、根因、修复后验证及限制 |
| `docs/issues/serial-parallel-consistency.md`（新增） | 可提交的英文 issue 草稿 |

### Task 1: 建立可重现基线

**Interfaces:** 输入为上述 Git 基线和当前主机；输出为验证报告中的完整环境、测试命令及结果记录。

- [x] 检查工作区和已有工作树，在合适的独立分支/工作树执行；保留现有用户文件。
- [x] 建立隔离 Python 环境，安装项目与测试依赖，记录 Python、OS、CPU、numpy/scipy/pandas/pathos/multiprocess/dill/gym/gymnasium 版本。
- [x] 检查现有测试的交互、动画与写盘行为，设置无图形后端后运行相关基线测试；保存失败原文，依赖问题与目标缺陷分开记录。
- [x] 以两个独立短任务验证 Windows 多进程启动，记录子进程 PID；配置导入失败或回退串行时不得标记通过。
- [x] 验收：报告能够从干净环境重建一次串行及真实多进程运行，不修改生产逻辑。

### Task 2: 复现与根因调查

**Interfaces:** 新增脚本提供 `python examples/verify_batch_consistency.py --output-dir results/consistency`；成功比较退出 0，发现不一致或运行失败退出非 0，并输出报告。任务 3 复用相同案例参数。

- [x] 定义任务 ID、患者、场景、患者初态种子、传感器种子、餐食种子、控制器、起始时间和仿真时长；每种执行模式从同一配置重新构造对象。
- [x] 主矩阵：三个年龄组各一名患者，固定餐食与随机餐食、BB 与 PID 两种控制器；使用 24 小时仿真以覆盖随机餐食，固定起始时间 `2018-01-01 00:00:00`，种子组 `(1,2,3)` 和 `(11,12,13)`，每组重复三次。固定餐食为 `(7,45),(12,70),(18,80)`；PID 参数采用仓库示例并在报告中记录。
- [x] 快速回归子集采用 3 小时固定餐食 `(1,20),(2,15)`，避免无餐食短运行掩盖差异；先以唯一输出目录隔离数值问题。
- [x] 独立复现同名患者两个不同场景共用输出路径，以及首次创建共享输出目录；记录返回数据正确但文件丢失的情况。
- [x] 比较复建对象、reset 后复用对象、连续批次，并检查高层 `simulate` 的传感器共享是否有实际影响。
- [x] 差异报告记录首个不同任务/时间/列、最大误差、文件清单、实际工作 PID 与异常堆栈；追踪至随机状态、对象序列化、目录创建或文件命名等具体边界。
- [x] 在本计划补充已确认根因、所需公开接口/文件行为与精确修复测试，再进入任务 3。数值若始终一致，应如实将 issue 聚焦于已复现的其他输出问题。

### Task 3: 最小修复及回归

**Interfaces:** 保留 `batch_sim(sim_instances, parallel=False)` 的基本调用与结果列表契约；如确需扩展，先在任务 2 的设计记录中给出确切签名及兼容说明。

- [x] 为每个已确认根因新增独立测试，明确期望结果；在未修复基线上运行，确认失败来自目标问题而非环境。
- [x] 在证据指向的模块实施最小修复，每个根因分别验证，避免同时改变模型或随机序列。
- [x] 执行 `python -m pytest tests/test_batch_consistency.py -q`，要求数值完全相等、任务顺序正确、落盘数据完整及 Windows 子进程真实执行。
- [x] 重新运行任务 2 的矩阵，并在新目录复测三次；保存可追溯的机器结果和简短人工解读。
- [x] 按改动范围运行既有仿真、seed/reset 和 UI 测试；全量检查参考 CI 的 `pytest` 及 `flake8 --select=E9,F63,F7,F82`，交互/环境失败单独记录，不能宣称全部通过。
- [x] 更新示例的必要使用说明，审查 diff 中是否引入多余改动、输出兼容破坏或未经验证的保证。

### Task 4: 形成 issue 可提交材料

**Interfaces:** 输入为已验证案例及修复 diff；输出为验证报告、英文 issue 草稿、用户可阅读的交付说明。

- [x] 报告记录基线 SHA、完整依赖、执行命令、失败前后对照、矩阵通过数量、跳过/失败原因和已知限制。
- [x] issue 包含准确标题、受影响版本/平台、最小可运行代码、预期与实际行为、错误日志或数据差异、根因说明和建议修复。
- [x] 对独立根因评估是否应拆成多个 issue；标题不得笼统声称全部并行数值错误。
- [x] 从干净进程按 issue 中的命令复核：原始版本复现问题，修复版本通过对应回归；所有引用与命令指向实际存在的文件。
- [x] 最终交付：最小代码 diff、回归测试、验证报告、可提交 issue 草稿；明确实际测试的平台，不把本地 Windows 验证描述为 Linux/macOS 已通过。
- [x] 仅当上述交付及验收完成，才把长期目标标为完成。

## 当前状态

2026-09-30：任务 1–4 已完成，改动位于 managed worktree `batch-consistency`、分支 `fix/batch-consistency`。

- 隔离 Python 3.11.15 环境和完整依赖记录已建立；原版完整套件 15 项通过。
- 确认并修复同名文件覆盖、目录创建竞争和进程池未释放；未复现数值轨迹差异。
- 最终新增回归 12 项通过，完整套件 27 项通过；24 配置、148 次仿真、268 次精确比较全部通过。
- 独立审查提出的自定义 `path=None` 保存兼容问题已完成失败复现及修复验证。
- 验证报告、原始失败/通过证据、独立复现脚本及两份英文 issue 草稿已保存。issue 索引位于原计划指定的 `docs/issues/serial-parallel-consistency.md`；未发布。
- 资源不足的早期中断运行如实记录；原版套件采用双进程，最终修复套件使用默认池配置。已测试范围限定于报告所列 Windows 环境。

## 复现后的修复设计

- 已观察到失败：同名患者两个不同场景仅保存一个 CSV（串行及真实双进程均失败）；模拟另一进程先创建目录时抛出 `FileExistsError`；重复批次同样丢失文件。
- 已观察到通过：三个年龄组的三轮短仿真，串行与双进程所有返回列和解析后的 CSV 完全相等；高层 UI 返回统计量与患者 CSV 相等。最终完整 24 小时矩阵也已通过精确比较。
- 修复接口：`SimObj.save_results(self, filename=None)` 与 `sim(sim_object, filename=None)` 增加可选输出文件名；未提供时保留原有患者名 CSV 行为。
- `batch_sim(sim_instances, parallel=False)` 保持现有签名，先将输入序列物化，在父进程统一计算文件名。以规范化真实目录和患者名判断冲突；唯一目标保留旧名，重复目标依输入顺序分配 `patient__1.csv`、`patient__2.csv` 等，跳过批次中已有的自然文件名。文件名不依赖工作进程 PID、完成顺序或磁盘上是否已有文件。
- 使用 `os.makedirs(path, exist_ok=True)` 修复目录检查与创建之间的竞争，保留真实权限/文件类型错误。
- 兼容边界：一次独立批次内保证每个任务输出唯一；相同批次再次写向同一目录会更新相同文件。独立程序若并发使用同一输出目录，仍应使用各自的运行目录。串行会推进调用者持有的对象，进程模式使用对象副本；结果比较使用返回 DataFrame，不承诺父进程对象状态一致。
- 必须通过的测试：`test_same_patient_different_scenarios_are_both_saved`、`test_save_results_accepts_directory_created_by_another_worker`、`test_reset_and_reuse_keep_duplicate_outputs_stable`、`test_alias_paths_and_generated_names_do_not_collide`；同时保留独立目录旧文件名、顺序和高层 UI 的正向验证。

- 额外确认的生命周期根因：pathos 0.3.5 上下文退出不清理池。每批次分配唯一池 ID，并在 `finally` 中显式 `clear()`；成功及异常清理均有回归。每批次需要重新启动工作进程。
- 自定义 `save_results()` 且 `path=None` 的任务跳过 CSV 文件名分配，保留原来的无参数调用方式；两种模式均已测试。
- 最终证据与复现命令见 `docs/serial-parallel-validation.md`，完成核对见 `docs/evidence/batch-consistency/completion-audit.md`。
