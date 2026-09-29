# 串行与多进程一致性验证

基线：`a6f6777586c860457a2a1e6cce6d0bd3fd33d1cc`，包版本 0.2.11。
工作分支：`fix/batch-consistency`。本文只报告实际在 Windows 上执行的结果。

## 已证实的问题

1. 两个同名患者的不同场景共享输出目录时，返回两个不同的 DataFrame，但只留下一个 CSV。串行、双进程均已复现。
2. 目录创建有检查与创建之间的竞争窗口。确定性交错测试使另一个工作者先创建目录，原始实现抛出 `FileExistsError`（WinError 183）。这不是对任意调度都必现的断言。
3. pathos 0.3.5 的进程池上下文退出不执行清理。原始 `batch_sim` 在正常和异常返回后均留有工作进程；其默认池会缓存复用，不应描述成每次默认调用都会无限新增进程。

初始原版回归记录见 `docs/evidence/batch-consistency/`。数值一致性与文件完整性分别判断，未将推测的随机状态问题列为确认缺陷。

## 数值验证范围

- `adolescent#001`、`adult#001`、`child#001`，代表三个年龄组各一名患者；不是全部 30 名患者。
- 自定义餐食 `(7,45),(12,70),(18,80)`，以及固定种子的 `RandomScenario`。
- BB 和 PID；PID 参数为 `P=0.001,I=0.00001,D=0.001,target=140`。
- `(患者初态,传感器,场景)` 种子为 `(1,2,3)` 和 `(11,12,13)`，启用随机患者初始血糖。
- 固定起点 `2018-01-01 00:00:00`，每次模拟 24 小时，24 种配置，各执行三轮串行及双进程，共 144 个矩阵仿真。
- 比较 Time 索引、七个结果列、dtype 和 NaN 位置，使用 `check_exact=True`；CSV 读取使用 `float_precision='round_trip'`。
- 验证器记录父/子 PID、包版本、源文件 SHA256、运行时间和差异。要求每个并行批次观察到两个非父进程 PID，拒绝把串行回退算作并行成功。
- 另外验证同名场景文件、三小时非空餐食、reset 后重复使用对象、高层 UI 返回统计量与患者 CSV。

## 环境与复现命令

Python 3.11.15，Windows 10 build 19045，16 个逻辑 CPU。完整测试依赖锁定在
`docs/evidence/batch-consistency/requirements.txt`；不是全局 Python 环境。

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r docs/evidence/batch-consistency/requirements.txt
uv pip install --python .venv/Scripts/python.exe --no-deps -e .
$env:MPLBACKEND = 'Agg'
.venv/Scripts/python.exe -m pytest tests/test_batch_consistency.py -q
.venv/Scripts/python.exe examples/verify_batch_consistency.py --output-dir results/new-check
.venv/Scripts/python.exe -m flake8 simglucose tests examples --count --select=E9,F63,F7,F82 --show-source --statistics
```

验证器输出目录必须是新的，以免旧文件掩盖漏写。验证器仅限制真实 pathos 池为两个工作进程并在批次结束时清理，不替换模型、ODE 求解器、控制器或随机数。

完整原有套件的资源受限运行命令如下，只将池大小设为 2，保留全部测试和实际 Agg 绘图：

```powershell
$env:MPLBACKEND = 'Agg'
.venv/Scripts/python.exe -u -c "from functools import partial; from simglucose.simulation import sim_engine as e; import pytest; e.Pool = partial(e.Pool, nodes=2); raise SystemExit(pytest.main(['-q', '-o', 'faulthandler_timeout=180']))"
```

原版证据由未修改的原始 checkout 执行。新增脚本通过 `PYTHONPATH` 指向原始 checkout；完整原有套件直接在原始 checkout 的工作目录运行。JSON 中 `engine_sha256` 区分被测实现，避免把修复后的运行误记为原始版本。

## 修复及兼容性

- 父进程分配输出文件名：唯一目标保留 `patient.csv`；同批次重复目标按输入次序获得 `patient__N.csv`，并预留其他患者的原始文件名。
- 对目录路径进行真实路径和大小写规范化，避免同一目录的不同写法绕过冲突判断。
- `os.makedirs(..., exist_ok=True)` 容忍另一进程先建目录；文件占位、权限等真实错误仍传播。
- 每批次拥有唯一 ID 的池，并在 `finally` 中显式清理；代价为每批次重新启动进程。
- `sim()` 和 `SimObj.save_results()` 支持可选 `filename`，原调用仍有效。覆盖 `save_results()` 的自定义子类若要参与重名批次，需接受这个可选参数。
- 自定义保存子类使用 `path=None` 时，批处理保留原来无参数调用其 `save_results()` 的行为；这不表示基类新增了不保存结果的模式。
- 同一批次再次写入同一目录会更新相同文件。不同程序并发运行须使用不同运行目录；本改动不提供跨程序锁或全目录历史版本管理。
- 串行会推进调用者对象，进程模式推进对象副本；一致性契约适用于返回和保存的数据，使用者应读返回的 DataFrame。
- 未修改患者模型、默认控制器、奖励函数、Gym reset 语义或传感器共享方式。

## 最终验证结果（2026-09-30）

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 原版完整套件 | 15 passed，580.50 秒 | [日志](evidence/batch-consistency/baseline-full-pytest.log) |
| 原版矩阵 | 267 次比较，2 个文件完整性错误；数值比较全部一致 | [JSON](evidence/batch-consistency/baseline-matrix.json) |
| 最终新增回归 | 12 passed，80.16 秒 | [日志](evidence/batch-consistency/final-regressions.log) |
| 最终完整套件 | 27 passed，594 warnings，610.12 秒 | [日志](evidence/batch-consistency/final-full-pytest.log) |
| 最终矩阵 | 24 种配置，含重复文件案例共 148 次仿真，268 次精确比较，0 错误 | [JSON](evidence/batch-consistency/fixed-matrix.json) |
| 致命 lint 检查 | E9/F63/F7/F82：0 | [日志](evidence/batch-consistency/flake8.log) |

最终完整套件在修复工作树运行，未替换默认池大小，仅限制数值库线程数：

```powershell
$env:MPLBACKEND = 'Agg'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
.venv/Scripts/python.exe -u -m pytest -q -o faulthandler_timeout=240
```

最终矩阵中的源文件 SHA256 已与当前 `sim_engine.py` 核对相同：
`76f561f41a34ebd0b0f9c53e61b04d887f8d376beb0059fc27fe5699f0d65c6d`。
验证时修复尚未提交，JSON 的 Git 提交字段仍指向基线；实际被测实现以该源码哈希为准。
串行批次 PID 均为父进程，并行批次均有两个不同的非父进程 PID。

两份 issue 草稿中的 Python 示例均按原文在原版与修复版执行，分别证实文件覆盖及工作进程留存的前后变化。Windows 高级教程执行成功。独立审查发现的自定义保存兼容问题已经以两项先失败、后通过的测试修复，详见 [审查记录](evidence/batch-consistency/review.md) 和 [交付核对](evidence/batch-consistency/completion-audit.md)。归档日志仅去除行尾空白，原始日志保存在工作树 `results/` 中。

## 运行限制与未扩大的结论

最初同时运行默认 16 工作者的完整套件与未及时释放独立测试池的验证器，触发了 Windows 内存/页面文件不足。已确认并结束这两棵测试进程，原始日志保留；这些中断运行不计为通过，也不能单独归因于模拟器的数值逻辑。后续矩阵及原版套件限制为两个工作进程，并在验证器每个批次结束时释放池；最终修复版完整套件使用默认池配置并成功结束。精确比较矩阵仍只覆盖双进程，未穷举工作进程数量。

没有承诺 `seed=None` 的独立运行相等，没有验证跨操作系统/跨依赖版本逐位一致，也没有以耗时声称加速比。现有 rllab 测试在缺少可选 rllab 时直接返回，pytest 的通过不代表训练后端已运行。依赖弃用警告和 Agg 非交互提示将如实保留。

Issue 草稿见 `docs/issues/batch-output-collisions.md` 与 `docs/issues/batch-pool-lifecycle.md`；尚未发布。
