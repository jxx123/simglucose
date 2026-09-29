# simglucose 串行与多进程一致性：问题、修复和完整验证报告

[English version](serial-process-report.en.md)

**报告日期：2026-09-30。状态：本地修复与验证完成，尚未推送或发布 issue/PR。**

## 1. 摘要

本次调查比较了相同配置、显式随机种子、同一运行环境下的串行与多进程批量仿真，检查返回数据、CSV 完整性和重复运行。

主要结论：

1. 已测原版数值轨迹一致；没有复现求解器或随机序列导致的串并行数值差异。
2. 已复现同名患者实验覆盖 CSV、创建共享目录的竞争，以及 pathos 工作进程在批次结束后留存。
3. 修复后完整测试套件 **27 项通过**；24 配置矩阵 **268 次精确比较通过**；追加的 6 名患者各 10 天实验 **60 次精确比较通过**。
4. 10 天实验中，串行平均 **174.10 秒**，4 进程平均 **65.55 秒**，本机约 **2.66 倍加速**。这比较的是修复版的两种执行模式，不是修复前后的性能提升。

## 2. 版本、环境与并行边界

| 项目 | 记录 |
| --- | --- |
| 上游 | [jxx123/simglucose](https://github.com/jxx123/simglucose) |
| 贡献者 fork | [FantasticLYW/simglucose_issue](https://github.com/FantasticLYW/simglucose_issue) |
| 原始基线 | `a6f6777586c860457a2a1e6cce6d0bd3fd33d1cc`，包版本 0.2.11 |
| 修复提交 | `afe5682b03a866d25a37e2ea57961368b0e508dc` |
| 10 天验证提交 | `7e9f3fe` |
| 本地分支 | `fix/batch-consistency` |
| 操作系统 / CPU | Windows 10 build 19045 / AMD Ryzen 7 4800H，8 核 16 逻辑处理器 |
| Python | 3.11.15 |
| 数值依赖 | NumPy 1.26.4、SciPy 1.11.4、pandas 2.3.3 |
| 多进程依赖 | pathos 0.3.5、multiprocess 0.70.19、dill 0.4.1 |
| 其他主要依赖 | matplotlib 3.8.4、gym 0.9.4、gymnasium 0.29.1 |

完整环境见 [依赖锁定文件](evidence/batch-consistency/requirements.txt)。

一个 `SimObj` 表示一个患者、设备、控制器、餐食场景和时长构成的实验。`batch_sim(..., parallel=False)` 依次运行实验，`parallel=True` 将不同实验分配给不同进程。每个实验内部的时间步仍依次推进；本次没有实现单实验时间步并行。只运行一个实验时，多进程不会把该实验拆开，还会增加启动和传输开销。

## 3. 已确认问题与根因

### 3.1 同名患者的不同实验覆盖输出

原版 `SimObj.save_results()` 只用患者名生成 CSV 路径。同一批次的两个场景使用相同患者和目录时，虽然返回两个不同 DataFrame，目录中仅有一个 `adolescent#001.csv`。串行模式后写覆盖先写；多进程模式不同工作者写同一个目标，没有独立实验文件。

项目原有高级教程就使用了同名患者、多场景、共用目录的组合。最小复现与原版/修复日志见第 7、9 节。

### 3.2 输出目录创建竞争

原版先判断 `isdir`，再调用 `makedirs`。两个工作者可能都判断目录不存在，其中一个先创建后，另一个抛出 `FileExistsError`。确定性交错回归在真实文件系统复现 WinError 183；这不表示每次普通并发运行都必然触发。

### 3.3 工作进程没有随批次释放

在已安装的 pathos 0.3.5 中，进程池上下文退出不执行清理。原版成功返回和普通 I/O 异常返回后，均观测到工作进程仍存活。默认池会缓存复用，不能将此描述为每次默认调用都会无限新增进程。

## 4. 修复与兼容性

生产改动仅涉及 [sim_engine.py](../simglucose/simulation/sim_engine.py)：

- 父进程根据规范化的真实输出路径统一分配文件名。唯一目标保留 `patient.csv`；重复目标按输入顺序分配 `patient__N.csv`，并避开批次其他任务的原始文件名。
- 用 `os.makedirs(path, exist_ok=True)` 容忍其他工作者已创建目录；权限、文件占位等真实错误继续传播。
- 每批次使用唯一 ID 的进程池，在 `finally` 中显式清理，避免清理其他调用者的共享缓存池。代价是每个并行批次都需要启动工作进程。
- `batch_sim(sim_instances, parallel=False)` 签名和结果顺序保持不变。`sim()` 与 `SimObj.save_results()` 增加可选 `filename` 参数。
- 唯一目标仍无参数调用保存函数；自定义子类 `path=None` 的非文件系统保存也保留该行为。若自定义保存函数参与重名文件任务，需要接受可选 `filename`。

高级教程添加 Windows 主入口保护，并为两个场景使用不同目录。新增回归和独立验证脚本；没有修改生理模型、控制器默认值、奖励、Gym reset 或传感器共享方式。

兼容边界：重复执行同一批次会更新相同文件；独立程序并发运行应使用各自目录。串行推进调用者对象，多进程推进副本，本报告保证返回/保存数据一致，不保证父进程对象状态相同。

## 5. 24 小时矩阵与完整回归

矩阵为 3 名患者（各年龄组 `#001`）× 2 种餐食场景 × 2 种控制器 × 2 组种子，共 24 种配置。

- 场景：自定义 `(7,45),(12,70),(18,80)`，以及 RandomScenario。
- 控制器：BB 和 PID；PID 为 `P=0.001,I=0.00001,D=0.001,target=140`。
- 患者初态/传感器/场景种子：`(1,2,3)`、`(11,12,13)`；患者启用随机初态。
- 起点 2018-01-01，每次 24 小时；三轮串行与双进程，共 144 次矩阵仿真，加 4 次重名文件案例，总计 148 次。
- 使用 `assert_frame_equal(..., check_exact=True)` 比较索引、列、dtype、NaN 和所有值，CSV 读取使用 `float_precision='round_trip'`。每个并行批次验证两个实际非父进程 PID。

| 检查 | 原版 | 最终修复版 |
| --- | --- | --- |
| 完整 pytest | 15 passed，580.50 秒 | 27 passed，594 warnings，610.12 秒 |
| 数值及保存矩阵 | 267 次比较、2 个文件完整性错误；数值一致 | 268 次比较、0 错误 |
| 新增专项回归 | 文件覆盖、目录竞争、进程留存均有失败证据 | 12 passed，80.16 秒 |
| 致命 lint：E9/F63/F7/F82 | 不作为本次修复结论 | 0 |

修复后多一次矩阵比较，来自恢复保存的第二份重名实验文件。专项回归还覆盖路径别名、生成名冲突、reset 后复用、重复批次、输入顺序、高层 UI 统计与 CSV、正常/异常清理和自定义保存兼容。独立审查发现的 `path=None` 兼容退化已经通过先失败后修复的两项测试验证，见 [审查记录](evidence/batch-consistency/review.md)。

原版完整套件限制为两个工作进程；最终完整套件使用默认池配置，设置 Agg 并限制 OpenBLAS/OMP 线程为 1。因此完整套件耗时不用于修复前后性能比较。

## 6. 六名患者各 10 天：耗时和结果

患者顺序为 `child#001`、`child#002`、`adolescent#001`、`adolescent#002`、`adult#001`、`adult#002`。使用 BB、Dexcom、Insulet 和每日随机餐食；第 `i` 名患者的三个种子为 `(100+i,200+i,300+i)`，`i=0…5`。每种模式重新构造对象。

时间范围为 2018-01-01 至 2018-01-11，每 3 分钟一条，含初始记录每人 4,801 行、7 列，10 个日期均有非零餐食。两轮分别先串行后并行、先并行后串行；各批次不重叠。并行使用 4 个实际子进程，OpenBLAS/OMP/MKL 线程数为 1。累计 24 次仿真、240 患者天、24 份 CSV。

| 轮次 | 串行（秒） | 4 进程（秒） | 加速比 |
| --- | ---: | ---: | ---: |
| 1 | 172.35 | 63.92 | 2.70× |
| 2 | 175.86 | 67.18 | 2.62× |
| 平均 | 174.10 | 65.55 | 2.66× |

计时包含 `batch_sim()` 的仿真、CSV 保存、进程启动、返回数据传输及清理，排除模块导入、对象构造和结果比较。两轮总串行时间除以总并行时间得到约 2.66 倍，平均耗时减少 62.35%。这是两轮本机测量，不是统计意义上的普遍加速保证。

**60 次精确比较全部通过：**24 次内存与 CSV、12 次串并行内存、12 次串并行 CSV、12 次跨轮重复性比较。`BG, CGM, CHO, insulin, LBGI, HBGI, Risk` 最大绝对误差均为 0。12 对对应 CSV 的 SHA256 也全部相等，即字节相同。详见 [JSON](evidence/ten-day-benchmark/report.json) 与 [运行日志](evidence/ten-day-benchmark/run.log)。

## 7. 最小文件覆盖复现

在安装了目标版本的仓库中，把以下完整代码保存为 `reproduce.py`，运行 `python reproduce.py`。该代码已分别对原版与修复版执行；两个不同餐食任务应产生两份文件。

```python
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from simglucose.actuator.pump import InsulinPump
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation import sim_engine as engine


def task(path, meal):
    env = T1DSimEnv(
        T1DPatient.withName('adolescent#001', random_init_bg=True, seed=1),
        CGMSensor.withName('Dexcom', seed=2),
        InsulinPump.withName('Insulet'),
        CustomScenario(datetime(2018, 1, 1), [(1, meal)]),
    )
    return engine.SimObj(env, BBController(), timedelta(hours=3),
                         animate=False, path=path)


if __name__ == '__main__':
    # Bound resources while still exercising the real process backend.
    engine.Pool = partial(engine.Pool, nodes=2)
    for parallel in (False, True):
        with TemporaryDirectory() as directory:
            frames = engine.batch_sim(
                [task(directory, 20), task(directory, 50)], parallel=parallel)
            files = sorted(p.name for p in Path(directory).glob('*.csv'))
            print('parallel:', parallel,
                  'different trajectories:', not frames[0].equals(frames[1]),
                  'saved:', files)

```

原版两种模式都显示 `different trajectories: True`，但只有 `adolescent#001.csv`；修复后都保留 `adolescent#001__1.csv` 与 `adolescent#001__2.csv`。完整 issue 草稿见 [文件问题](issues/batch-output-collisions.md) 与 [进程池问题](issues/batch-pool-lifecycle.md)。

## 8. 从隔离环境复跑

以下命令在修复分支仓库根目录执行，要求已安装 uv。两个输出目录都必须不存在；重复运行时换新目录。

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r docs/evidence/batch-consistency/requirements.txt
uv pip install --python .venv/Scripts/python.exe --no-deps -e .
$env:MPLBACKEND = 'Agg'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
.venv/Scripts/python.exe -m pytest tests/test_batch_consistency.py -q
.venv/Scripts/python.exe -m pytest -q -o faulthandler_timeout=240
.venv/Scripts/python.exe examples/verify_batch_consistency.py --output-dir results/new-matrix
.venv/Scripts/python.exe examples/benchmark_ten_days.py --output-dir results/new-ten-day
.venv/Scripts/python.exe -m flake8 simglucose tests examples --count --select=E9,F63,F7,F82 --show-source --statistics
```

要复现原版问题，另取基线 checkout，使用相同依赖安装该 checkout 后运行第 7 节。不要在修复分支只改报告里的版本字符串来代替真正切换被测代码。批量验证脚本也可通过 `PYTHONPATH` 指向独立基线 checkout，原有完整测试须在该 checkout 中执行；详细原版命令见 [原始验证记录](serial-parallel-validation.md)。

## 9. 证据、可追溯性和限制

- [原版矩阵](evidence/batch-consistency/baseline-matrix.json)、[修复矩阵](evidence/batch-consistency/fixed-matrix.json)、[原版完整套件](evidence/batch-consistency/baseline-full-pytest.log)、[最终完整套件](evidence/batch-consistency/final-full-pytest.log)、[12 项回归](evidence/batch-consistency/final-regressions.log)。
- [原版文件复现](evidence/batch-consistency/issue-baseline-files.log)、[修复文件复现](evidence/batch-consistency/issue-fixed-files.log)、[原版进程池复现](evidence/batch-consistency/issue-baseline-pool.log)、[修复进程池复现](evidence/batch-consistency/issue-fixed-pool.log)。
- [测试代码](../tests/test_batch_consistency.py)、[矩阵脚本](../examples/verify_batch_consistency.py)、[10 天脚本](../examples/benchmark_ten_days.py)、[验收核对](evidence/batch-consistency/completion-audit.md)。
- 原始 CSV 在本地 `results/ten-day-benchmark-20260930/`，被 Git 忽略；脚本可重建。归档文本日志仅去除行尾空白。

被测引擎 SHA256：`76f561f41a34ebd0b0f9c53e61b04d887f8d376beb0059fc27fe5699f0d65c6d`。初期矩阵在修复提交前执行，因此其 Git HEAD 字段仍指向基线，实际代码由源码哈希区分；10 天实验已在修复提交上执行。核心代码在两次验证之间未变化。

早期同时运行默认 16 工作者套件与未及时清理独立池的验证器，发生 Windows 内存/页面文件不足；这些中断不算通过。之后约束资源并重跑，最终完整套件成功。日志包含依赖弃用及 Agg 提示；缺少可选 rllab 时相应测试直接返回，27 项通过不代表 rllab 训练已运行。

没有验证全部 30 名患者、任意自定义控制器、无种子独立运行、跨平台/依赖版本逐位一致、所有工作进程数、跨程序同目录锁、进程意外崩溃恢复。2.66 倍只对应第 6 节配置。当前代码没有跨批次历史文件保留，也没有承诺父对象状态一致。

## 10. 如何用于开源贡献

英文报告可作为技术附件；issue 使用短复现和预期/实际行为，PR 使用修复摘要和测试证据。建议先反馈保存问题，将进程池生命周期作为独立问题说明；当前本地修复提交同时包含两类改动，若维护者要求拆分，应另行拆分并复测。

用户操作步骤见 [fork 后续流程](contributing-next-steps.zh-CN.md)，英文 PR 正文见 [PR 草稿](pull-request-batch-consistency.md)。报告完成不代表已提交上游或获得维护者认可。
