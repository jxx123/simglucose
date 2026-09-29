# Fork 后的下一步：针对本次 simglucose 修复

核对日期：2026-09-30。下列是待执行的发布步骤；报告准备阶段没有推送分支、发布 issue 或创建 PR。

## 1. 你目前在哪一步

截图中的流程是通用流程。你已经完成 fork，本地也已完成克隆、创建修复分支、修改、测试和提交，因此可以直接进入“把分支推送到自己的 fork”。不用重新克隆再手动搬运代码。

| 项目 | 当前状态 |
| --- | --- |
| 原作者仓库 | `jxx123/simglucose`，默认分支 `master` |
| 你的 fork | `FantasticLYW/simglucose_issue`，默认分支 `master` |
| 两个远程 master | 核对时均为 `a6f6777586c860457a2a1e6cce6d0bd3fd33d1cc` |
| 已完成的代码与测试 | `afe5682`：修复；`7e9f3fe`：10 天实测与证据 |
| 要推送的本地分支 | `fix/batch-consistency`，包含上述提交及本次报告 |
| 修复所在目录 | `C:\Users\10047\.codex\worktrees\batch-consistency\simglucose` |
| 当前 origin | **指向原作者仓库**，不是你的 fork |

这些改动在独立工作树，不在最初 `D:\work\ABW\Diabetes_Project\simglucose` 的 `master` 中。请在下面指定的目录操作。

## 2. 推送修复分支到自己的 fork

保留现有 `origin`，另加名为 `fork` 的远程，避免更改其他工作树使用的远程配置。远程名称并不强制必须叫 origin/upstream。

```powershell
Set-Location 'C:\Users\10047\.codex\worktrees\batch-consistency\simglucose'
git status --short
git branch --show-current
git remote -v

# 只在第一次设置时执行；当前尚未添加 fork 远程。
git remote add fork https://github.com/FantasticLYW/simglucose_issue.git

# 将当前已有的修复分支推到你的 fork。
git push -u fork fix/batch-consistency
```

预期分支为 `fix/batch-consistency`，工作区干净。若以后重复操作，已有 `fork` 就跳过 `remote add`。GitHub 登录提示使用你的账户完成认证。这里推送的是修复分支，不是 `master`。

## 3. 反馈问题：issue 与 PR 的区别

- **Issue**：向原作者解释发现了什么、怎样复现、预期与实际结果。
- **Pull Request（PR）**：请求原作者审查并合入你的代码改动。

如果先报告问题，去 [上游 Issues](https://github.com/jxx123/simglucose/issues)，先搜索已打开和已关闭的同类报告（例如 `batch_sim`、`overwrite`、`ProcessPool`）。有相同问题时补充证据；没有时使用 [文件覆盖英文草稿](issues/batch-output-collisions.md)。进程池生命周期有 [单独草稿](issues/batch-pool-lifecycle.md)。不要将问题命名为已证实的“数值不一致”，因为本次原版数值比较已通过。

核对时未登录页面显示 issue 创建受限，不能据此判断你登录后的权限。如果你的账户仍不能新建 issue，可直接创建 PR，将复现和根因写入正文。先发 issue 是建议的沟通顺序，不是创建 PR 的必要条件。只想报告问题，也可以发 issue 后暂不提交 PR。

## 4. 创建指向原作者仓库的 PR

推送成功后，从 [上游 Pull requests](https://github.com/jxx123/simglucose/pulls) 选择 New pull request，再选择 **compare across forks**，填写：

| 字段 | 值 |
| --- | --- |
| base repository | `jxx123/simglucose` |
| base branch | `master` |
| head repository / head fork | `FantasticLYW/simglucose_issue` |
| compare branch | `fix/batch-consistency` |

这个方向表示“把你 fork 中的修复，申请合入原作者 master”。GitHub 官方说明见 [Creating a pull request from a fork](https://docs.github.com/en/pull-requests/how-tos/create-pull-requests/creating-a-pull-request-from-a-fork)。

建议标题：

> Preserve batch simulation outputs and clean up owned process pools

正文使用 [英文 PR 草稿](pull-request-batch-consistency.md)，附上 [英文完整报告](serial-process-report.en.md)。若已经创建 issue，再填入真实 issue 编号进行关联；未创建时无需虚构编号。当前提交同时修复文件保存和进程池生命周期，正文已写明；维护者要求拆分时再分别整理、验证。

在 Files changed 中确认改动范围。代码改动集中在仿真引擎和高级教程，其他主要为测试、脚本和证据文档。完整归档包含较多日志，若维护者不希望合入全部证据，可按反馈精简 PR 的文件范围，将详细报告保留在你的 fork 中。

## 5. 提交之后

查看 CI 状态并回应维护者意见。后续修改仍提交到同一个分支，再执行 `git push fork fix/batch-consistency`，现有 PR 会包含新的提交。上游合并由维护者决定；在自己的 fork 合并不等于贡献已经进入上游。

## 文档用途

- [中文完整报告](serial-process-report.zh-CN.md)：自己审阅和中文交流。
- [English full report](serial-process-report.en.md)：英文技术附件。
- [两个 issue 草稿](issues/serial-parallel-consistency.md)：复现和问题讨论。
- [英文 PR 正文](pull-request-batch-consistency.md)：代码审查说明。

推送前本地文件链接可直接阅读；推送后可以在 fork 的修复分支中打开对应 Markdown，再复制 GitHub 文件地址给维护者。
