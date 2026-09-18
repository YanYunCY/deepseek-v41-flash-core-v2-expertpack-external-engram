# DeepSeek-V4.1-Flash：Core v2 + ExpertPack + 外置 Engram

本仓库保存基于官方 `deepseek-ai/DeepSeek-V4.1-Flash` 权重转换的社区部署产物，以及配套的自定义 llama.cpp/ROCm 运行时。当前发布路线为 **Core v2 + DSEXP2 ExpertPack + DSEGRAM1 外置 Engram**。这里没有重新训练模型；Core 文件结构、ExpertPack 读取和 Engram loader 都是本项目维护的实现，不能把它当作 DeepSeek 官方 GGUF，也不能直接用未修改的上游 llama.cpp 加载。

发布版本为 `20260916-external-engram`，在 ModelScope DSW 的 AMD gfx942、ROCm 7.2.3 实例上完成文件、编译、部署、短请求、并发、16K 与 64K 长输入验收。上游模型见 [deepseek-ai/DeepSeek-V4.1-Flash](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash)。感谢 DeepSeek、llama.cpp 社区及 JigSawPT Expert Streaming 的工作。模型权重使用必须遵守上游许可证；运行时和第三方组件许可证保留在源码包中。

> 2026-09-16 验收范围：64K 实际输入已通过；实际百万输入尚无通过结论。已分配 1M context 不等于已经完成 1M token 输入验收。

## 文件结构、大小与精度

|部分|内容|字节数|
|---|---|---:|
|Core|`core/DeepSeek-V4.1-Flash-Core-ExternalEngram-BF16.gguf`，888 个 tensor|10,274,108,096|
|ExpertPack|`expertpack/expertpack.v2.manifest` 与 49 个数据文件；40 层 × 384 routed experts × 3 矩阵|288,777,830,400|
|Engram|`engram/v1/engram.v1.manifest.json` 与 104 个数据文件，即 52 对权重/缩放分块|202,758,032,400|
|部署|`deployment/20260916/` 中的 release、部署器、已合并源码和逐文件 SHA256|约 36 MB 源码及小文件|
|证据|`evidence/20260916/` 的报告、响应、日志和验证记录|见仓库文件列表|

Core 与两类数据合计 **501,809,970,896 字节，约 501.81 GB / 467.35 GiB**，未计下载临时文件和构建空间。新磁盘建议预留至少约 560 GiB 可用空间；实际容量仍应以文件系统、下载余量和构建目录为准。

Core 名称中的 `BF16` 是历史命名，**不表示所有 tensor 都是 BF16**。Core 的实际 tensor 类型为 Q8_0 × 330、BF16 × 65、F32 × 489、I32 × 1、I64 × 3。它保留 attention、router、shared expert 与 Engram 必需的辅助/哈希 tensor，但不重复写入 routed expert aggregate 和巨型 Engram 表。ExpertPack 使用 MXFP4 路径；外置 Engram 每行包含 256 B FP8 E4M3 权重和 8 B E8M0 scale。缓存配置只改变驻留和读取策略，不改变上述文件精度。模型页自动解析出的参数量或精度标签可能只针对某个 Core/分片，不能代表本外置布局的完整模型规模。

仓库还保留历史的 11 片整体 GGUF 与恢复材料，它们属于另一条部署路线。当前 `release.json` 只选择 Core v2、49 个 ExpertPack、104 个 Engram sidecar、两个 manifest 和当前源码；不要把旧整块 GGUF 与当前 Core/清单混用。

## 已验证平台与文件完整性

验收机器为 AMD `gfx942`，显存 196288 MiB（约 191.69 GiB），容器主存上限 200 GiB，CPU 配额/亲和性 23。完整 Core、153 个数据文件和两个清单共 156 项 SHA256 已通过；部署器还核对 DSEXP2 的 49 个连续 pack id、Engram 层 1/14 的连续行覆盖，并生成确定性的 `DSEGRAM1` index。运行时没有静默填零兜底。

一次从当前模型库恢复到可服务状态的完整流程，包括文件恢复、完整校验、源码恢复、HIP 编译、health 与真实聊天 smoke，记录为 **17 分 45 秒**。ModelScope Gallery 的验收 Notebook 已运行 **6 个 cell 且全部通过**。这些时间和状态证明当前版本可恢复，不承诺其他磁盘、网络、ROCm 版本或 GPU 上具有相同耗时。

构建目标固定为 `gfx942`：`GGML_HIP=ON`、`CMAKE_HIP_ARCHITECTURES=gfx942`、Release、`LLAMA_BUILD_UI=OFF`、`LLAMA_USE_PREBUILT_UI=OFF` 与 `LLAMA_BUILD_APP=OFF`，仅构建 `llama-server`。关闭 UI 避免 npm 构建和预构建 UI 下载；API 服务不依赖嵌入式 Web UI。

## 一键下载、编译和启动

`deployment/20260916/deploy.py` 是标准部署入口，验收 Notebook 固定并校验它及发布描述符。该入口只下载当前 release 所列资产，模型始终留在 `--data-root`，不会复制第二套约 500 GB 权重到构建目录；`prepare` 首次进行完整 SHA256，`start` 只复用已准备状态、收据和构建，不调用 ModelScope 或重复下载。

Linux 主机需要 Python 3.10+、适配 gfx942 的 ROCm/HIP、CMake、C/C++ 编译器和 ModelScope SDK。服务默认仅监听 `127.0.0.1:48241`。在新实例上可运行：

```bash
python3 -m pip install modelscope
python3 - <<'PY'
from pathlib import Path
import shutil, subprocess
from modelscope.hub.file_download import model_file_download
root = Path('/root/dsv41')
root.mkdir(parents=True, exist_ok=True)
source = model_file_download(
    model_id='Yanyunawa/DeepSeek-V4.1-Flash-MXFP4-GGUF',
    file_path='deployment/20260916/deploy.py')
launcher = root / 'deploy.py'
shutil.copy2(source, launcher)
subprocess.run(['python3', str(launcher), 'all', '--root', str(root)], check=True)
PY
```

已有 ModelScope snapshot 或完整下载目录时，`--data-root` 应指向同时包含 `core/`、`expertpack/`、`engram/` 和发布小文件的目录，且路径不能含空白或使用符号链接：

```bash
python3 /root/dsv41/deploy.py all --root /root/dsv41 \
  --data-root /root/ms-cache/models/Yanyunawa--DeepSeek-V4.1-Flash-MXFP4-GGUF/snapshots/master

# 已 prepare 后的离线日常重启，不会重新下载或编译
python3 /root/dsv41/deploy.py start --root /root/dsv41
python3 /root/dsv41/deploy.py status --root /root/dsv41
python3 /root/dsv41/deploy.py stop --root /root/dsv41
```

启动器会检查 `/health`，再调用 `/v1/chat/completions`，要求 HTTP 200、非空内容和 `finish_reason=stop`。日志在 `root/state/server.log`，收据、进程身份、原始 smoke 响应也在 `root/state/`。`stop` 只终止 PID 与 `/proc` 启动时间均匹配的启动器子进程；端口被占用或文件状态变化会失败而非覆盖现有状态。`--rehash` 可强制重算完整 SHA256。

### 接入本地 Windows DSH

模型服务和 DSH 连接是两个独立步骤。`deploy.py` 只负责让模型 API
监听 `127.0.0.1:48241`，不会自动创建适配器或 SSH 隧道。新建实例也不会
自动继承上一个实例的 `/root/dsv41/dsh-integration` 目录。

模型服务启动后，先下载 DSH 运行文件（不要只下载
`deployment/20260916/deploy-tuned.py`）：

```bash
python3 - <<'PY'
from pathlib import Path
from modelscope.hub.file_download import model_file_download

repo = "Yanyunawa/DeepSeek-V4.1-Flash-MXFP4-GGUF"
out = Path("/root/dsv41/dsh-integration")
out.mkdir(parents=True, exist_ok=True)
for name in ("start-dsh.py", "dsh_api_adapter.py", "remote-connect.py", "encoding.py"):
    source = Path(model_file_download(
        model_id=repo, file_path=f"deployment/dsh/{name}"))
    (out / name).write_bytes(source.read_bytes())
PY
python3 /root/dsv41/dsh-integration/start-dsh.py
```

启动器需要实例中已有的
`/mnt/workspace/.dsv41-connection/dmit_ed25519` 和 `known_hosts`；密钥不会
上传模型库。它会在 `48242` 启动官方 V4.1 encoder/parser 适配器，并把它
通过受限反向 SSH 隧道接到 Windows DSH 的
`http://127.0.0.1:48241/v1`。Windows 端的 `DSV41 DSH Tunnel` 计划任务
负责登录后启动和断线重连。

Windows 端先验证：

```powershell
Invoke-RestMethod http://127.0.0.1:48241/v1/models
```

能够返回 `deepseek-v4.1-flash-local` 后，再关闭旧 DSH 会话并新开终端运行
`dsh`。如果看到 `start-dsh.py: No such file or directory`，说明本次实例
只下载了模型部署器，尚未下载上述四个 DSH 文件。

已验证的本地路径包括文字对话、思考配置和函数工具续轮；图像输入、负载下取消
请求以及云端专属功能不属于这个本地适配器的承诺范围。

若已把全部必要模型文件、源码包、源码哈希清单和 `release.json` 下载到 `/root/model`，并保持库内相对路径，可直接一键校验、编译和启动：

```bash
python3 /root/model/deployment/20260916/deploy.py all \
  --root /root/dsv41 --data-root /root/model \
  --release /root/model/deployment/20260916/release.json
```

全部文件和编译依赖齐备时，此方式无需再次下载。标准部署器的 SHA256 为 `b340a9615fdb6134a084130e2bf02e6336adde18b6f2f55d82833f2e3cea60ce`。

## 多用户并发

多槽服务使用可选的 `deployment/20260916/deploy-parallel.py`。它与同一 `release.json` 兼容，默认单槽，使用 `--parallel` 指定槽数。下载后可核对其 SHA256：

```text
829e9300f4a147147cb4835adcb743b7cd6a2d86364da6bfd1c9a704b918bd10
```

下载并启动时可先保留标准部署器完成 `prepare`，再取得此可选入口：

```bash
python3 - <<'PY'
from pathlib import Path
import shutil
from modelscope.hub.file_download import model_file_download
target = Path('/root/dsv41/deploy-parallel.py')
source = model_file_download(
    model_id='Yanyunawa/DeepSeek-V4.1-Flash-MXFP4-GGUF',
    file_path='deployment/20260916/deploy-parallel.py')
shutil.copy2(source, target)
PY
sha256sum /root/dsv41/deploy-parallel.py

# 先停止同一 root 受控的服务，再启动两槽或四槽服务
python3 /root/dsv41/deploy.py stop --root /root/dsv41
python3 /root/dsv41/deploy-parallel.py start --root /root/dsv41 \
  --gpu-cache-gib 160 --host-cache-gib 96 --batch 512 \
  --context 16384 --parallel 4
```

完整参数、边界和恢复流程见 [`deployment/20260916/README-parallel.md`](deployment/20260916/README-parallel.md)。两个入口共用已准备的模型与构建；并行入口使用独立文件名，验收 Notebook 仍固定标准部署器的版本。

## 已测配置：按使用场景选择

下表来自同一 gfx942 实例。短请求缓存扫描使用 4096 context、单槽、batch/ubatch=128、6 个短输入任务，每题 `max_tokens=384`；“热合并 decode”是两轮热请求的实际输出 token/实际 decode 秒。首轮没有清空操作系统页缓存，因此不称物理冷盘。显存列采用服务 readiness 后约 1 秒开始的采样，不包含启动窗口；它是 launcher 定义的近似观察起点，不是 GPU 硬件精确峰值。

|GPU 专家缓存|Host L2|首轮 decode|热合并 decode|服务就绪后显存峰值|适用判断|
|---:|---:|---:|---:|---:|:---|
|172 GiB|128 GiB|13.976 tok/s|17.560 tok/s|180.897 GiB|主存占用较高，不作默认|
|172 GiB|96 GiB|12.895 tok/s|17.066 tok/s|180.897 GiB|短请求单槽基线|
|176 GiB|96 GiB|13.885 tok/s|17.168 tok/s|185.100 GiB|增益有限，余量更小|
|180 GiB|96 GiB|14.188 tok/s|17.874 tok/s|189.302 GiB|本轮约快 4.7%，不作通用默认|

建议的短请求配置是 **172/96 GiB、batch/ubatch 128、context 4096、parallel 1、I/O 线程 4、CPU 线程 8、O_DIRECT、reasoning off**。这是该任务集的稳妥基线，不是全负载最优承诺。本轮 172/96 的热合并为 17.066 tok/s；历史证据中另有约 17.747 tok/s 的不同批次记录，两者不可混算。180/96 虽有约 4.7% 的本轮优势，readiness 后采样已到 189.302 GiB，不能被称作默认值。

交互并发推荐 **parallel 2、总 context 8192（每槽 4096）、GPU/Host 160/96 GiB、batch 512**；吞吐并发推荐 **parallel 4、总 context 16384（每槽 4096）、GPU/Host 160/96 GiB、batch 512**。固定 `max_tokens=128` 下，两轮有效输出/批次墙钟为：

|并发|总 context / 每槽|两轮有效输出 / 墙钟|有效吞吐|all-valid TTFT p50|stop / length|
|---:|---:|---:|---:|---:|:---:|
|1|4096 / 4096|188 / 35.100 s|5.356 tok/s|11.035 s|2 / 0|
|2|8192 / 4096|424 / 52.016 s|8.151 tok/s|16.223 s|3 / 1|
|4|16384 / 4096|802 / 78.635 s|10.199 tok/s|22.168 s|7 / 1|
|8|32768 / 4096|1413 / 126.764 s|11.147 tok/s|36.366 s|16 / 0|

`length` 表示完整请求达到输出上限，不等于自然完成；2/4 并发不能称为全部正常 stop。TTFT p50 按每档两轮的全部有效请求汇总，是固定提示和当前机器下的实验观测，**不是 SLA**。8 并发已经实测，16 个请求均 stop，但相对于 4 并发的总吞吐增幅较小，同时全请求 TTFT p50/p95 为 36.366/55.850 s、总响应 p50/p95 为 61.502/72.379 s，所以不强行作为推荐配置。并发显存的 readiness 后采样约为 169.144、169.076、169.342、169.509 GiB（np=1/2/4/8）；启动窗口可能受前一服务释放滞后影响，不能用全程峰值推导配置间的显存因果关系。

## 1M 单槽容量与已完成长输入

单槽 `context=1048576` 已通过容量分配、16K 与 64K 实际输入检索。日志均确认 `n_slots=1, n_ctx_slot=1048576`；实际长度由服务端 usage 验证，而不是以 context 参数代替。早期 16K/64K 使用 `structured-values-v1` 的确定性键值探针；最新 batch 2048 的 16K 使用独立 `independent-sha256-v2` needle。两种探针都只验证其三处位置检索和流式生成，不能等同于完整长文理解或长上下文能力评测。

|实际输入探针|GPU/Host 缓存|batch|usage prompt / 输出|TTFT / 总耗时|prefill / decode|GPU / RSS 峰值*|结果|
|---|---:|---:|---:|---:|---:|---:|:---|
|16K structured-values-v1|160/64 GiB|128|16370 / 30|290.775 s / 292.814 s|56.305 / 14.716 tok/s|175.813 / 75.431 GiB|stop，三键值命中|
|16K structured-values-v1|160/64 GiB|512|16370 / 30|170.077 s / 172.058 s|96.271 / 15.149 tok/s|181.787 / 78.246 GiB|stop，三键值命中|
|16K structured-values-v1|160/64 GiB|1024|16370 / 30|154.700 s / 156.694 s|105.843 / 15.050 tok/s|189.374 / 82.055 GiB|stop，三键值命中|
|16K independent-sha256-v2|136/96 GiB|2048|16382 / 42|164.128 s / 167.339 s|99.835 / 13.083 tok/s|180.727 / 121.668 GiB|stop，独立 needle 全中|
|64K structured-values-v1|152/96 GiB|512|65518 / 30|1197.825 s / 1200.648 s|54.705 / 10.634 tok/s|174.075 / 110.279 GiB|stop，三键值命中|

64K 的 raw token needle 位置为 6585、32784、58983；batch 2048 的 16K 独立 needle raw `/tokenize` 为 16378、位置为 1679、8215、14748，服务端 usage 为 16382，均已命中且 `quality_pass=true`。上表显存均使用 readiness 后的同一采样口径，16K batch 128/512/1024 的复核值与原记录相同。

\* GPU 采用服务就绪后采样峰值；RSS 是该次进程全程的主存峰值。

这证明“1M 单槽容量可分配，16K 与 64K 实际输入可通过”，不证明已经处理接近 1M token。百万输入验证使用：GPU/Host 136/96 GiB、batch 2048、`ctx=1048576`、单槽、目标输入 1,048,000 token、`max_tokens=128`、timeout 19,800 s。请求已被服务接受，日志输入为 1,048,006 token；截至本次文档更新尚无完整响应。运行与结束状态见长输入证据快照，不计入已通过结果。

长输入工具位于 `tools/extended-benchmark.py`；它使用服务端 `/tokenize` 计数、为 chat 模板与输出预留预算、记录 needle token 偏移，并保存 SSE、usage、timings、原始文本和失败原因。

## 证据、文章与限制

本轮证据位于 `evidence/20260916/extended/`，包括 `cache-and-concurrency.zip`、`cache-and-concurrency-summary.json`、`notebook-and-long16k.zip` 和 `long-context-progress.zip/json`。长输入快照区分运行中与结束状态，后台测试结束后另外保存 `long-context-final.zip/json`；旧的缓存扫描、恢复和部署证据继续保留，供追溯历史任务、参数和失败记录。并行部署说明位于 `deployment/20260916/README-parallel.md`，扩展基准工具位于 `tools/extended-benchmark.py`。原始响应、manifest、日志和源码哈希应一起保存，速度表不能脱离其输入、finish reason 和缓存状态传播。

- 部署文章：[ModelScope AI 开发者活动部署篇](https://modelscope.cn/learn/436737)
- 可复现实验：[ModelScope Gallery — amd-deepseek-v41-deployment](https://modelscope.cn/gallery/Yanyunawa/amd-deepseek-v41-deployment)
- 性能文章：[专家缓存、长上下文与并发的实测取舍](https://modelscope.cn/learn/436741)

已验证的范围是文件完整性、当前运行时的可恢复部署、所列中文/代码/算术/检索任务、固定短请求、1/2/4/8 并发批次、16K 与 64K 实际长输入。它不等价于官方 logits 对齐、完整能力评测、多模态验证，或任何提示、上下文、硬件上恒定的 tok/s。实际百万输入尚不属于已通过验收的范围。请依据上游许可证和实际硬件余量使用本转换产物。
