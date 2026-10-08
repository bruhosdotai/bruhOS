# bruhOS 设计文档

**一句话简介**

bruhOS is the skill OS that stands next to the robot.

bruhOS: an operating system that acts like your bruh.

---

**Slogan（可选）**

- Runs like your bruhhh.
- Built like a bruh. Backs you like an OS.
- Not a brain in the cloud. A bruh on the body.

**对外**

简介：bruhOS is a partner OS for robots—it hears you, runs skills, and leaves a trace.

Slogan：Runs like your bruhhh.

**一句话**

TonyPi 上的具身操作系统：输入指令、输出技能；规划与数据层对接开源 Physical AI 栈（NVIDIA / LeRobot / 开源 VLM），控制仍留在机上技能环。执行轨迹可审计，并预留上链。

**定位**

技能操作系统 + 开源大脑适配层。

---

## 1. 为什么要做

缺的不是又一个模型，是 **模型到身体之间的系统**。

开源侧已经有现成大脑：NVIDIA Isaac GR00T、Cosmos、Isaac Lab、Hugging Face LeRobot、openpi、Qwen-VL。它们能规划、能仿真、能出动作 chunk。TonyPi 这一侧仍是散装动作组、云端聊天和不可复现的演示。

两端对不上，会出现三种浪费：

1. 开源 VLA 假设连续关节 / 标准数据集，TonyPi 只有动作组和开合手。
2. 大模型在机上跑不动（树莓派 5），在云上又直接幻想舵机。
3. 一次成功执行没有标准轨迹，既不能评测，也不能进 LeRobot / 仿真回流，更谈不上上链。

bruhOS 存在的理由：**把已开源的大脑接到这台身体上，而不改写那些模型的训练代码。** Runtime 负责观察、技能、恢复和 trace；Brain 可以换 GR00T 规划头、Qwen-VL 或云 API，身体接口不变。

---

## 2. 解决什么问题

| 问题 | 没有 bruhOS | 有 bruhOS |
| --- | --- | --- |
| 开源模型用不上 | 各写各的 demo | 统一技能 JSON + 观察 schema |
| 模型直接控舵机 | 幻觉 / 摔机 | 只调白名单技能 |
| 看不见还在动 | 视频硬塞大模型 | 机上压缩观察 |
| 摔倒即结束 | 任务作废 | 恢复后续跑 |
| 英语语音 | 中英混用 | ASR / TTS / prompt 默认英语 |
| 数据断在终端 | 一段视频 | LeRobot 兼容 trace，可仿真、可哈希、可上链 |

---

## 3. 亮点

- **技能内核**：系统单元是 skill，不是聊天回复。
- **开源大脑可插拔**：同一 Runtime 可接云 API、Qwen-VL、LeRobot 策略服务、未来的 GR00T 规划头。
- **NVIDIA 放对层**：Cosmos / Isaac 用于数据与仿真，不进 50Hz 控制。
- **观察压缩 + 英语优先**。
- **失败是一等公民**。
- **执行即数据**：trace 同时服务评测、开源训练格式和可选上链。

---

## 4. 开源集成原则

1. **控制环不依赖闭源云，也不依赖巨型端侧模型。**
2. **能复用的复用适配器，不复用他们的关节定义。** GR00T 的 action 是全身连续控制；接到 TonyPi 必须经技能层翻译，或只把它当 System 2。
3. **仿真和真机共用同一套 instruction / skill / outcome schema**，这样 Cosmos 生成的失败案例和真机摔倒日志能进同一评测。
4. **许可分清**：NVIDIA GR00T / Cosmos 偏可商用；部分中文数据集是 NC。进产品路径只走可商用许可。

---

## 5. 集成地图

```
                    English command
                           │
                           ▼
              ┌────────────────────────┐
              │  bruhOS Brain          │
              │  planner adapters      │
              │  Qwen-VL / OpenAI API  │
              │  optional: GR00T S2    │
              │  optional: LeRobot svc │
              └────────────┬───────────┘
                           │ skill JSON
                           ▼
              ┌────────────────────────┐
              │  bruhOS Runtime (Pi 5) │
              │  observe · dispatch    │
              │  recover · trace       │
              └────────────┬───────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
      TonyPi body    Trace Writer    Offline stack
      action groups  hash / sinks    Isaac + Cosmos
                                     LeRobot datasets
```

### 5.1 现在就接（v0.1）

| 资源 | 放哪 | 在 bruhOS 里干什么 |
| --- | --- | --- |
| **Hugging Face LeRobot** | 数据 + 评测格式 | trace / 示教导出为 LeRobot Dataset，方便以后换臂、换策略 |
| **Qwen2.5/3-VL**（开源） | Brain | 英语指令 + 观察 JSON → 技能调用；可本地 vLLM |
| **OpenAI-compatible API** | Brain 备路 | 同一 schema，换 endpoint 即可 |
| **Hiwonder/TonyPi** | Body | 动作组、相机、IMU、RPC、语音盒 |
| **YOLO / MediaPipe / OpenCV** | Runtime 感知 | 机上目标、人脸、姿态，禁止把原视频当唯一输入 |

### 5.2 NVIDIA（v0.1 预留接口，v0.2 起真正吃到）

树莓派跑不了这些。它们属于 **电脑侧 / 集群侧**。

| 资源 | 层 | 用法 |
| --- | --- | --- |
| **Isaac Sim + Isaac Lab** | 仿真 | 搭「桌面色带 + 物块 + 盒」数字孪生；先测规划与恢复，再上车 |
| **Isaac Lab-Arena + LeRobot EnvHub** | 评测 | 同一 20 条英语指令在 sim 里回归，防止每次改 Brain 都炸真机 |
| **Isaac GR00T N1.7** | Brain 可选头 | **只用 System 2（推理 / 分解任务）** 产出技能序列，不把 DiT 关节动作直接打到舵机 |
| **Cosmos 3** | 数据 | 生成遮挡、推倒、换位的世界视频 / 状态，补真机拍不够的失败样本 |
| **Isaac Teleop** | 采集（有臂之后） | v0.3 双机时采操作数据；TonyPi 仍走技能遥操作而不是全身遥操作 |
| **Jetson Thor / APXInf** | 未来端侧 | 若换 Orin/Thor 本体，π0.5 / 小 VLA 才考虑上机；TonyPi 世代不作为目标 |

GR00T 适配器约定：

```
GR00T_S2(image, "Put the red one in the box.")
    → ["detect", "walk_toward(red_block)", "grab", "walk_toward(box)", "release"]
    → Runtime 逐个执行
```

禁止：GR00T action chunk → 16 路舵机。那是错误集成。

### 5.3 按需再接

| 资源 | 何时 |
| --- | --- |
| **openpi π0 / π0.5** | 有 NexArm / SO-ARM101 之后，作为臂的 System 1 |
| **SmolVLA** | 臂上轻量语言条件策略；不在 TonyPi 全身跑 |
| **APXInf** | 有 Jetson 时的端侧推理引擎 |
| **AgiBot World / GO-1** | 仅研究对照；NC 许可不进默认产品权重 |

---

## 6. 产品定义

| 项 | 说明 |
| --- | --- |
| 名称 | bruhOS |
| 形态 | Runtime + 可插拔 Brain + 技能包 + Trace + 开源适配器 |
| 首发本体 | TonyPi / TonyPi Pro |
| 语音 | 英语优先 |
| 开源立场 | 站在 LeRobot / NVIDIA / 开源 VLM 之上做身体系统，不从零训基座 |
| 不是 | 端侧巨型 VLA；把控制权交给 Cosmos 或智能合约 |

---

## 7. 运行时能力（不变，作为集成的锚）

观察 JSON、技能白名单（`look_at` `detect` `walk_toward` `follow_line` `grab` `release` `kick` `wave` `stand_up` `stop`）、摔倒恢复、英语短指令，仍是内核。

开源模型必须输出这一套 schema，而不是反过来改 Runtime 去迁就某个 checkpoint 的 action 维。

---

## 8. 数据、仿真与上链

开源栈让「执行即数据」有地方去：

1. **本地 trace**（v0.1 必做）
2. **导出 LeRobot**（v0.1 schema 对齐，v0.2 出包）→ 社区训练工具可直接吃
3. **Cosmos / Isaac 回流**（v0.2）→ 失败案例增强，不替代真机验收
4. **哈希 + 可选 ChainSink**（接口 v0.1，启用 v0.3）→ 上链的是 `trace_root` 和模型/技能包版本哈希，不是视频，不是控制权

控制环永不阻塞在出块或 Cosmos 推理上。`stop` 只走 Runtime。

---

## 9. 模块

```
bruhOS/
  runtime/
  brain/
    adapters/openai.py
    adapters/qwen_vl.py
    adapters/groot_s2.py      # stub in v0.1
    adapters/lerobot_policy.py
  skills/
  speech/                     # EN-first
  trace/                      # writer, hash, LocalSink | IpfsSink | ChainSink
  sim/                        # Isaac Lab task spec, 20-prompt eval
  schemas/                    # obs, skill, trace  (LeRobot-aligned)
  eval/
  configs/                    # tonypi / tonypi_pro
```

---

## 10. 版本

| 版本 | 集成 | 产品 |
| --- | --- | --- |
| **v0.1** | TonyPi + Qwen-VL/API + LeRobot schema + 本地 trace | 英语技能 OS，20 条真机评测 |
| **v0.2** | Isaac Lab 回归；Cosmos 补失败数据；GR00T S2 适配器可选 | 改 Brain 不必每次都上真机 |
| **v0.3** | 双机：TonyPi 技能 + 臂上 π0.5/SmolVLA；ChainSink 可开 | 走、看、说与抓分离 |
| 之后 | 有 Jetson 再谈端侧 VLA | 仍不在 Pi 5 上跑 GR00T 全身动作头 |

---

## 11. 成功标准

1. 换 Brain 适配器（API ↔ Qwen-VL ↔ GR00T S2 stub）不改技能名。
2. 20 条英语真机指令完成率可报；仿真集同一份 prompt。
3. 推倒恢复率单独记账。
4. 默认中文 ASR、未登记技能、GR00T 关节直出舵机 = 测试失败。
5. 每条任务可导出哈希；LeRobot 字段能对上；关掉 Isaac / 链模块不影响 `stop`。

---

## 12. 对外说法

bruhOS exists because open brains have nowhere to stand on a real body. It plugs NVIDIA, LeRobot, and open VLMs into TonyPi as planners and data engines—while the robot itself only runs skills, speaks English, gets back up, and leaves a trace.
