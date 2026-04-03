# CLAUDE.md - 工作准则与系统状态

## 所有回复必须用中文

## 偷懒判定（出现任何一项=偷懒，必须在结尾认错）

以下任何一项出现就是偷懒行为：
1. cron巡检总耗时<5分钟
2. TodoWrite checklist<10条
3. 全链路验证少于5条帧（必须5种不同场景：flee帧、fight帧、counterattack命中帧、navigate帧、policy explore帧，每条从obs→action→Java→MC效果逐层追踪）
4. 出现问题没有深入检查分析+系统修复
5. 出现"可能""大概""会""应该"这种模糊词
6. 没有反复通读过去5分钟的全部log，没有一个字一个字读
7. 把事情拖延（说"等下轮""观察一下"）
8. 赖到用户身上，要用户来做
9. 生存没有任何进步却说没问题
10. 不看真实表现只看数据

## 核心原则：初心勿忘，使命必达

1. **定期记录进度** - 每次重要操作后记录结果到本文件，不遗漏
2. **定期检查任务目标** - 回顾当前目标，确保没有偏离方向
3. **牢记初心勿忘使命** - RL micro policy 要真正学会 value，偶尔参与控制，逐步成长
4. **不要偏离主题** - 一切工作围绕让 bot 通过 RL 真正成长
5. **能自己解决的就不要骚扰用户** - 所有问题自己解决，所有操作自己执行
6. **能多想一步的绝对不偷懒** - 不浅尝辄止，不说"这样就行了"
7. **反复剖析 + 实际验证测试** - 改了代码必须验证效果，不能只改不验
8. **打破砂锅问到底** - 遇到问题深入分析 log 搞明白发生了什么，而不只是统计数据
9. **从根源上解决问题** - 从系统上优化方案，而不是只在补丁上打补丁

## 强制执行流程（巡检/修复都必须遵守）

### 第一步：记录开始时间
```
开始时间: HH:MM:SS
```
用`date '+%H:%M:%S'`记录。结束时也记录。**如果总耗时<5分钟，就是在偷懒——没有认真思考和验证。**

### 第二步：列出plan（用TodoWrite）
开始工作前必须用TodoWrite列出具体要做的事。不是"检查系统"这种空话，是具体到：
- "取一条rl_obs JSON，检查threats有几个字段"
- "追踪fight模式下dyaw的计算路径：dx/dz坐标系→atan2→normalizeAngle→最终值"
- "在MC里验证attack=1时怪物hp是否下降"

### 第三步：逐项执行，完成一项用TodoWrite打勾
每完成一项就标记完成。不能跳过，不能批量打勾。做完一项再做下一项。

### 第四步：记录结束时间，检查耗时
```
结束时间: HH:MM:SS
总耗时: X分钟
```
如果<5分钟 → 你在偷懒，重新做。5分钟都不到说明没有深入看代码、没有追踪数据流、没有验证实际效果。

## 历史教训（绝不能重犯）

- **绝不能因为难就disable功能** — skill imitation数据差→直接禁用→模型永远学不会。PPO出问题→直接暂停→不诊断根因。正确做法：控制变量找根因，修复后重新启用。
- **监控不能只查数字，必须查实际数据样本** — hindsight训练跑了11轮，goalDist全是0没人发现。loss下降≠模型在学对的东西。每次新功能上线后必须取5条数据样本检查字段值。
- **不能被动等用户发现问题** — 所有重大发现都是用户问出来的。要主动审查。
- **出问题不能一刀切关掉** — 必须做控制变量测试，逐项排查，找到具体根因。
- **改了代码必须验证实际数据** — 代码逻辑对≠运行时数据对。每次改数据收集/处理代码后，下一轮cron必须打印实际数据样本验证。
- **每次cron必须问自己**："最近改的代码，实际产生的数据我看过样本了吗？有没有全是0/null的字段？"
- **50轮巡检说"能打能跑"实际打空气跑反方向** — attack按了≠打到了。bot在移动≠跑对方向。fight的dyaw坐标系错了50轮没发现。flee 67%朝怪物跑50轮没发现。9项审查说"全部通过"实际一项都没真正验证。

## 端到端验证（强制执行，不做不算审查）

**每次审查/巡检任何组件，必须拿一条真实数据从头走到尾。没走过=没审查。**

### 数据流验证checklist（每层都要打印实际值确认）

1. **Java层**：取一条原始rl_obs JSON → 逐字段看：threats发了几个字段？dx/dz是什么坐标系（player-relative）？有hp吗？有type吗？
2. **Node层**：formatObs之后变成什么？有没有字段被丢弃？坐标系有没有被错误转换？
3. **写入层**：combat_demos里的action=1时，MC里实际发生了什么？attack=1→准心对准了吗→cooldown满了吗→伤害判定了吗？fwd=1→方向对吗→实际在远离怪物吗？
4. **训练层**：parse_obs_to_tensors的tensor和Node端的值一一对应吗？
5. **推理层**：模型输出→actionToRlAction→Java端→MC客户端，最终效果是什么？

### "能打"的判据（硬标准）
- ❌ "attack帧占比高" — 按了不等于打到
- ✅ "怪物hp下降了" 或 "怪物被击杀了"
- 如果数据里看不到怪物hp → 这本身就是bug，先修

### "能跑"的判据（硬标准）
- ❌ "bot在移动" — 移动不等于跑对方向
- ✅ "threat距离持续增大且增大比例>70%"
- 如果50/50 → 不是"勉强能跑"，是"跑不掉"，深入查方向计算

### "审查通过"的判据（硬标准）
- ❌ "函数存在" / "代码有这行" — 存在不等于正确
- ✅ "拿了真实数据样本走了一遍，每层值都打印出来确认了"

## 项目架构

```
LLM Planner (qwen3.5-9b, 战略) → Baritone 命令
    ↓
Tactical Layer (MicroController, 每tick 20Hz)
    ↓ flee/fight/continue/hide
Micro Policy (PyTorch 1.27M params, GPU RTX 4060, inference ~40ms)
    ↓ 当前策略：15% 概率参与控制（从5%提升，模型已学到fwd=76%/atk=14%）
    ↓ 目标：继续提升参与率到30%+
Java rl_action → MC 客户端
```

## 关键文件

- `src/agent/micro_controller.js` - RL tick 控制器
  - 15% policy explore (flee/fight 时，从5%提升)
  - `_collectCombatDemo()` 收集 flee/fight 真实按键到 combat_demos.jsonl
  - getLatestAction 在 sendObs 之前调用（修复了 hasAction=false 问题）
  - `_lastPolicyExplore` 标记避免收集 policy 自己的输出
- `src/agent/policy_client.js` - Node TCP client, fire-and-forget + cached action
- `rl_policy/server.py` - Policy server (多线程 threading, port 7860, 支持同时推理+训练)
- `rl_policy/server_v2.py` - 启动脚本，加载 v2 checkpoint
- `rl_policy/model.py` - PyTorch MicroPolicy (TerrainCNN+ThreatAttention+LSTM)
- `rl_policy/trainer.py` - 训练逻辑 (tactical + imitation + PPO)
  - `load_combat_demo_samples()` 读取 combat_demos.jsonl
  - `load_skill_demo_samples()` 读取 skill_demos.jsonl（注意：skill demos 里 action 大部分是 None/全 false，数据质量差）
- `rl_policy/train_offline.py` - 离线训练脚本（不走 server TCP，直接加载模型训练）
  - 训练顺序：加载 v2 checkpoint → tactical → skill imitation → combat imitation → 保存 → hot-reload 到 server

## 数据文件

- `bots/GaoBot1/survival_episodes.jsonl` - 130K+ 生存数据（tactical 标签：79% flee）
- `bots/GaoBot1/skill_demos.jsonl` - 157 条（数据质量差：action=None，keyFwd 全 false）
- `bots/GaoBot1/combat_demos.jsonl` - flee/fight 真实按键（escape planner 输出），这是最有价值的 imitation 数据
- `bots/GaoBot1/micro_policy_v2.pt` - 当前 checkpoint

## 启动命令（必须从正确目录）

- Bot: `cd /c/Users/pm/Desktop/mindcraft-0.1.3/mindcraft-0.1.3 && node main.js > bot.log 2> bot.err.log`
- Policy server: `cd /c/Users/pm/Desktop/mindcraft-0.1.3/mindcraft-0.1.3/rl_policy && python server_v2.py`
- 离线训练（必须用nohup，不能用Claude后台任务——会阻塞cron）:
  1. 先提取最新数据: `tail -400000 bots/GaoBot1/combat_demos.jsonl > bots/GaoBot1/combat_demos_recent.jsonl`
  2. 用nohup跑: `cd /c/Users/pm/Desktop/mindcraft-0.1.3/mindcraft-0.1.3/rl_policy && nohup python train_offline.py > train.log 2>&1 &`
  3. 查结果: `cat /c/Users/pm/Desktop/mindcraft-0.1.3/mindcraft-0.1.3/rl_policy/train.log`

## 已知问题

- **skill_demos.jsonl 已废弃**：旧的数据收集方式（freezeMicro=true导致action=None）已废弃。技能现在直接在execute()中主动写action到combat_demos.jsonl（通过writeSkillDemo），绕过freezeMicro问题。dig_shelter记录dig+seal动作，pillar_up记录jump+place动作，hit_and_run记录approach+attack+retreat动作。
- **combat_demos.jsonl 是真正有效的 imitation 数据**：从 escape planner/fight heuristic 收集的真实按键。flee 时 74% forward，数据分布合理。
- **threats obs 没有怪物类型**：只有 dx, dz, dist，模型无法区分 skeleton(远程) 和 zombie(近战)。
- **hit_and_run / swim_to_shore 注册了但从未被调用**（2026-03-29修复：fight模式现已接入hit_and_run）
- **CronCreate在continued session里不触发**：需要杀掉旧session进程才能恢复（2026-03-29已修复）
- **Goal被旧memory.json覆盖**：agent_process.js重启时用load_memory=true读saved goal。解决：必须杀全部node进程（WMIC terminate）+删memory.json才能干净启动新goal。taskkill有时不起作用。
- **LLM memory被搞坏**：qwen3.5-9b把saving_memory prompt当成用户对话回答，生成垃圾内容。
- **setControlState是空函数**：fabric_bridge.js:510 `bot.setControlState = () => {}` 导致所有依赖setControlState的skills无效（jump/sprint/forward）。pillar_up已修复为用rl_action命令。其他skills如果需要setControlState也会失败。
- **pillar_up实际执行的是digUp**：digUp里有pillar逻辑但依赖setControlState(noop)。重写为rl_action后仍然失败(delta=0)——rl_action每tick重置，无法持续jump。加了3次连续失败熔断。
- **pillar_up无限循环bug**：8秒cooldown不够防止重复触发，27次连续触发浪费4分钟并导致死亡。已修复：3次连续失败后禁用直到respawn。
- **LLM respawn后忽略goal中的"dig dirt"**：总是优先找木头。goal已更新为更明确地指示用!digDown挖dirt。
- **badlands地形digDown失败**：bot在峡谷边"reached a drop below"挖0块。需要先移到平地。
- **LLM自己覆盖goal**：qwen3.5-9b发了!goal("简化版")把6步goal缩成1步，然后!endGoal停了。已block !goal和!endGoal命令。
- **Bot重启丢失inventory**：respawn后所有物品都没了。需要重新收集。之前有30dirt+13oak_log白费了。
- **skeleton_horse被误判hostile**：entity.name.includes('skeleton')匹配了skeleton_horse。已修复：NON_HOSTILE排除列表。
- **Baritone flee时不收集combat_demos**：threatDist>9时走Baritone flee路径直接return，跳过了_collectCombatDemo。已修复：在return前从baritone inputs构造action并收集demo+发obs给policy server。

## 技能使用统计（4个builtin技能）

| 技能 | 触发次数 | 成功次数 | 问题 |
|------|---------|---------|------|
| dig_shelter | 4 | 4 | 数据记录到skill_demos但action=None |
| pillar_up | 35+ | 0 | 全部失败。rl_action每tick重置无法持续jump。加了3次失败熔断。需要Java端持续action支持 |
| hit_and_run | 0 | 0 | fight=43次但hit_and_run未触发，需查原因 |
| swim_to_shore | 0 | 0 | 未触发（bot很少掉水） |

## 2026-03-29 重大修改

1. **assessCombat勇气覆盖**: 满血(>=18)+单个近战怪 → dangerous提升为manageable(canWin=true)。之前fist打zombie=safetyMargin~2.9被判dangerous永远flee。
2. **fight模式接入hit_and_run**: TACTICAL.FIGHT执行时优先用hit_and_run技能，带cooldown 5s。
3. **dangerous级别brave fight**: 满血(>=16)+单个近战怪+非远程非爆炸 → dangerous也选fight而不是flee。
4. **LLM prompt加入方块优先级**: BLOCKS排第3（武器>食物>方块>工具>护甲>庇护所），要求保持10+可放方块。
5. **Cron深度监控**: prompt从简单检查改为4步深度分析（诊断→深度思考→推进→对齐检查）。

## 训练记录

| 时间 | tactical loss | imitation loss | combat imitation | combat demos 数量 | Policy fwd% | 备注 |
|------|--------------|----------------|------------------|------------------|-------------|------|
| 2026-03-29 18:05 | 0.89 | 7.55 | N/A | 0 | ~42% | 第一次训练 v2 |
| 2026-03-29 ~20:00 | 0.89 | 4.17 | N/A | 0 | ~42% | 从 v2 继续训练 |
| 2026-03-29 ~21:30 | 0.86 | 3.44 | 4.47 | 25.8K | 67% | 首次combat demos训练 |
| 2026-03-29 23:10 | - | - | - | 30.1K | 67% | 监控确认改善，fight=0次 |
| 2026-03-30 00:40 | **0.39** | **2.28** | **4.15** | 34.2K | 76%(n=37) | fight=43次! tactical loss↓55% |
| 2026-03-30 01:50 | **0.24** | **2.26** | **4.04** | 40.9K | - | tactical↓38%再降，动态参与率15%起步(5-40%自动调) |
| 2026-03-30 02:30 | 0.25 | **1.82** | **3.92** | 46.9K | 62%(n=183) | tactical收敛，skill↓19%，动态rate自动升到17% |
| 2026-03-30 03:40 | 0.24 | 92.9⚠️ | **3.73** | 51.9K | 80%(n=15) | combat↓继续改善，skill爆炸(数据质量差不影响) |
| 2026-03-30 04:20 | **0.20** | 88.1⚠️ | **3.68** | 58K | 68%(n=496) | tactical↓77%从起点！动态rate=21% |
| 2026-03-30 07:10 | 0.20 | 147⚠️ | **3.57** | 63K | - | tactical收敛0.20，combat持续↓ |
| 2026-03-30 08:20 | 0.22 | 跳过 | **3.13** | 69K | 49%(污染中) | 禁用skill后combat↓12%！等验证fwd恢复 |
| 2026-03-30 09:30 | **0.20** | 跳过 | **2.88** | 74K | 58%(恢复中) | combat新低！tactical=0.20 |
| 2026-03-30 11:30 | 0.20 | 跳过 | **2.80** | 79K | 62% | combat持续↓，第10次训练 |
| 2026-03-30 13:00 | 0.20 | 跳过 | **2.70** | 85K | 68%(n=3513) | combat↓40%从起点，fwd恢复68% |
| 2026-03-30 15:10 | 0.20 | 跳过 | **2.58** | 91K | - | combat↓42%从起点，第12次训练 |
| 2026-03-30 16:30 | 0.20 | 跳过 | **2.48** | 96K | - | combat↓44%，第13次，接近100K demos |
| 2026-03-30 18:40 | 0.197 | 跳过 | **2.33** | 101K | - | **100K里程碑！** combat↓48%从起点，第14次 |
| 2026-03-30 20:10 | 0.197 | 跳过 | **2.20** | 108K | - | combat↓51%从起点！第15次 |
| 2026-03-30 21:50 | 0.196 | 跳过 | **2.08** | 113K | 70%(n=10K) | combat↓53%！exploreRate=35%，第16次 |
| 2026-03-30 23:30 | 0.196 | - | frame:**2.65** seq:**2.40** | 123K | - | 首次hindsight+LSTM序列训练！890段，seq<frame说明LSTM有效 |
| 2026-03-31 01:10 | 0.195 | - | frame:**2.37** seq:**2.30** | 159K | 73%(n=3192) | hindsight持续↓，navigate 34K帧流入，max_samples提至200K |
| 2026-03-31 02:30 | 0.195 | - | frame:**1.86** seq:**1.53** | 194K | 74%,atk=8% | 197K帧参与！seq↓33%破2！修复fight不执行bug |
| 2026-03-31 04:00 | 0.195 | - | frame:**1.56** seq:**1.44** | 237K | 75%,atk=**14%** | 第20次！hit_and_run首次触发！SkillDemo在记录 |
| 2026-03-31 06:00 | 0.197 | - | frame:**1.10** seq:**0.96** | 377K | atk=29% | 第23次！seq破1！372K帧+8567段，P1/P2/P3全生效 |
| 2026-03-31 ~12:00 | - | - | 重大架构重构 | 930K | - | 删除Baritone flee，goal重构(方向向量)，PPO重设计(冻结共享层+完整reward+安全门控) |
| 2026-03-31 16:30 | 0.194 | - | frame:**0.86** seq:**0.75** | 951K | 50%(新架构适应中) | 新goal架构首次训练，loss略升(正常) |
| 2026-03-31 16:40 | - | - | 修复 | 984K | - | hit_and_run改用rl_action(不再timeout)，Q-net残留crash修复(fb undefined) |
| 2026-04-02 ~10:45 | 0.194 | - | frame:**1.46** seq:**1.28** | 398K(tail400K) | - | 第36次。上次session完成 |
| 2026-04-02 ~11:40 | 0.194 | - | frame:**1.76**(⚠️↑) seq:进行中 | 398K(tail400K) | - | 第37次。frame loss上升(navigate数据占83%,pitch>45°=14%) |

## 2026-04-02 修复清单（本session）

### Node端修复（已部署，bot已重启生效）
1. **flee maxDyaw分级**: 15→45°(error>90)/30(>45)/15(aligned) — local_escape_planner.js:422
2. **flee error>120°不forward**: 只strafe+转向 — local_escape_planner.js:432
3. **flee continuation dyaw分级**: 同上逻辑 — local_escape_planner.js:627
4. **flee移除anyClose jump**: mob近不无条件跳(69%stuck=airborne) — local_escape_planner.js:451
5. **flee stuck检测排除airborne**: onGround才累加stuckTicks — local_escape_planner.js:467
6. **fight heading gate**: attack=1要求aimAligned(<40°) — micro_controller.js:57
7. **fight EMA加速**: 0.3→0.5 — micro_controller.js:1233
8. **fight approach heading gate**: >60°停止forward只转向 — micro_controller.js:61
9. **fight approach dyaw提速**: ±7.5→±25°/tick — micro_controller.js:1238
10. **fight dpitch加速**: ±15→±45, gain 0.5→0.8 — micro_controller.js:1259

### Java mod修复（已构建部署到mods/，需MC重启）
1. **attackEntity**: setPressed改为interactionManager.attackEntity — CommandHandler.java:601
2. **setSprinting**: 新增player.setSprinting(true) — CommandHandler.java:597

### 验证状态
- fight approach: dist<4从~1%→20% ✅
- fight heading gate: 不再heading=130°打空气 ✅
- flee向mob跑: 21.6%→15% ✅
- flee jump: anyClose jump移除确认 ✅
- **MC未重启: attackEntity+setSprinting未生效** ❌ P0阻塞

## 下一步目标

1. **P0: 等MC重启** → 验证attackEntity(mob hp下降) + setSprinting(sprint>90%)
2. MC重启后: 验证flee gaining从22%提升(sprint修复应带来大幅改善)
3. MC重启后: 验证fight attack命中(dist<3 + heading<40 + pitch<15 + mob hp下降)
4. 训练#37完成后: 检查seq loss，如果ok触发#38(含新修复的数据)
5. 继续提高exploreRate(当前0.25)
7. 增加 threats obs 中的怪物类型信息
