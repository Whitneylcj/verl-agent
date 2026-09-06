# ALFWorld Exact-G：实现规范与验证边界

实现基于 `origin/codex/exact-credit@3c52b79f57968b2349781ae80a069b25fa315b5a`。
研究逻辑集中在 `recipe/exact/alfworld_{semantics,adapter,metrics}.py`，原生环境、
manager 与 rollout 只负责传递状态、执行一次动作和排除已结束的采样行。

## 守恒与保证范围

环境目标质量仅来自原有 `R = 10 * won`。reset 固定 K 个过程通道，默认每个
权重为 `1/K`，令 `phi_k(s)=w_k * predicate_k(s)`：

```text
D[q,k] = phi_k(s[q]) - phi_k(s[q-1])
B[k]   = -sum_q D[q,k]
C_R    = R
sum_k(sum_q D[q,k] + B[k]) + C_R = R
```

每个 B 按自己的通道作用域路由，C_R 始终连接每个有效 token。所有通道的
`channel_role=process_verifier`、目标质量为零。这里没有把完整合取成功目标
拆成可独立兑现的奖励，也不要求各动作 credit 之和等于 R。

保证针对**同一护栏、协议、上下文和 horizon 下的原始 on-policy score estimator**。
PPO clipping、KL、off-policy ratio 与重复优化不在这个恒等式的保证之内。
护栏改变可执行转移，不能声称启用护栏前后 rollout 分布相同。

## 官方语义编译

`compile_semantics(game_data)` 读取实际 `.tw-pddl` 的 `pddl_domain` 与
`pddl_problem`，大小写规范化后计算 AST SHA256。支持域的固定哈希为
`decd886f9a13e537453b3d9eb8e391aac64fc434eae47d82e65c21ff50750bd9`。
部署还核验 TextWorld 两个原生 Python 模块的 Git blob 哈希；版本号相同而
源码不同也会停止。manifest 记录源锁，逐轨迹 schema 记录 problem 哈希。

依据：[官方域](https://github.com/alfworld/alfworld/blob/aaba6870f86c5be6a08a491f32a50b906227bc3e/alfworld/data/alfred.pddl)、
[六类完整目标](https://github.com/alfworld/alfworld/blob/aaba6870f86c5be6a08a491f32a50b906227bc3e/alfworld/gen/goal_library.py)、
[TextWorld PDDL 状态与空 Action 前后条件](https://github.com/microsoft/TextWorld/blob/ebae03b2a65440f8baed46a885811719b1b948f2/textworld/envs/pddl/logic/__init__.py)。

- 完整保留 exists/forall、and/or/not、实例相异条件与共同容器；静态事实约束
  grounding，不作为通道。通道来自动态目标叶谓词和直接建立目标的动作前置条件。
- heat 增加 isHot、删除 isCool；cool 反向写入。拿放涉及共享 holdsAny、holds、
  物体位置；照明读取 isToggled、物体与 agent 位置、手持关系，不能用 isOn 代替。
- clean、历史 toggle 的正谓词单调；toggle 的两个条件效果都读取动作前状态。
- 完整 grounded 域用于未来分析，包括当前不可执行动作、负前置条件与条件效果。
  当前动作映射复用 `_valid_actions.mapping` 与 `_valid_commands`；不读取其空的
  `preconditions/postconditions` 来猜效果。
- snapshot 使用 reset/step 已有 `_facts`、`_entity_infos`；将显式负事实规范化为
  闭世界布尔状态。静态事实、原生 won、实际完整状态差分均与编译语义核对。
  不增加环境查询或反事实执行。未知域、THOR、缺失状态或不一致都不能获得证书。

## 动作、提交与可达性证书

```json
{"op":"heat","args":["mug 1","microwave 1"],"commit":true}
```

只允许一份按 op/args/commit 顺序的 JSON，不允许重复键、额外字段或解释文字。
支持原生 navigation、take、put、open/close、clean/heat/cool、toggle、slice、
examine、look/inventory/help。提交只允许 clean、heat、cool、toggle、put 所建立的
一个正物体谓词。成功后才加入保护集合；不开隐式解锁，不保护 agent 位置、
手持槽或容器开关。提交错误实例/位置照常保留，由官方结果承担后果。

后来可能删除保护事实的命令由执行层转换为原生无效命令，每个响应仍消耗
一个正常步；不限制 decoder support，不修改采样概率或训练 log-prob。
条件删除也保守拒绝，即使它此刻不触发。禁用护栏时 JSON 协议相同，commit
不产生保护。模型提示只包含普通环境观察、动作列表、已有上下文和自己成功
提交的保护记录；隐藏目标、facts、通道值、证明内容不进入提示。

reset/step 后对每个 grounded 谓词的可能真值进行同步 powerset 传播：从当前
具体状态出发，保留 identity 和所有未被保护约束禁止的算子，传播至剩余
horizon 或不动点。合并时忘记相关性，因此只会保留额外边。新 commit 不参与
产生本次响应的证书；任意未来策略选择均在包络内，未因清空历史剪断依赖。

解析每个 token **之前**的已解码前缀，只能用已完整出现的 op 和参数缩小
当前动作范围。畸形后缀始终允许产生无效动作。未来包络目前只使用响应前的
状态与保护集合，故在不同参数前缀间共享，可能比进一步分支敏感分析更保守。
只有未来真值集合为当前单值的通道才删除未来 delta 与 closure；当前 delta
还可依据所有合法后缀都无法改变该谓词而删除。

细分前缀要求可验证的 ByteLevel decoder（测试驱动用确定的字符 decoder）；
未完成 UTF-8 后缀或其他 decoder 回退到响应前的宽包络，仍可使用与 decoder
无关的状态不变性证书，不根据完整响应推断此前的解码稳定性。

例如，显式保护 `inReceptacle(apple1,table)` 后，take 被禁；即使 heat/cool
本身不直接删除这条位置关系，分析仍能证明它们需要的 holds(apple1) 永远
不可达，从而冻结 apple1 温度。其他物体、共享资源与尚可解锁动作仍保留。

证书记载 source/state/prefix 哈希、字符前缀长度、span token 起止、剩余
horizon、已有保护、当前/未来通道及每个不变谓词值与证明方法。信用编译
核对来源、覆盖、前状态及观测到的后续状态；后验数据只用于拒绝不成立的
证书，绝不用于重新选边。每个有效 token 恰属一个 span，包括 malformed、
commit、被拒绝命令和结束 token。

提前成功后把过程状态逻辑延拓到 reset 指定的最大 H；固定原子槽位继续存在，
不调用环境或产生延拓 score。批处理中仅对活动 episode 采样，剩余槽位以
副本填充并由 active_masks 排除；分布式整除 padding 仍按原框架处理。

## 公平对照和研究判据

远端使用相同 checkpoint、过程通道、JSON、history、H、采样及优化预算：

| 组别 | 环境变量 |
|---|---|
| 无护栏 T / G | `ALFWORLD_COMMIT_GUARD=False EXACT_MODE=temporal/graph` |
| 有护栏 T / G | `ALFWORLD_COMMIT_GUARD=True EXACT_MODE=temporal/graph` |
| 同信息前缀 baseline | 相同护栏，`EXACT_MODE=prefix_baseline` |
| 旧 planner T，单列 | `ALFWORLD_EXACT_SIGNAL=planner EXACT_MODE=temporal` |

以上分别执行 `bash examples/exact_trainer/run_alfworld.sh`。默认研究配置是
TextWorld + predicates + commit guard + hard graph，H=30，最大 prompt=4096。
旧的通用 PPO YAML 仍默认 planner，避免改变其他使用原生文本协议的入口；
研究 launcher 显式覆盖它。实验名包括 signal 和 guard，manifest 保存完整覆盖项。

本版所证明的是**通道从此前缀起为常量**，可写出完全相同的信息 baseline：

```text
credit_G(u) = R - sum_{k in mutable(prefix_u)} (phi_k(pre_u) - phi_k(initial))
```

代码逐轨迹断言这一等式误差 <= 1e-10，并提供 baseline 训练模式。因而本版
G 与 baseline 的收益在数学上相同，不能宣称发现了超出该 baseline 的独立图
收益。现有图的作用是编译与审计可证的依赖删边。没有全局最小图保证，也
不承诺任意轨迹稀疏、整个梯度稀疏，或方差/成功率必然改善。

## 诊断与验证

`exact/alfworld/*` 与逐轨迹 `alfworld_comparison` 同时记录：

1. delta / closure 的 span 对及 token 加权删边比例，并保留分子分母；固定 H
   的补零 delta 不进入比较分母。
2. 两类被删的非零绝对质量、`|credit_G-credit_T|` 和非零差异 token 比例；
   baseline 等价误差单列。
3. 语义编译和 capture 耗时、既有图编译耗时及训练框架成功率。真实梯度方差
   必须另取采样 checkpoint 的 score，不能用 credit variance 或 grad norm 冒充。

`alfworld_gradient_audit.py` 接受同批 on-policy 原始 score 向量/固定投影与
三种未作 batch 缩放的 token credit，计算轨迹梯度协方差 trace、配对均值差和
标准误。输入契约见模块 docstring；仅记录的 log-prob 标量不能替代其梯度。
它不自动启动模型反向传播，也不声称有限样本均值差证明无偏。

本地只运行 CPU 测试：

```bash
python -m pytest -q tests/recipe/exact/test_alfworld_predicates.py \
  tests/recipe/exact/test_alfworld_credit.py \
  tests/recipe/exact/test_alfworld_prefix_manager.py \
  tests/recipe/exact/test_alfworld_gradient_audit.py
```

覆盖六类官方目标、不同实例/共同容器、冷热交叉写入、条件 toggle、自然单调、
错误提交、拒绝/畸形动作、前缀一致性、逐 token 覆盖、枚举状态空间证书反例
搜索、完整分支期望梯度等价、源篡改、实体重命名、隐藏信息、提前结束与 padding。
守恒、baseline 和 float64 枚举梯度比较均使用 1e-10。

已运行的 7 步合成 clean-and-wrong-lock 例子（实际域与目标、人工小状态空间，
**不是真实 rollout**；字符作 token）删掉 22.53% 的 delta 对、10.78% 的 closure
对和 13.87% 的非零绝对质量；平均绝对 credit 差为 0.04313。被删 delta 非零
质量为 0，非零收益全部来自 closure，baseline 差为 0。不能外推成真实数据收益。

真实环境检查脚本已实现，尚未在本次本地工作中执行：

```bash
python -m recipe.exact.probe_alfworld_exact_g \
  --games-root "$ALFWORLD_DATA/json_2.1.1/train" --per-type 2 --horizon 60 \
  --output /tmp/alfworld-exact-g-probe.json
```

只允许在准备好的 Linux 环境运行。六类各选择至少两个不同未切片任务，每个
运行无护栏成功、有护栏成功和错误放置提交后拒绝/超时三条轨迹（默认 36 条）。
测试驱动可用官方 planner，训练路径不请求 planner。检查状态差分、完整 goal
与 won、一响应一原生调用及终止后零调用，再通过实际信用管线验证守恒和
baseline。报告只保存任务哈希、来源和指标，不保存观察或命令；其中稀疏率
为字符加权诊断，梯度方差明确为 null。仍需远端向量/Ray/模型 smoke、真实
token 稀疏性、score 方差和任务成功率实验，才能作实证结论。

本次本地回归结果为 **167 passed、10 skipped**，Ruff、Python 语法编译、shell
语法与 `git diff --check` 通过。完整 recipe 测试中旧 AppWorld / ALFWorld manager
模块因本机没有 Ray / Gym 环境依赖而排除；新增谓词 manager 测试独立通过。
仓库 `tests/test_protocol.py` 的收集也受 Ray 缺失阻挡，未宣称全仓检查通过。
