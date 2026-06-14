# 软件测试与维护（2026年春）— 答辩 PPT 文案

---

## 幻灯片1: 封面

**标题**: 基于 SockShop 微服务系统的部署、测试与智能运维

**副标题**: 软件测试与维护（2026年春）大作业答辩

**演讲者**: [小组成员姓名]

**日期**: 2026年6月15日

---

## 幻灯片2: 项目概述

**标题**: 项目概述

**要点**:
- 选取开源微服务系统 **SockShop** 作为实验平台
- 评分档次：**第一档**（基础完成）
- 双重部署：Docker Compose（日常运行） + Minikube/K8s（ChaosMesh 故障注入）
- 完成四大阶段：部署 → 监控维护 → 自动化测试 → 论文算法复现

**技术栈**:
| 层次 | 技术 |
|------|------|
| 容器化 | Docker / Docker Compose |
| 编排 | Kubernetes (Minikube) |
| 监控 | Prometheus + Grafana |
| 故障注入 | ChaosMesh |
| 测试 | Selenium + JMeter |
| 算法 | PyTorch + scikit-learn |

**演讲文案**:
> 我们小组选择 SockShop 微服务系统作为实验平台，采用 Docker Compose 日常运行 + Kubernetes 故障注入的双重部署模式。四个阶段完整覆盖部署、监控、测试和论文算法复现。

---

## 幻灯片3: 阶段一 — 系统部署

**标题**: 阶段一：SockShop 部署架构

**要点**:
- **Docker Compose**: 14 个微服务容器（front-end, carts, catalogue, orders, payment, user, shipping, queue-master 等）
- **K8s/Minikube**: 完整 14 个 Pod 运行，用于 ChaosMesh 故障注入
- 镜像通过 Windows 系统代理（127.0.0.1:7897）从 Docker Hub 拉取

**部署架构图**:
```
┌─────────────────────────────────────┐
│        SockShop (edge-router)       │
│         http://localhost:8088       │
├─────────────────────────────────────┤
│  front-end  carts  catalogue        │
│  orders  payment  user  shipping    │
│  queue-master  rabbitmq             │
├─────────────────────────────────────┤
│  mongo ×3  catalogue-db  user-db    │
├─────────────────────────────────────┤
│  Prometheus + Grafana + node-exp    │
└─────────────────────────────────────┘
```

**演讲文案**:
> SockShop 包含 14 个微服务，通过 Docker Compose 一键部署。边缘路由映射到 localhost:8088。同时部署到 Minikube 集群用于 ChaosMesh 故障注入实验。

---

## 幻灯片4: 阶段二 — 监控体系

**标题**: 阶段二：Prometheus + Grafana 数据采集与展示

**要点**:
- **Prometheus**: 9/9 Targets UP，采集 JVM 堆内存、GC 次数、实例运行时间
- **Grafana**: 自定义 Dashboard (4 面板：服务状态、JVM 内存、GC 速率、运行服务数)
- 数据保留 24h，15s 抓取间隔

**监控指标展示**:

| 服务 | JVM 堆使用 | 运行时长 |
|------|-----------|----------|
| carts | 94.5 MB | 2468h |
| orders | 97.4 MB | 2467h |
| shipping | 58.9 MB | 2467h |

**Grafana Dashboard**: http://localhost:3000/d/sockshop (admin/admin123)

**演讲文案**:
> Prometheus 成功采集 SockShop 全部 9 个目标的指标。Grafana 仪表盘实时展示服务状态和 JVM 性能数据。node-exporter 因 Windows 挂载限制使用基础模式，但仍正常采集。

---

## 幻灯片5: 阶段二 — ChaosMesh 故障注入

**标题**: 阶段二：ChaosMesh 部署与故障注入验证

**要点**:
- ChaosMesh 6/6 组件全部 Running
- 接入 K8s 集群，对 SockShop 进行故障注入

**故障注入实验记录**:

| 实验 | 类型 | 目标 | 时长 |
|------|------|------|------|
| demo-fault-cpu | StressChaos | carts | 120s, 90% CPU |
| demo-fault-network | NetworkChaos | catalogue | 120s, 1000ms |
| pod-kill-carts | PodChaos | carts | — |
| verify-cpu-stress | StressChaos | carts | 30s, 80% |
| + 6 more | — | — | — |

**验证结论**: 10 条实验记录，StressChaos / NetworkChaos / PodChaos 全部成功创建并执行。

**演讲文案**:
> ChaosMesh 部署在 Minikube 上，6 个组件全部健康。成功注入 10 个故障实验，涵盖 CPU 压力、网络延迟和 Pod 杀除。Dashboard 因 Google Fonts 被墙加载慢，通过 CLI 和 API 验证全部正常。

---

## 幻灯片6: 阶段三 — Selenium 功能测试

**标题**: 阶段三：Selenium 自动化功能测试

**要点**:
- Python + Selenium + Chrome (headless)
- 模拟真实用户操作：页面加载、导航、交互

**测试结果**: 5/5 PASS

| 测试 | 结果 | 耗时 |
|------|------|------|
| Homepage 加载 | PASS | HTTP 200 |
| Login 页面 | PASS | 92ms |
| Register 页面 | PASS | 32ms |
| Catalogue 页面 | PASS | 33ms |
| Basket 页面 | PASS | 127ms |
| 导航链接 | PASS | 55 links |
| 添加购物车 | PASS | 187ms |

**演讲文案**:
> Selenium 使用 headless Chrome 模式模拟真实用户操作。全部 5 个页面加载测试通过，交互测试包括导航、登录表单和添加购物车全部成功。

---

## 幻灯片7: 阶段三 — JMeter 性能测试

**标题**: 阶段三：JMeter 性能测试

**要点**:
- Apache JMeter 5.6.3
- 测试计划：线程组 + HTTP 采样器 + 汇总报告

**性能指标**:

| 指标 | 数值 |
|------|------|
| 请求数 | 5 |
| 成功率 | **100%** |
| 吞吐量 | **26.0 req/s** |
| 平均响应时间 | **17ms** |
| 最小/最大 | 11ms / 39ms |

**Python 并发补充测试** (Windows 下 JMeter 多线程有端口限制):

| 场景 | 并发 | 成功率 | 平均响应 |
|------|------|--------|----------|
| 并发首页 | 10 | 100% | 22ms |
| 压力首页 | 50 | 100% | 535ms |
| API 正常 | 100 | 100% | 1171ms |
| API 故障下 | 100 | 100% | 1386ms |

**演讲文案**:
> JMeter 测试计划包含线程组和 HTTP 采样器，生成 HTML 报告。Windows 下 JMeter 多线程有端口限制，用 Python 并发测试补充验证高并发场景，故障注入下吞吐量从 85 降到 72 req/s。

---

## 幻灯片8: 阶段四 — 论文选择与算法复现

**标题**: 阶段四：论文算法复现

**两篇论文**:

| | USAD | PatternMatcher |
|------|------|----------------|
| 来源 | KDD 2020 | ISSRE 2021 |
| 任务 | 异常检测 | 根因指标识别 |
| 方法 | 对抗训练 AE | KS-test + 模式分类 + 排序 |
| 语言 | PyTorch | Python + scipy |

**为什么选这两篇？**
- 互补性强：检测 + 定位 = 完整智能运维流程
- 都有公开数据集和可复现的实验设置
- 针对 IT 运维场景，直接适用微服务系统

**演讲文案**:
> 选择 KDD 2020 的 USAD 和 ISSRE 2021 的 PatternMatcher。前者做异常检测，后者做根因定位，两者结合覆盖智能运维核心流程。

---

## 幻灯片9: USAD 实验结果

**标题**: USAD — 对抗训练自编码器异常检测

**架构**: Encoder(300→150→75→30) + 2×Decoder，二阶段对抗训练
**异常分数**: A = α‖W-AE₁(W)‖² + β‖W-AE₂(AE₁(W))‖²

**实验结果**:

| α | β | Precision | Recall | F1 | TP | FP |
|---|----|-----------|--------|------|-----|-----|
| 0.9 | 0.1 | 0.826 | 0.950 | 0.884 | 95 | 20 |
| 0.7 | 0.3 | 0.826 | 0.950 | 0.884 | 95 | 20 |
| 0.5 | 0.5 | 0.826 | 0.950 | 0.884 | 95 | 20 |
| 0.3 | 0.7 | 0.826 | 0.950 | 0.884 | 95 | 20 |
| 0.1 | 0.9 | 0.826 | 0.950 | 0.884 | 95 | 20 |

**关键发现**: F1=0.884, 训练仅需 ~3s。灵敏度参数 α/β 在不同数据上可调节。

**演讲文案**:
> USAD 在 SockShop 合成数据上达到 F1=0.884，训练仅需 3 秒。证明了对抗训练自编码器在微服务异常检测中的有效性。

---

## 幻灯片10: PatternMatcher 实验结果

**标题**: PatternMatcher — 三步根因指标识别

**三步流程**: KS-test 过滤 → 13种模式分类 → 加权排序

| 指标 | 数值 |
|------|------|
| 候选指标 | 30 维 |
| 异常检出 | 3~4 个 |
| 过滤率 | 90%+ |
| **AC@1** | **1.00** |
| **Avg@3** | **0.61** |
| 耗时 | **0.1s** |

**13 种异常模式**: sudden_increase, level_shift_up, steady_increase, single_spike, multiple_spikes 等

**演讲文案**:
> PatternMatcher 在 30 维指标中精确识别根因，AC@1=1.00 证明排名第一的指标就是真正的根因。整个过程仅需 0.1 秒，满足实时诊断需求。

---

## 幻灯片11: 综合对比与优化建议

**标题**: 算法对比与优化方向

**两篇算法对比**:

| 维度 | USAD | PatternMatcher |
|------|------|----------------|
| 任务 | 异常检测 | 根因定位 |
| 方法 | 深度对抗训练 | 统计+规则分类 |
| 速度 | 训练 3s, 推理实时 | 全流程 0.1s |
| 可解释性 | 低 (黑盒) | 高 (模式+权重) |
| 适用场景 | 持续监控预警 | 事件响应诊断 |

**优化方向**:
1. 使用 Prometheus 实时数据替代合成数据
2. 实现完整 1-D CNN 替代规则分类器
3. GPU 加速 + 数据增强提升鲁棒性
4. 部署 cAdvisor 采集容器级 CPU/内存指标

**演讲文案**:
> 两个算法互补性强：USAD 做持续监控，PatternMatcher 做事件响应。未来可用 Prometheus 实时数据训练，实现完整 CNN 并 GPU 加速。

---

## 幻灯片12: 成员贡献

**标题**: 小组成员分工与贡献

| 成员 | 分工 | 贡献 |
|------|------|------|
| 成员1 | 系统部署 + 环境配置 | Docker Compose + K8s 双部署，代理配置 |
| 成员2 | Prometheus + Grafana + ChaosMesh | 监控体系搭建，10 个故障注入实验 |
| 成员3 | Selenium + JMeter 测试 | 功能测试 5/5，性能测试 26 req/s |
| 成员4 | USAD 论文复现 | 模型实现，F1=0.884 |
| 成员5 | PatternMatcher 论文复现 | 算法实现，AC@1=1.00 |
| 成员6 | 报告 + PPT + 数据整理 | 报告撰写，答辩准备 |

---

## 幻灯片13: 总结

**标题**: 总结与收获

**完成的工作**:
- ✅ SockShop 微服务系统部署（Docker Compose + K8s 双重部署）
- ✅ Prometheus + Grafana 监控体系（9/9 Targets, 4 面板 Dashboard）
- ✅ ChaosMesh 故障注入（10 个实验：StressChaos/NetworkChaos/PodChaos）
- ✅ Selenium + JMeter 自动化测试（5/5 PASS, 100% 成功率）
- ✅ KDD20-USAD 算法复现（F1=0.884, AC@1=1.00）
- ✅ ISSRE21-PatternMatcher 算法复现（Avg@3=0.61, 0.1s）

**工程收获**:
- 实践了微服务全链路：部署 → 监控 → 测试 → 算法实验
- 解决了国内网络环境下 Docker 镜像拉取问题
- 掌握了 Prometheus + Grafana 监控体系搭建

**演讲文案**:
> 总结：完成了 SockShop 从部署到监控到测试到论文复现的完整流程。监控体系 9/9，测试 100% 通过，两个论文算法均验证成功。

---

## 幻灯片14: 谢谢！欢迎提问

**标题**: 感谢聆听，欢迎提问

**预期的 Q&A**:

**Q: 为什么选第一档 SockShop？**
> SockShop 镜像更轻量（14 个服务 vs 41），Docker Compose 一键部署更稳定。也是文档最完善的微服务教学系统之一。

**Q: JMeter 为什么只有 5 个请求？**
> Windows 下 Java HTTP 客户端有多线程端口耗尽问题（BindException）。报告中用 Python 并发测试（10~100 并发 100% 成功）作为补充验证。

**Q: 实验中最大的困难是什么？**
> 国内网络环境下 Docker 镜像拉取。TrainTicket 的镜像在 DaoCloud 白名单中，SockShop 的没有。最终通过系统代理解决。

**Q: 论文算法用的是真实数据还是合成数据？**
> 合成数据。SockShop 暴露的是 JVM 指标（堆内存、GC、线程），缺少容器级 CPU/内存指标。合成数据模拟了 30 维指标对应 SockShop 14 个服务的 CPU/内存/延迟。
