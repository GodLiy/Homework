# work.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

《软件测试与维护（2026年春）》课程大作业。选取 **SockShop** 微服务系统，完成部署、监控、测试和论文算法复现。评分档位：第一档。

**双重部署**：Docker Compose（日常开发 + Prometheus/Grafana） + K8s/Minikube（ChaosMesh 故障注入）。

## 环境与工具路径

| 工具 | 路径/命令 | 说明 |
|------|-----------|------|
| Python (3.14) | `py -3` | |
| JMeter 5.6.3 | `F:/CCode/Homework/apache-jmeter-5.6.3/bin/jmeter.bat` | |
| Minikube 1.38 | `minikube` | Docker 驱动，仅用于 ChaosMesh |
| Docker Compose | `docker compose` | SockShop 主体运行 |
| Helm | `helm` | ChaosMesh 安装 |
| Java 21 | `java` | JMeter 需要 |

**代理**: Windows 系统代理 `127.0.0.1`（端口通过 `reg query "HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings"` 查询，通常为 7897）。Docker Desktop 自动走系统代理，拉取镜像需确保代理开启。

**Python 依赖**: `py -3 -m pip install torch numpy scipy scikit-learn selenium webdriver-manager`

## 核心目录结构

```
F:/CCode/Homework/
├── sockshop/                  # SockShop 微服务系统
│   └── deploy/
│       ├── docker-compose/    # Docker Compose 部署 (日常运行 + 监控)
│       │   ├── docker-compose.yml
│       │   └── docker-compose.monitoring.yml  # Prometheus + Grafana + node-exporter
│       └── kubernetes/        # K8s 部署 (用于 ChaosMesh)
│           └── complete-demo.yaml
├── experiments/               # 阶段四：论文算法复现
│   ├── usad.py                #   USAD (Encoder+2Decoder，对抗训练)
│   ├── pattern_matcher.py     #   PatternMatcher (KS-test + 模式分类 + 排序)
│   ├── run_experiments.py     #   一键运行实验
│   └── results/               #   所有 JSON 报告
├── tests/                     # 阶段三：测试脚本
│   ├── selenium_test.py       #   Selenium 功能测试 (9 类，headless Chrome)
│   ├── sockshop_test.jmx      #   JMeter 测试计划
│   └── performance_data.json  #   页面加载时间数据
├── report/                    # 报告与答辩材料
├── KDD20-USAD.pdf             # 论文1
└── ISSRE21-PatternMatcher.pdf # 论文2
```

## 服务访问地址

| 服务 | 地址 | 凭据 |
|------|------|------|
| SockShop 首页 | http://localhost:8088/ | 无 |
| Prometheus | http://localhost:9090 | 无 |
| Grafana | http://localhost:3000 | admin / admin123 |
| Chaos Dashboard | http://localhost:2333 | chaos-admin / (K8s token) |
| Grafana Dashboard | http://localhost:3000/d/sockshop | 同上 |

### 启动所有服务

```bash
# SockShop + 监控 (Docker Compose)
cd F:/CCode/Homework/sockshop/deploy/docker-compose
docker compose -f docker-compose.yml up -d
docker compose -f docker-compose.monitoring.yml up -d prometheus grafana
```

### Minikube + ChaosMesh (K8s 部分)

```bash
minikube start --driver=docker --memory=4096
# 加载镜像到 minikube，然后部署 SockShop + ChaosMesh
# 详见: 答辩操作指南.md
```

## 实验运行

### 一键运行全部实验

```bash
py -3 F:/CCode/Homework/experiments/run_experiments.py
```

生成 30 维 SockShop 模拟数据，注入 3 个根因异常，训练 USAD 并运行 PatternMatcher。

### USAD 灵敏度分析

```python
from usad import run_usad_experiment
r = run_usad_experiment(train, test, labels, ws=10, ls=30, ep=80, alpha=0.5, beta=0.5)
```

Encoder(300→150→75→30) + 2×Decoder。二阶段训练：Phase-1 自编码器最小化重构误差，Phase-2 AE1 欺骗 AE2 放大异常。异常分数 = α‖W-AE₁‖² + β‖W-AE₂(AE₁)‖²。

### PatternMatcher 根因定位

```python
from pattern_matcher import run_pattern_matcher
r = run_pattern_matcher(metrics, inc_t=400, gt=[2,7,15])
```

三步：KS-test 过滤 → 13 种模式分类 → -log(P)×pw 排序。

### Selenium 功能测试

```bash
# 需要代理下载 ChromeDriver (首次)
HTTPS_PROXY=http://127.0.0.1:7897 py -3 F:/CCode/Homework/tests/selenium_test.py
```

## 已知问题

1. **镜像拉取需要代理**：`weaveworksdemos/*` 镜像不在国内镜像白名单，须通过 Windows 系统代理拉取。确认代理开启后 `docker pull`。
2. **Grafana 密码**：首次启动时数据库初始化的 admin/admin 可能不生效，用 `docker rm -f grafana && GF_SECURITY_ADMIN_PASSWORD=admin123 docker run ...` 重建。
3. **Node Exporter 挂载限制**：Docker Desktop Windows 不支持 `rslave` 挂载，需修改 `docker-compose.monitoring.yml` 移除 volumes/pid/command 配置段。
4. **Chaos Dashboard Google Fonts 被墙**：页面加载慢但 HTTP 200，等 10 秒或直接用 `kubectl get stresschaos` CLI。
5. **ChaosMesh Daemon MinGW 路径**：Helm 安装时 socket 路径被翻译为 Windows 路径，需 `kubectl patch daemonset chaos-daemon -n chaos-mesh --type json -p '[{"op":"replace","path":"/spec/template/spec/volumes/0/hostPath/path","value":"/var/run"}]'`。
6. **JMeter XML 格式严格**：手写 JMX 需包含完整的 `guiclass`/`testclass` 属性，否则加载失败。建议用并发 curl 脚本替代。

## 架构决策

- **双部署模式**：SockShop 主服务用 Docker Compose（稳定、资源少、快速启动），ChaosMesh 必须用 K8s（它是 K8s CRD 工具）。两套部署共享 SockShop 镜像。
- **合成数据**：实验用 `generate_sockshop_data()` 生成 30 维模拟数据（14 个 SockShop 服务的 CPU/内存指标），注入异常模式对应 PatternMatcher 的 13 种分类。
- **PatternMatcher 简化**：原论文 1-D CNN 用 `scipy.signal.find_peaks` + 趋势分析的规则分类器替代。
