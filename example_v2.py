"""
模块级自适应标签平滑器 - v2 示例
演示: 训练入口、拓扑合并、状态保存、诊断导出
"""

import torch
import torch.nn as nn
import torch.optim as optim

from adaptive_label_smoothing import AdaptiveLabelSmoother


class MultiHeadNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads=3):
        super().__init__()
        self.num_heads = num_heads
        self.module_names = [f"head_{i}" for i in range(num_heads)]
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_heads)
        ])

    def forward(self, x):
        feats = self.backbone(x)
        return {name: head(feats) for name, head in zip(self.module_names, self.heads)}


def demo_compute_losses():
    print("=" * 60)
    print("Demo 1: compute_losses() 训练入口")
    print("=" * 60)

    torch.manual_seed(42)
    model = MultiHeadNet(32, 64, 10, 3)
    smoother = AdaptiveLabelSmoother(
        num_classes=10, module_names=model.module_names,
        base_smoothing=0.1, consistency_weight=0.01,
    )
    opt = optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(8, 32)
    y = torch.randint(0, 10, (8,))

    opt.zero_grad()
    logits = model(x)
    result = smoother.compute_losses(logits, y)

    print(f"  total_loss: {result['total_loss'].item():.4f}")
    print(f"  classification_loss: {result['classification_loss'].item():.4f}")
    print(f"  consistency_loss: {result['consistency_loss'].item():.6f}")
    print(f"  step: {result['step']}")
    print(f"  per_module_loss keys: {list(result['per_module_loss'].keys())}")

    result["total_loss"].backward()
    opt.step()

    print("  [OK] 反向传播正常")
    print()


def demo_topology_merge():
    print("=" * 60)
    print("Demo 2: 拓扑重构 - 模块收敛自动合并")
    print("=" * 60)

    smoother = AdaptiveLabelSmoother(
        num_classes=10,
        module_names=["mod_a", "mod_b", "mod_c"],
        merge_threshold=0.02,
        merge_window=10,
        max_groups=10,
    )

    print(f"  初始组数: {smoother.topology_manager.num_groups()}")
    print(f"  初始组映射: {smoother.topology_manager.get_module_group_map()}")

    print("  模拟 15 步训练，mod_a 和 mod_b 收敛到相近值...")
    for i in range(15):
        smoother.topology_manager.record_smoothing_value("mod_a", 0.10)
        smoother.topology_manager.record_smoothing_value("mod_b", 0.101)
        smoother.topology_manager.record_smoothing_value("mod_c", 0.15)

    merged = smoother.topology_manager.try_merge_groups()
    group_map = smoother.topology_manager.get_module_group_map()

    print(f"  合并后组数: {smoother.topology_manager.num_groups()}")
    print(f"  合并后组映射: {group_map}")
    print(f"  合并的组对: {merged}")

    if group_map["mod_a"] == group_map["mod_b"] and group_map["mod_a"] != group_map["mod_c"]:
        print("  [OK] mod_a 和 mod_b 已合并，mod_c 独立")
    else:
        print("  [WARN] 合并结果未达预期")
    print()


def demo_save_load():
    print("=" * 60)
    print("Demo 3: 状态保存与恢复")
    print("=" * 60)

    torch.manual_seed(42)
    smoother = AdaptiveLabelSmoother(
        num_classes=10, module_names=["h0", "h1", "h2"],
        merge_threshold=0.05, merge_window=5,
    )

    print("  预训练 20 步...")
    for _ in range(20):
        y = torch.randint(0, 10, (4,))
        logits = {n: torch.randn(4, 10) for n in ["h0", "h1", "h2"]}
        smoother.compute_losses(logits, y)

    step_before = smoother._step_count
    groups_before = smoother.topology_manager.get_module_group_map()
    ema_before = smoother.smoothing_modules["h0"].ema_confidence.item()

    print(f"  保存前: step={step_before}, groups={groups_before}")

    state = smoother.state_dict()
    print(f"  state_dict 键数: {len(state)}")

    new_smoother = AdaptiveLabelSmoother(
        num_classes=10, module_names=["h0", "h1", "h2"],
    )
    new_smoother.load_state_dict(state)

    step_after = new_smoother._step_count
    groups_after = new_smoother.topology_manager.get_module_group_map()
    ema_after = new_smoother.smoothing_modules["h0"].ema_confidence.item()

    print(f"  加载后: step={step_after}, groups={groups_after}")

    ok = (step_before == step_after
          and groups_before == groups_after
          and abs(ema_before - ema_after) < 1e-6)
    print(f"  状态一致性: {'[OK]' if ok else '[FAIL]'}")
    print()


def demo_diagnostics():
    print("=" * 60)
    print("Demo 4: 诊断数据导出")
    print("=" * 60)

    torch.manual_seed(42)
    smoother = AdaptiveLabelSmoother(
        num_classes=10, module_names=["h0", "h1", "h2"],
        merge_threshold=0.05, merge_window=8,
    )

    print("  训练 20 步...")
    for i in range(20):
        y = torch.randint(0, 10, (4,))
        logits = {n: torch.randn(4, 10) for n in ["h0", "h1", "h2"]}
        smoother.compute_losses(logits, y)

    log = smoother.get_diagnostic_log()
    print(f"  诊断日志条数: {len(log)}")
    print(f"  第1条 keys: {list(log[0].keys())}")
    print(f"  第1条 step: {log[0]['step']}, num_groups: {log[0]['num_groups']}")

    events = smoother.get_merge_split_events()
    print(f"  合并事件数: {len(events['merge_events'])}")
    print(f"  分裂事件数: {len(events['split_events'])}")

    json_str = smoother.export_diagnostics_json()
    print(f"  JSON 导出长度: {len(json_str)} chars")

    csv_str = smoother.export_diagnostics_csv()
    csv_lines = csv_str.strip().split('\n')
    print(f"  CSV 导出: {len(csv_lines) - 1} 行数据")

    print("  [OK] 诊断导出正常")
    print()


def demo_training_loss():
    print("=" * 60)
    print("Demo 5: training_loss() 训练循环入口")
    print("=" * 60)

    torch.manual_seed(42)
    model = MultiHeadNet(32, 64, 10, 3)
    smoother = AdaptiveLabelSmoother(
        num_classes=10, module_names=model.module_names,
        base_smoothing=0.1, consistency_weight=0.01,
    )
    opt = optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(8, 32)
    y = torch.randint(0, 10, (8,))

    print("  简单用法 - 直接返回 loss：")
    opt.zero_grad()
    logits = model(x)
    loss = smoother.training_loss(logits, y)
    print(f"    loss: {loss.item():.4f}, requires_grad: {loss.requires_grad}")
    loss.backward()
    opt.step()
    print("    [OK] 反向传播正常")

    print()
    print("  详细模式 - 返回 loss + details + log：")
    logits = model(x)
    loss, details = smoother.training_loss(logits, y, return_details=True)
    print(f"    loss: {loss.item():.4f}")
    print(f"    classification_loss: {details['classification_loss'].item():.4f}")
    print(f"    consistency_loss: {details['consistency_loss'].item():.6f}")
    print(f"    log keys: {list(details['log'].keys())}")
    print(f"    log.step: {details['log']['step']}")
    print(f"    log.num_groups: {details['log']['num_groups']}")
    print("    [OK] 详细信息完整")

    print()


def demo_warmup_and_freeze():
    print("=" * 60)
    print("Demo 6: warmup 预热 & 模块冻结")
    print("=" * 60)

    smoother = AdaptiveLabelSmoother(
        num_classes=10,
        module_names=["h0", "h1", "h2"],
        warmup_steps=8,
    )

    print(f"  warmup_steps: {smoother.warmup_steps}")
    print(f"  初始 step=0, in_warmup={smoother.in_warmup}")

    y = torch.randint(0, 10, (4,))
    logits = {n: torch.randn(4, 10) for n in ["h0", "h1", "h2"]}

    params_before = smoother.smoothing_modules["h0"].log_alpha.item()

    for i in range(5):
        smoother.training_loss(logits, y)

    params_after_warmup = smoother.smoothing_modules["h0"].log_alpha.item()
    print(f"  5 步后 (仍在 warmup): in_warmup={smoother.in_warmup}")
    print(f"  h0 log_alpha 变化: {params_before:.6f} -> {params_after_warmup:.6f}")
    print(f"  参数是否未变: {abs(params_before - params_after_warmup) < 1e-5}")

    for i in range(5):
        smoother.training_loss(logits, y)

    params_after_adjust = smoother.smoothing_modules["h0"].log_alpha.item()
    print(f"  10 步后 (已退出 warmup): in_warmup={smoother.in_warmup}")
    print(f"  h0 log_alpha 变化: {params_after_warmup:.6f} -> {params_after_adjust:.6f}")

    print()
    print("  冻结 h1：")
    smoother.freeze_module("h1")
    print(f"  冻结模块: {smoother.get_frozen_modules()}")

    params_h1_before = smoother.smoothing_modules["h1"].log_alpha.item()
    for i in range(10):
        smoother.training_loss(logits, y)
    params_h1_after = smoother.smoothing_modules["h1"].log_alpha.item()
    print(f"  10 步后 h1 log_alpha: {params_h1_before:.6f} -> {params_h1_after:.6f}")
    print(f"  冻结参数是否未变: {abs(params_h1_before - params_h1_after) < 1e-5}")

    smoother.unfreeze_module("h1")
    print(f"  解冻后冻结模块: {smoother.get_frozen_modules()}")
    print("  [OK] warmup 和冻结功能正常")
    print()


def demo_module_summary_and_curves():
    print("=" * 60)
    print("Demo 7: 模块汇总 & 曲线数据结构")
    print("=" * 60)

    torch.manual_seed(42)
    smoother = AdaptiveLabelSmoother(
        num_classes=10, module_names=["h0", "h1", "h2"],
        merge_threshold=0.05, merge_window=8,
    )

    print("  训练 15 步...")
    for i in range(15):
        y = torch.randint(0, 10, (4,))
        logits = {n: torch.randn(4, 10) for n in ["h0", "h1", "h2"]}
        smoother.training_loss(logits, y)

    summary = smoother.get_module_summary()
    print(f"  模块汇总 - {len(summary)} 个模块:")
    for name, s in summary.items():
        print(f"    {name}:")
        print(f"      当前组: {s['current_group_id']}")
        print(f"      平滑均值: {s['current_smoothing_mean']:.4f}")
        print(f"      平均平滑值: {s['avg_smoothing_mean']:.4f}")
        print(f"      最大方差: {s['max_smoothing_var']:.6f}")
        print(f"      合并次数: {s['merge_count']}, 分裂次数: {s['split_count']}")
        print(f"      是否冻结: {s['is_frozen']}")

    print()
    curves = smoother.get_smoothing_curves()
    print(f"  曲线数据结构:")
    print(f"    steps 长度: {len(curves['steps'])}")
    print(f"    num_groups 长度: {len(curves['num_groups'])}")
    print(f"    per_module 模块数: {len(curves['per_module'])}")
    print(f"    h0.smoothing_mean 长度: {len(curves['per_module']['h0']['smoothing_mean'])}")
    print(f"    合并事件数: {len(curves['merge_events'])}")
    print(f"    分裂事件数: {len(curves['split_events'])}")
    print()
    print("  [OK] 汇总和曲线数据正常")
    print()


def main():
    print()
    print("  模块级自适应标签平滑器 v3.0")
    print()

    demo_compute_losses()
    demo_topology_merge()
    demo_save_load()
    demo_diagnostics()
    demo_training_loss()
    demo_warmup_and_freeze()
    demo_module_summary_and_curves()

    print("=" * 60)
    print("  所有演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
