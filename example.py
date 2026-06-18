"""
模块级自适应标签平滑器 - 使用示例

演示如何在多模块网络中使用自适应标签平滑器，
包括动态调整平滑系数、拓扑重构（合并/分裂）和拓扑一致性约束。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict

from adaptive_label_smoothing import AdaptiveLabelSmoother


class MultiModuleClassifier(nn.Module):
    """
    具有多个分类头的示例网络。
    每个分类头可以看作一个"模块"，需要独立的标签平滑策略。
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_modules: int = 3):
        super().__init__()
        self.num_modules = num_modules
        self.num_classes = num_classes

        self.shared_backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_modules)
        ])

        self.module_names = [f"head_{i}" for i in range(num_modules)]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.shared_backbone(x)

        outputs = {}
        for i, head in enumerate(self.heads):
            logits = head(features)
            probs = F.softmax(logits, dim=-1)
            outputs[self.module_names[i]] = probs

        return outputs


def generate_synthetic_data(batch_size: int, input_dim: int, num_classes: int):
    """生成合成训练数据"""
    x = torch.randn(batch_size, input_dim)
    y = torch.randint(0, num_classes, (batch_size,))
    return x, y


def main():
    torch.manual_seed(42)

    input_dim = 64
    hidden_dim = 128
    num_classes = 10
    num_modules = 4
    batch_size = 32
    num_epochs = 5
    steps_per_epoch = 20

    print("=" * 70)
    print("模块级自适应标签平滑器 - 演示")
    print("=" * 70)

    model = MultiModuleClassifier(input_dim, hidden_dim, num_classes, num_modules)
    module_names = model.module_names

    smoother = AdaptiveLabelSmoother(
        num_classes=num_classes,
        module_names=module_names,
        base_smoothing=0.1,
        min_smoothing=0.01,
        max_smoothing=0.4,
        consistency_weight=0.05,
        merge_threshold=0.03,
        merge_window=10,
        max_groups=3,
    )

    optimizer = optim.Adam([
        {"params": model.parameters()},
        {"params": smoother.parameters(), "lr": 0.001}
    ], lr=0.001)

    print(f"\n模型配置:")
    print(f"  输入维度: {input_dim}")
    print(f"  隐藏层维度: {hidden_dim}")
    print(f"  类别数: {num_classes}")
    print(f"  模块数: {num_modules}")
    print(f"  模块名: {module_names}")
    print(f"  初始平滑系数组数: {smoother.topology_manager.num_groups()}")

    print("\n" + "=" * 70)
    print("训练开始")
    print("=" * 70)

    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_cons_loss = 0.0

        for step in range(steps_per_epoch):
            global_step += 1

            x, y = generate_synthetic_data(batch_size, input_dim, num_classes)

            optimizer.zero_grad()

            probs_dict = model(x)

            train_result = smoother.training_step(
                labels=y,
                module_probs=probs_dict,
                use_group_params=True
            )

            smooth_labels_dict = train_result["smooth_labels"]
            consistency_loss = train_result["consistency_loss"]

            cls_loss = 0.0
            for name in module_names:
                probs = probs_dict[name]
                smooth_labels = smooth_labels_dict[name]
                log_probs = torch.log(probs + 1e-8)
                loss = -(smooth_labels * log_probs).sum(dim=-1).mean()
                cls_loss = cls_loss + loss

            cls_loss = cls_loss / len(module_names)
            total_loss = cls_loss + consistency_loss

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_cons_loss += consistency_loss.item()

            if step == 0:
                info = smoother.get_smoothing_info()
                print(f"\nEpoch {epoch + 1}/{num_epochs}, Step {step + 1}:")
                print(f"  总损失: {total_loss.item():.4f}")
                print(f"  分类损失: {cls_loss.item():.4f}")
                print(f"  一致性损失: {consistency_loss.item():.6f}")
                print(f"  平滑系数组数: {info['num_groups']}")
                print(f"  组大小: {info['group_sizes']}")

                for name in module_names:
                    mod_info = info[name]
                    print(f"  {name}:")
                    print(f"    平滑均值(组): {mod_info['group_mean']:.4f}")
                    print(f"    平滑方差(组): {mod_info['group_var']:.6f}")
                    print(f"    置信度(EMA): {mod_info['confidence']:.4f}")
                    print(f"    边缘熵(EMA): {mod_info['marginal_entropy']:.4f}")
                    print(f"    组ID: {mod_info['group_id']}")

            if train_result["topology_info"] is not None:
                topo_info = train_result["topology_info"]
                if topo_info["merged_pairs"] or topo_info["split_groups"]:
                    print(f"\n  [拓扑重构] Step {global_step}:")
                    print(f"    合并的组对: {topo_info['merged_pairs']}")
                    print(f"    分裂的组: {topo_info['split_groups']}")
                    print(f"    当前组数: {topo_info['num_groups']}")
                    print(f"    模块-组映射: {topo_info['module_group_map']}")

        avg_loss = epoch_loss / steps_per_epoch
        avg_cls_loss = epoch_cls_loss / steps_per_epoch
        avg_cons_loss = epoch_cons_loss / steps_per_epoch

        print(f"\nEpoch {epoch + 1} 平均:")
        print(f"  总损失: {avg_loss:.4f}")
        print(f"  分类损失: {avg_cls_loss:.4f}")
        print(f"  一致性损失: {avg_cons_loss:.6f}")

    print("\n" + "=" * 70)
    print("训练完成")
    print("=" * 70)

    final_info = smoother.get_smoothing_info()
    print(f"\n最终状态:")
    print(f"  平滑系数组数: {final_info['num_groups']}")
    print(f"  组大小: {final_info['group_sizes']}")

    for name in module_names:
        mod_info = final_info[name]
        print(f"  {name}: 平滑均值={mod_info['group_mean']:.4f}, "
              f"组ID={mod_info['group_id']}")

    print("\n" + "=" * 70)
    print("拓扑一致性验证")
    print("=" * 70)

    x_test, y_test = generate_synthetic_data(64, input_dim, num_classes)
    with torch.no_grad():
        probs_test = model(x_test)
        smooth_test = smoother.smooth_all_modules(
            labels=y_test,
            module_probs=probs_test,
            use_mean=True
        )

        from adaptive_label_smoothing import TopologyConsistencyLoss
        tcl = TopologyConsistencyLoss(num_classes=num_classes, distance_type="cosine")

        print("\n各模块平滑标签之间的一致性得分 (越高越好, 范围[0,1]):")
        _, scores = tcl(smooth_test)
        for pair, score in scores.items():
            print(f"  {pair}: {score:.4f}")

        print("\n平滑前后的拓扑保持度 (排序相关系数, 越高越好):")
        for name in module_names:
            orig_onehot = F.one_hot(y_test, num_classes=num_classes).float()
            preservation = tcl.compute_topology_preservation(orig_onehot, smooth_test[name])
            print(f"  {name}: {preservation:.4f}")

    print("\n演示完成!")


if __name__ == "__main__":
    main()
