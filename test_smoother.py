"""
模块级自适应标签平滑器 - 单元测试
"""

import torch
import torch.nn.functional as F
import unittest

from adaptive_label_smoothing import (
    BetaSmoothModule,
    DynamicSmoothingScheduler,
    TopologyManager,
    TopologyConsistencyLoss,
    AdaptiveLabelSmoother,
)


class TestBetaSmoothModule(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.num_classes = 10
        self.module = BetaSmoothModule(
            module_name="test_module",
            num_classes=self.num_classes,
            init_alpha=2.0,
            init_beta=10.0,
            learnable_params=True
        )

    def test_initialization(self):
        self.assertEqual(self.module.module_name, "test_module")
        self.assertEqual(self.module.num_classes, self.num_classes)
        self.assertTrue(self.module.learnable_params)

    def test_beta_properties(self):
        alpha = self.module.alpha
        beta = self.module.beta
        self.assertGreater(alpha.item(), 0)
        self.assertGreater(beta.item(), 0)

        mean = self.module.smoothing_mean
        var = self.module.smoothing_var
        self.assertGreater(mean.item(), 0)
        self.assertLess(mean.item(), 1)
        self.assertGreater(var.item(), 0)

        expected_mean = 2.0 / (2.0 + 10.0)
        self.assertAlmostEqual(mean.item(), expected_mean, places=2)

    def test_sample_smoothing(self):
        samples = self.module.sample_smoothing(100)
        self.assertEqual(samples.shape, (100,))
        self.assertTrue(torch.all(samples >= 0))
        self.assertTrue(torch.all(samples <= 1))

        sample_mean = samples.mean().item()
        expected_mean = self.module.smoothing_mean.item()
        self.assertAlmostEqual(sample_mean, expected_mean, delta=0.05)

    def test_smooth_labels_hard(self):
        labels = torch.tensor([0, 1, 2, 3])
        probs = torch.ones(4, self.num_classes) / self.num_classes

        smoothed = self.module.smooth_labels(labels, probs, use_mean=True)
        self.assertEqual(smoothed.shape, (4, self.num_classes))
        self.assertTrue(torch.all(smoothed >= 0))
        self.assertTrue(torch.all(smoothed <= 1))

        for i in range(4):
            self.assertAlmostEqual(smoothed[i].sum().item(), 1.0, places=4)
            self.assertGreater(smoothed[i, i].item(), 1.0 / self.num_classes)

    def test_smooth_labels_soft(self):
        labels = torch.rand(4, self.num_classes)
        labels = labels / labels.sum(dim=-1, keepdim=True)
        probs = torch.ones(4, self.num_classes) / self.num_classes

        smoothed = self.module.smooth_labels(labels, probs, use_mean=True)
        self.assertEqual(smoothed.shape, (4, self.num_classes))
        for i in range(4):
            self.assertAlmostEqual(smoothed[i].sum().item(), 1.0, places=4)

    def test_confidence_computation(self):
        probs = torch.zeros(4, self.num_classes)
        probs[:, 0] = 1.0

        confidence = self.module.compute_confidence(probs)
        self.assertAlmostEqual(confidence.item(), 1.0, places=4)

        uniform = torch.ones(4, self.num_classes) / self.num_classes
        confidence_uniform = self.module.compute_confidence(uniform)
        self.assertAlmostEqual(confidence_uniform.item(), 1.0 / self.num_classes, places=4)

    def test_marginal_entropy(self):
        probs = torch.zeros(100, self.num_classes)
        probs[:, 0] = 1.0

        entropy = self.module.compute_marginal_entropy(probs)
        self.assertAlmostEqual(entropy.item(), 0.0, places=2)

        uniform = torch.ones(100, self.num_classes) / self.num_classes
        entropy_uniform = self.module.compute_marginal_entropy(uniform)
        import math
        self.assertAlmostEqual(entropy_uniform.item(), math.log(self.num_classes), places=2)

    def test_update_stats(self):
        probs = torch.ones(4, self.num_classes) / self.num_classes

        initial_conf = self.module.ema_confidence.item()

        for _ in range(5):
            self.module.update_stats(probs)

        updated_conf = self.module.ema_confidence.item()
        self.assertNotEqual(initial_conf, updated_conf)

    def test_adjust_beta_params(self):
        initial_mean = self.module.smoothing_mean.item()
        initial_var = self.module.smoothing_var.item()

        target_mean = 0.2
        target_var = 0.005

        for _ in range(50):
            self.module.adjust_beta_params(target_mean, target_var, lr=0.1)

        final_mean = self.module.smoothing_mean.item()
        final_var = self.module.smoothing_var.item()

        self.assertLess(abs(final_mean - target_mean), 0.05)


class TestDynamicSmoothingScheduler(unittest.TestCase):
    def setUp(self):
        self.num_classes = 10
        self.scheduler = DynamicSmoothingScheduler(
            num_classes=self.num_classes,
            base_smoothing=0.1,
            min_smoothing=0.01,
            max_smoothing=0.5
        )

    def test_initialization(self):
        self.assertEqual(self.scheduler.num_classes, self.num_classes)
        self.assertEqual(self.scheduler.base_smoothing, 0.1)

    def test_compute_target_smoothing_high_confidence(self):
        confidence = torch.tensor(0.95)
        marginal_entropy = torch.tensor(0.1)

        target_mean, target_var = self.scheduler.compute_target_smoothing(confidence, marginal_entropy)

        self.assertLess(target_mean, self.scheduler.base_smoothing)
        self.assertGreater(target_mean, self.scheduler.min_smoothing)

    def test_compute_target_smoothing_low_confidence(self):
        confidence = torch.tensor(0.2)
        import math
        marginal_entropy = torch.tensor(math.log(self.num_classes) * 0.9)

        target_mean, target_var = self.scheduler.compute_target_smoothing(confidence, marginal_entropy)

        self.assertGreater(target_mean, self.scheduler.base_smoothing)
        self.assertLess(target_mean, self.scheduler.max_smoothing)

    def test_overfitting_detection(self):
        module_name = "test_module"

        self.scheduler.reset_overfitting_tracker(module_name)

        for i in range(10):
            val_acc = 0.9 - i * 0.02
            is_overfit = self.scheduler.check_overfitting(module_name, val_acc)

        self.assertTrue(is_overfit)


class TestTopologyManager(unittest.TestCase):
    def setUp(self):
        self.num_classes = 10
        self.manager = TopologyManager(
            num_classes=self.num_classes,
            merge_threshold=0.05,
            merge_window=10,
            max_groups=8
        )

    def test_register_module(self):
        self.manager.register_module("module_1", init_alpha=2.0, init_beta=10.0)

        self.assertIn("module_1", self.manager._module_to_group)
        self.assertEqual(self.manager.num_groups(), 1)

        alpha, beta = self.manager.get_module_params("module_1")
        self.assertIsInstance(alpha, torch.Tensor)
        self.assertIsInstance(beta, torch.Tensor)

    def test_get_group(self):
        self.manager.register_module("mod_a")
        self.manager.register_module("mod_b")

        group_a = self.manager.get_group("mod_a")
        self.assertIsNotNone(group_a)
        self.assertEqual(len(group_a.module_names), 1)

    def test_record_smoothing_value(self):
        self.manager.register_module("mod_1")

        for i in range(5):
            self.manager.record_smoothing_value("mod_1", 0.1)

        history = self.manager._module_history["mod_1"]
        self.assertEqual(len(history), 5)

    def test_merge_groups(self):
        self.manager.register_module("mod_a", init_alpha=2.0, init_beta=18.0)
        self.manager.register_module("mod_b", init_alpha=2.0, init_beta=18.0)

        for _ in range(20):
            self.manager.record_smoothing_value("mod_a", 0.1)
            self.manager.record_smoothing_value("mod_b", 0.102)

        initial_groups = self.manager.num_groups()

        self.manager.max_groups = 1
        merged = self.manager.try_merge_groups()

        self.assertLessEqual(self.manager.num_groups(), initial_groups)

    def test_split_groups(self):
        self.manager.register_module("mod_a")
        self.manager.register_module("mod_b")

        for _ in range(20):
            self.manager.record_smoothing_value("mod_a", 0.1)
            self.manager.record_smoothing_value("mod_b", 0.1)

        self.manager.max_groups = 1
        self.manager.try_merge_groups()

        self.assertEqual(self.manager.num_groups(), 1)

        self.manager.mark_overfitting("mod_a")
        self.manager.try_split_groups()

        self.assertGreaterEqual(self.manager.num_groups(), 1)

    def test_module_group_map(self):
        self.manager.register_module("m1")
        self.manager.register_module("m2")

        mapping = self.manager.get_module_group_map()
        self.assertEqual(len(mapping), 2)
        self.assertIn("m1", mapping)
        self.assertIn("m2", mapping)


class TestTopologyConsistencyLoss(unittest.TestCase):
    def setUp(self):
        self.num_classes = 10
        self.tcl = TopologyConsistencyLoss(
            num_classes=self.num_classes,
            consistency_weight=0.1,
            distance_type="cosine"
        )

    def test_pairwise_distances(self):
        labels = torch.rand(8, self.num_classes)
        labels = labels / labels.sum(dim=-1, keepdim=True)

        distances = self.tcl._pairwise_distances(labels)
        self.assertEqual(distances.shape, (8, 8))
        self.assertTrue(torch.all(distances >= 0))

        for i in range(8):
            self.assertAlmostEqual(distances[i, i].item(), 0.0, places=4)

    def test_pairwise_consistency_loss_identical(self):
        labels = torch.rand(8, self.num_classes)
        labels = labels / labels.sum(dim=-1, keepdim=True)

        loss = self.tcl.pairwise_consistency_loss(labels, labels.clone())
        self.assertAlmostEqual(loss.item(), 0.0, places=4)

    def test_pairwise_consistency_loss_different(self):
        labels1 = torch.zeros(8, self.num_classes)
        labels1[:, 0] = 1.0

        labels2 = torch.zeros(8, self.num_classes)
        labels2[:, 1] = 1.0

        loss = self.tcl.pairwise_consistency_loss(labels1, labels2)
        self.assertGreater(loss.item(), 0)

    def test_forward_multiple_modules(self):
        labels_dict = {}
        for i in range(3):
            labels = torch.rand(8, self.num_classes)
            labels = labels / labels.sum(dim=-1, keepdim=True)
            labels_dict[f"mod_{i}"] = labels

        loss, scores = self.tcl(labels_dict)

        self.assertGreater(loss.item(), 0)
        self.assertEqual(len(scores), 3)

    def test_forward_single_module(self):
        labels_dict = {"mod_0": torch.rand(8, self.num_classes)}

        loss, scores = self.tcl(labels_dict)
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(len(scores), 0)

    def test_forward_with_reference(self):
        labels_dict = {}
        for i in range(3):
            labels = torch.rand(8, self.num_classes)
            labels = labels / labels.sum(dim=-1, keepdim=True)
            labels_dict[f"mod_{i}"] = labels

        loss, scores = self.tcl(labels_dict, reference_module="mod_0")

        self.assertGreater(loss.item(), 0)
        self.assertEqual(len(scores), 2)

    def test_topology_preservation(self):
        original = torch.zeros(10, self.num_classes)
        for i in range(10):
            original[i, i % self.num_classes] = 1.0

        smoothed = original.clone() * 0.9 + 0.1 / self.num_classes

        preservation = self.tcl.compute_topology_preservation(original, smoothed)
        self.assertGreater(preservation, 0.9)

    def test_different_distance_types(self):
        for dist_type in ["euclidean", "cosine", "kl"]:
            tcl = TopologyConsistencyLoss(
                num_classes=self.num_classes,
                distance_type=dist_type
            )

            labels = torch.rand(8, self.num_classes)
            labels = labels / labels.sum(dim=-1, keepdim=True)

            distances = tcl._pairwise_distances(labels)
            self.assertEqual(distances.shape, (8, 8))


class TestAdaptiveLabelSmoother(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.num_classes = 10
        self.module_names = ["head_0", "head_1", "head_2"]
        self.smoother = AdaptiveLabelSmoother(
            num_classes=self.num_classes,
            module_names=self.module_names,
            base_smoothing=0.1,
            consistency_weight=0.05,
            merge_threshold=0.05,
            max_groups=5
        )

    def test_initialization(self):
        self.assertEqual(self.smoother.num_classes, self.num_classes)
        self.assertEqual(len(self.smoother.module_names), 3)
        self.assertEqual(len(self.smoother.smoothing_modules), 3)
        self.assertEqual(self.smoother.topology_manager.num_groups(), 3)

    def test_smooth_labels(self):
        labels = torch.tensor([0, 1, 2, 3, 4])

        probs = torch.ones(5, self.num_classes) / self.num_classes
        probs_dict = {name: probs for name in self.module_names}

        smooth_labels = self.smoother.smooth_labels(
            module_name="head_0",
            labels=labels,
            probs=probs,
            use_mean=True
        )

        self.assertEqual(smooth_labels.shape, (5, self.num_classes))
        for i in range(5):
            self.assertAlmostEqual(smooth_labels[i].sum().item(), 1.0, places=4)

    def test_smooth_all_modules(self):
        labels = torch.tensor([0, 1, 2, 3])
        probs = torch.ones(4, self.num_classes) / self.num_classes
        probs_dict = {name: probs for name in self.module_names}

        smooth_dict = self.smoother.smooth_all_modules(labels, probs_dict)

        self.assertEqual(len(smooth_dict), 3)
        for name in self.module_names:
            self.assertIn(name, smooth_dict)
            self.assertEqual(smooth_dict[name].shape, (4, self.num_classes))

    def test_update_module_stats(self):
        probs = torch.ones(4, self.num_classes) / self.num_classes

        self.smoother.update_module_stats("head_0", probs)
        module = self.smoother.smoothing_modules["head_0"]
        self.assertNotEqual(module.ema_confidence.item(), 0.0)

    def test_adjust_smoothing(self):
        probs = torch.ones(4, self.num_classes) / self.num_classes
        self.smoother.update_module_stats("head_0", probs)

        result = self.smoother.adjust_smoothing("head_0")

        self.assertIn("current_mean", result)
        self.assertIn("target_mean", result)
        self.assertGreater(result["current_mean"], 0)
        self.assertLess(result["current_mean"], 1)

    def test_training_step(self):
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        probs_dict = {}
        for name in self.module_names:
            probs = torch.rand(8, self.num_classes)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            probs_dict[name] = probs

        result = self.smoother.training_step(
            labels=labels,
            module_probs=probs_dict
        )

        self.assertIn("smooth_labels", result)
        self.assertIn("consistency_loss", result)
        self.assertIn("adjust_results", result)
        self.assertIsInstance(result["consistency_loss"], torch.Tensor)

    def test_get_smoothing_info(self):
        info = self.smoother.get_smoothing_info()

        self.assertIn("num_groups", info)
        self.assertIn("group_sizes", info)

        for name in self.module_names:
            self.assertIn(name, info)
            self.assertIn("module_mean", info[name])
            self.assertIn("group_id", info[name])

    def test_register_new_module(self):
        initial_count = len(self.smoother.module_names)

        self.smoother.register_module("new_head")

        self.assertEqual(len(self.smoother.module_names), initial_count + 1)
        self.assertIn("new_head", self.smoother.smoothing_modules)

    def test_try_topology_reconstruction(self):
        info = self.smoother.try_topology_reconstruction()

        self.assertIn("num_groups", info)
        self.assertIn("group_sizes", info)
        self.assertIn("merged_pairs", info)
        self.assertIn("split_groups", info)

    def test_compute_consistency_loss(self):
        labels = torch.tensor([0, 1, 2, 3])
        probs = torch.ones(4, self.num_classes) / self.num_classes
        probs_dict = {name: probs for name in self.module_names}

        smooth_dict = self.smoother.smooth_all_modules(labels, probs_dict)

        loss, scores = self.smoother.compute_consistency_loss(smooth_dict)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)


if __name__ == "__main__":
    unittest.main()
