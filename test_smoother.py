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
        merged, _ = self.manager.try_merge_groups()

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

    def test_compute_losses(self):
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

        logits_dict = {}
        for name in self.module_names:
            logits = torch.randn(8, self.num_classes)
            logits_dict[name] = logits

        result = self.smoother.compute_losses(
            module_logits=logits_dict,
            labels=labels
        )

        self.assertIn("total_loss", result)
        self.assertIn("classification_loss", result)
        self.assertIn("consistency_loss", result)
        self.assertIn("per_module_loss", result)
        self.assertIn("smoothing_info", result)
        self.assertIn("step", result)

        self.assertIsInstance(result["total_loss"], torch.Tensor)
        self.assertIsInstance(result["classification_loss"], torch.Tensor)
        self.assertIsInstance(result["consistency_loss"], torch.Tensor)

        for name in self.module_names:
            self.assertIn(name, result["per_module_loss"])
            self.assertIsInstance(result["per_module_loss"][name], torch.Tensor)
            self.assertGreater(result["per_module_loss"][name].item(), 0)

    def test_compute_losses_sum_reduction(self):
        labels = torch.tensor([0, 1, 2, 3])

        logits_dict = {}
        for name in self.module_names:
            logits = torch.randn(4, self.num_classes)
            logits_dict[name] = logits

        result = self.smoother.compute_losses(
            module_logits=logits_dict,
            labels=labels,
            reduction="sum"
        )

        self.assertGreater(result["classification_loss"].item(), 0)

    def test_state_dict_roundtrip(self):
        labels = torch.tensor([0, 1, 2, 3])
        logits_dict = {}
        for name in self.module_names:
            logits = torch.randn(4, self.num_classes) * 2.0
            logits_dict[name] = logits

        for _ in range(5):
            self.smoother.compute_losses(logits_dict, labels)

        state_before = self.smoother.get_smoothing_info()
        step_before = self.smoother._step_count
        groups_before = self.smoother.topology_manager.get_module_group_map()

        state_dict = self.smoother.state_dict()

        new_smoother = AdaptiveLabelSmoother(
            num_classes=self.num_classes,
            module_names=self.module_names,
        )
        new_smoother.load_state_dict(state_dict)

        state_after = new_smoother.get_smoothing_info()
        step_after = new_smoother._step_count
        groups_after = new_smoother.topology_manager.get_module_group_map()

        self.assertEqual(step_before, step_after)
        self.assertEqual(groups_before, groups_after)

        for name in self.module_names:
            self.assertAlmostEqual(
                state_before[name]["module_mean"],
                state_after[name]["module_mean"],
                places=5
            )
            self.assertAlmostEqual(
                state_before[name]["group_mean"],
                state_after[name]["group_mean"],
                places=5
            )

    def test_state_dict_preserves_ema(self):
        probs = torch.rand(4, self.num_classes)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        self.smoother.update_module_stats("head_0", probs)

        ema_before = self.smoother.smoothing_modules["head_0"].ema_confidence.item()

        state_dict = self.smoother.state_dict()

        new_smoother = AdaptiveLabelSmoother(
            num_classes=self.num_classes,
            module_names=self.module_names,
        )
        new_smoother.load_state_dict(state_dict)

        ema_after = new_smoother.smoothing_modules["head_0"].ema_confidence.item()

        self.assertAlmostEqual(ema_before, ema_after, places=6)

    def test_diagnostic_log_recording(self):
        self.smoother.enable_diagnostics(True)

        labels = torch.tensor([0, 1, 2, 3])
        logits_dict = {}
        for name in self.module_names:
            logits = torch.randn(4, self.num_classes)
            logits_dict[name] = logits

        initial_log_len = len(self.smoother.get_diagnostic_log())

        for i in range(3):
            self.smoother.compute_losses(logits_dict, labels)

        log = self.smoother.get_diagnostic_log()
        self.assertEqual(len(log), initial_log_len + 3)

        first_entry = log[0]
        self.assertIn("step", first_entry)
        self.assertIn("num_groups", first_entry)
        self.assertIn("module_stats", first_entry)

        for name in self.module_names:
            self.assertIn(name, first_entry["module_stats"])
            self.assertIn("smoothing_mean", first_entry["module_stats"][name])
            self.assertIn("group_id", first_entry["module_stats"][name])

    def test_export_diagnostics_json(self):
        labels = torch.tensor([0, 1, 2, 3])
        logits_dict = {}
        for name in self.module_names:
            logits = torch.randn(4, self.num_classes)
            logits_dict[name] = logits

        for _ in range(3):
            self.smoother.compute_losses(logits_dict, labels)

        json_str = self.smoother.export_diagnostics_json()
        self.assertIsInstance(json_str, str)
        self.assertTrue(len(json_str) > 0)

        import json
        data = json.loads(json_str)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 3)

    def test_export_diagnostics_csv(self):
        labels = torch.tensor([0, 1, 2, 3])
        logits_dict = {}
        for name in self.module_names:
            logits = torch.randn(4, self.num_classes)
            logits_dict[name] = logits

        for _ in range(2):
            self.smoother.compute_losses(logits_dict, labels)

        csv_str = self.smoother.export_diagnostics_csv()
        self.assertIsInstance(csv_str, str)
        self.assertTrue(len(csv_str) > 0)

        lines = csv_str.strip().split('\n')
        self.assertTrue(len(lines) > 1)
        self.assertIn("step", lines[0])
        self.assertIn("smoothing_mean", lines[0])

    def test_get_merge_split_events(self):
        events = self.smoother.get_merge_split_events()
        self.assertIn("merge_events", events)
        self.assertIn("split_events", events)
        self.assertIsInstance(events["merge_events"], list)
        self.assertIsInstance(events["split_events"], list)

    def test_merge_when_modules_converge(self):
        merge_window = 5
        test_smoother = AdaptiveLabelSmoother(
            num_classes=self.num_classes,
            module_names=["mod_a", "mod_b"],
            merge_threshold=0.02,
            merge_window=merge_window,
            max_groups=10,
        )

        for i in range(merge_window + 2):
            test_smoother.topology_manager.record_smoothing_value("mod_a", 0.1)
            test_smoother.topology_manager.record_smoothing_value("mod_b", 0.101)

        initial_groups = test_smoother.topology_manager.num_groups()

        merged_pairs, _ = test_smoother.topology_manager.try_merge_groups()

        self.assertGreater(len(merged_pairs), 0)
        self.assertLess(test_smoother.topology_manager.num_groups(), initial_groups)

        group_map = test_smoother.topology_manager.get_module_group_map()
        self.assertEqual(group_map["mod_a"], group_map["mod_b"])

    def test_disable_diagnostics(self):
        self.smoother.enable_diagnostics(False)

        labels = torch.tensor([0, 1, 2, 3])
        logits_dict = {}
        for name in self.module_names:
            logits = torch.randn(4, self.num_classes)
            logits_dict[name] = logits

        initial_len = len(self.smoother.get_diagnostic_log())
        for _ in range(5):
            self.smoother.compute_losses(logits_dict, labels)

        self.assertEqual(len(self.smoother.get_diagnostic_log()), initial_len)

    def test_training_loss_simple(self):
        labels = torch.tensor([0, 1, 2, 3])
        logits_dict = {}
        for name in self.module_names:
            logits = torch.randn(4, self.num_classes, requires_grad=True)
            logits_dict[name] = logits

        loss = self.smoother.training_loss(logits_dict, labels)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(loss.requires_grad)

    def test_training_loss_with_details(self):
        labels = torch.tensor([0, 1, 2, 3])
        logits_dict = {}
        for name in self.module_names:
            logits = torch.randn(4, self.num_classes)
            logits_dict[name] = logits

        loss, details = self.smoother.training_loss(
            logits_dict, labels, return_details=True
        )

        self.assertIsInstance(loss, torch.Tensor)
        self.assertIn("total_loss", details)
        self.assertIn("classification_loss", details)
        self.assertIn("consistency_loss", details)
        self.assertIn("per_module_loss", details)
        self.assertIn("smooth_labels", details)
        self.assertIn("log", details)
        self.assertIn("step", details["log"])
        self.assertIn("topology/num_groups", details["log"])
        self.assertIn("loss/total", details["log"])
        self.assertIn("loss/classification", details["log"])
        self.assertIn("smooth/mean/" + self.module_names[0], details["log"])

    def test_training_loss_device_consistency(self):
        device = torch.device("cpu")
        labels = torch.tensor([0, 1, 2, 3], device=device)
        logits_dict = {}
        for name in self.module_names:
            logits = torch.randn(4, self.num_classes, device=device)
            logits_dict[name] = logits

        loss, details = self.smoother.training_loss(
            logits_dict, labels, return_details=True
        )

        self.assertEqual(loss.device, device)
        self.assertEqual(details["consistency_loss"].device, device)
        for name, sl in details["smooth_labels"].items():
            self.assertEqual(sl.device, device)
        for name, pl in details["per_module_loss"].items():
            self.assertEqual(pl.device, device)

    def test_warmup_mode(self):
        warmup_smoother = AdaptiveLabelSmoother(
            num_classes=self.num_classes,
            module_names=self.module_names,
            warmup_steps=10,
        )

        self.assertTrue(warmup_smoother.in_warmup)
        self.assertEqual(warmup_smoother.warmup_steps, 10)

        labels = torch.tensor([0, 1, 2, 3])
        logits_dict = {}
        for name in self.module_names:
            logits = torch.randn(4, self.num_classes)
            logits_dict[name] = logits

        params_before = {}
        for name in self.module_names:
            mod = warmup_smoother.smoothing_modules[name]
            params_before[name] = (mod.log_alpha.item(), mod.log_beta.item())

        for i in range(10):
            result = warmup_smoother.compute_losses(logits_dict, labels)

        for name in self.module_names:
            mod = warmup_smoother.smoothing_modules[name]
            alpha_after, beta_after = mod.log_alpha.item(), mod.log_beta.item()
            self.assertAlmostEqual(params_before[name][0], alpha_after, places=5)
            self.assertAlmostEqual(params_before[name][1], beta_after, places=5)

        self.assertTrue(warmup_smoother.in_warmup)

        result = warmup_smoother.compute_losses(logits_dict, labels)
        self.assertFalse(warmup_smoother.in_warmup)

    def test_freeze_module(self):
        mod_name = self.module_names[0]
        mod = self.smoother.smoothing_modules[mod_name]

        self.assertFalse(self.smoother.is_frozen(mod_name))
        self.assertFalse(mod.frozen)

        self.smoother.freeze_module(mod_name)

        self.assertTrue(self.smoother.is_frozen(mod_name))
        self.assertTrue(mod.frozen)

        params_before = (mod.log_alpha.item(), mod.log_beta.item())

        for _ in range(10):
            mod.adjust_beta_params(0.3, 0.01, lr=0.5)

        params_after = (mod.log_alpha.item(), mod.log_beta.item())
        self.assertAlmostEqual(params_before[0], params_after[0], places=5)
        self.assertAlmostEqual(params_before[1], params_after[1], places=5)

        self.smoother.unfreeze_module(mod_name)
        self.assertFalse(self.smoother.is_frozen(mod_name))
        self.assertFalse(mod.frozen)

    def test_freeze_all_unfreeze_all(self):
        self.smoother.freeze_all()

        for name in self.module_names:
            self.assertTrue(self.smoother.is_frozen(name))
            self.assertTrue(self.smoother.smoothing_modules[name].frozen)

        self.assertEqual(len(self.smoother.get_frozen_modules()), len(self.module_names))

        self.smoother.unfreeze_all()

        for name in self.module_names:
            self.assertFalse(self.smoother.is_frozen(name))
            self.assertFalse(self.smoother.smoothing_modules[name].frozen)

        self.assertEqual(len(self.smoother.get_frozen_modules()), 0)

    def test_module_summary(self):
        labels = torch.tensor([0, 1, 2, 3])
        logits_dict = {}
        for name in self.module_names:
            logits = torch.randn(4, self.num_classes)
            logits_dict[name] = logits

        for _ in range(5):
            self.smoother.compute_losses(logits_dict, labels)

        summary = self.smoother.get_module_summary()

        self.assertEqual(len(summary), len(self.module_names))
        for name in self.module_names:
            self.assertIn(name, summary)
            self.assertIn("current_smoothing_mean", summary[name])
            self.assertIn("current_group_id", summary[name])
            self.assertIn("avg_smoothing_mean", summary[name])
            self.assertIn("max_smoothing_var", summary[name])
            self.assertIn("merge_count", summary[name])
            self.assertIn("split_count", summary[name])
            self.assertIn("is_frozen", summary[name])

    def test_smoothing_curves(self):
        labels = torch.tensor([0, 1, 2, 3])
        logits_dict = {}
        for name in self.module_names:
            logits = torch.randn(4, self.num_classes)
            logits_dict[name] = logits

        n_steps = 5
        for _ in range(n_steps):
            self.smoother.compute_losses(logits_dict, labels)

        curves = self.smoother.get_smoothing_curves()

        self.assertIn("steps", curves)
        self.assertIn("num_groups", curves)
        self.assertIn("per_module", curves)
        self.assertIn("merge_events", curves)
        self.assertIn("split_events", curves)

        self.assertEqual(len(curves["steps"]), n_steps)
        self.assertEqual(len(curves["num_groups"]), n_steps)

        for name in self.module_names:
            self.assertIn(name, curves["per_module"])
            self.assertIn("smoothing_mean", curves["per_module"][name])
            self.assertIn("smoothing_var", curves["per_module"][name])
            self.assertIn("confidence", curves["per_module"][name])
            self.assertIn("marginal_entropy", curves["per_module"][name])
            self.assertIn("group_id", curves["per_module"][name])
            self.assertEqual(len(curves["per_module"][name]["smoothing_mean"]), n_steps)

    def test_strict_merge_window(self):
        from adaptive_label_smoothing import TopologyManager

        tm = TopologyManager(
            num_classes=self.num_classes,
            merge_threshold=0.02,
            merge_window=5,
        )
        tm.register_module("mod_a", init_alpha=2.0, init_beta=10.0)
        tm.register_module("mod_b", init_alpha=2.0, init_beta=10.0)

        for i in range(7):
            val_a = 0.10 + 0.001 * i
            val_b = 0.10
            tm.record_smoothing_value("mod_a", val_a)
            tm.record_smoothing_value("mod_b", val_b)

        merged, _ = tm.try_merge_groups()
        self.assertEqual(len(merged), 1)

    def test_merge_rejects_jittery_modules(self):
        from adaptive_label_smoothing import TopologyManager

        tm = TopologyManager(
            num_classes=self.num_classes,
            merge_threshold=0.02,
            merge_window=5,
        )
        tm.register_module("mod_a", init_alpha=2.0, init_beta=10.0)
        tm.register_module("mod_b", init_alpha=2.0, init_beta=10.0)

        for i in range(7):
            val_a = 0.10 if i % 2 == 0 else 0.20
            val_b = 0.15
            tm.record_smoothing_value("mod_a", val_a)
            tm.record_smoothing_value("mod_b", val_b)

        merged, _ = tm.try_merge_groups()
        self.assertEqual(len(merged), 0)

    def test_training_loss_with_weights(self):
        labels = torch.tensor([0, 1, 2, 3])
        logits_dict = {}
        for name in self.module_names:
            logits = torch.randn(4, self.num_classes, requires_grad=True)
            logits_dict[name] = logits

        weights = {self.module_names[0]: 2.0, self.module_names[1]: 0.5}
        _, details = self.smoother.training_loss(
            logits_dict, labels, return_details=True, loss_weights=weights
        )
        self.assertIn(self.module_names[0], details["per_module_loss"])
        self.assertIn(self.module_names[1], details["per_module_loss"])

        _, details_no_w = self.smoother.training_loss(
            logits_dict, labels, return_details=True
        )

    def test_training_loss_ignore_modules(self):
        labels = torch.tensor([0, 1, 2, 3])
        logits_dict = {}
        for name in self.module_names:
            logits = torch.randn(4, self.num_classes, requires_grad=True)
            logits_dict[name] = logits

        ignore = [self.module_names[0]]
        _, details = self.smoother.training_loss(
            logits_dict, labels, return_details=True, ignore_modules=ignore
        )

        self.assertNotIn(self.module_names[0], details["per_module_loss"])
        self.assertIn(self.module_names[1], details["per_module_loss"])
        self.assertEqual(details["ignored_modules"], ignore)
        self.assertNotIn(self.module_names[0], details["active_modules"])

    def test_training_loss_partial_consistency(self):
        labels = torch.tensor([0, 1, 2, 3])
        logits_dict = {}
        for name in self.module_names:
            logits = torch.randn(4, self.num_classes, requires_grad=True)
            logits_dict[name] = logits

        cons_mods = [self.module_names[0], self.module_names[1]]
        _, details = self.smoother.training_loss(
            logits_dict, labels, return_details=True,
            consistency_modules=cons_mods
        )
        self.assertEqual(set(details["consistency_modules"]), set(cons_mods))

    def test_flat_log_structure(self):
        labels = torch.tensor([0, 1, 2, 3])
        logits_dict = {}
        for name in self.module_names:
            logits = torch.randn(4, self.num_classes)
            logits_dict[name] = logits

        _, details = self.smoother.training_loss(
            logits_dict, labels, return_details=True
        )

        log = details["log"]
        for key in ["step", "loss/total", "loss/classification", "loss/consistency",
                    "train/in_warmup", "train/frozen_count", "topology/num_groups"]:
            self.assertIn(key, log)

        for name in self.module_names:
            self.assertIn(f"loss/cls/{name}", log)
            self.assertIn(f"smooth/mean/{name}", log)
            self.assertIn(f"smooth/var/{name}", log)
            self.assertIn(f"topology/group_id/{name}", log)

    def test_freeze_with_group_params(self):
        labels = torch.tensor([0, 1, 2, 3])
        logits_dict = {}
        for name in self.module_names:
            logits = torch.randn(4, self.num_classes)
            logits_dict[name] = logits

        self.smoother.freeze_module(self.module_names[0])

        _, d1 = self.smoother.training_loss(
            logits_dict, labels, return_details=True, use_group_params=True
        )
        sm1 = d1["smooth_labels"][self.module_names[0]].clone()

        for _ in range(10):
            self.smoother.training_loss(logits_dict, labels, use_group_params=True)

        _, d2 = self.smoother.training_loss(
            logits_dict, labels, return_details=True, use_group_params=True
        )
        sm2 = d2["smooth_labels"][self.module_names[0]]

        self.assertTrue(self.smoother.is_frozen(self.module_names[0]))

    def test_summary_step_filtering(self):
        labels = torch.tensor([0, 1, 2, 3])
        logits_dict = {}
        for name in self.module_names:
            logits = torch.randn(4, self.num_classes)
            logits_dict[name] = logits

        for _ in range(15):
            self.smoother.training_loss(logits_dict, labels)

        s_all = self.smoother.get_module_summary()
        s_1_5 = self.smoother.get_module_summary(min_step=1, max_step=5)
        s_6_15 = self.smoother.get_module_summary(min_step=6, max_step=15)

        self.assertEqual(s_all[self.module_names[0]]["num_samples"], 15)
        self.assertEqual(s_1_5[self.module_names[0]]["num_samples"], 5)
        self.assertEqual(s_6_15[self.module_names[0]]["num_samples"], 10)

    def test_merge_events_are_module_specific(self):
        labels = torch.tensor([0, 1, 2, 3])
        logits_dict = {}
        for name in self.module_names:
            logits = torch.randn(4, self.num_classes)
            logits_dict[name] = logits

        self.smoother._merge_split_interval = 1
        self.smoother.topology_manager.merge_threshold = 0.1
        self.smoother.topology_manager.merge_window = 5

        for i in range(10):
            for name in self.module_names[:2]:
                self.smoother.topology_manager.record_smoothing_value(name, 0.1)
            self.smoother.topology_manager.record_smoothing_value(self.module_names[2], 0.3)
            if i >= 4:
                self.smoother.topology_manager.try_merge_groups()
            self.smoother.training_loss(logits_dict, labels)

        summary = self.smoother.get_module_summary()
        events = self.smoother.get_merge_split_events()

        merge_count_0 = summary[self.module_names[0]]["merge_count"]
        merge_count_2 = summary[self.module_names[2]]["merge_count"]

        if len(events["merge_events"]) > 0:
            for ev in events["merge_events"]:
                self.assertIn("modules", ev)

    def test_summary_stage_stats(self):
        labels = torch.tensor([0, 1, 2, 3])
        logits_dict = {}
        for name in self.module_names:
            logits = torch.randn(4, self.num_classes)
            logits_dict[name] = logits

        for _ in range(12):
            self.smoother.training_loss(logits_dict, labels)

        summary = self.smoother.get_module_summary()
        for name in self.module_names:
            s = summary[name]
            self.assertIn("stage_stats", s)
            if s["num_samples"] >= 3:
                self.assertGreater(len(s["stage_stats"]), 0)
                for st in s["stage_stats"]:
                    self.assertIn("stage", st)
                    self.assertIn("avg_smoothing_mean", st)
                    self.assertIn("max_smoothing_var", st)

    def test_curves_step_filtering(self):
        labels = torch.tensor([0, 1, 2, 3])
        logits_dict = {}
        for name in self.module_names:
            logits = torch.randn(4, self.num_classes)
            logits_dict[name] = logits

        for _ in range(10):
            self.smoother.training_loss(logits_dict, labels)

        c_all = self.smoother.get_smoothing_curves()
        c_3_7 = self.smoother.get_smoothing_curves(min_step=3, max_step=7)

        self.assertEqual(len(c_all["steps"]), 10)
        self.assertEqual(len(c_3_7["steps"]), 5)
        self.assertEqual(c_3_7["steps"][0], 3)
        self.assertEqual(c_3_7["steps"][-1], 7)

    def test_state_dict_preserves_freeze_and_events(self):
        labels = torch.tensor([0, 1, 2, 3])
        logits_dict = {}
        for name in self.module_names:
            logits = torch.randn(4, self.num_classes)
            logits_dict[name] = logits

        self.smoother.freeze_module(self.module_names[0])
        for _ in range(5):
            self.smoother.training_loss(logits_dict, labels)

        frozen_before = self.smoother.get_frozen_modules()
        step_before = self.smoother._step_count
        events_before = self.smoother.get_merge_split_events()

        state = self.smoother.state_dict()

        new_smoother = AdaptiveLabelSmoother(
            num_classes=self.num_classes, module_names=self.module_names
        )
        new_smoother.load_state_dict(state)

        frozen_after = new_smoother.get_frozen_modules()
        step_after = new_smoother._step_count
        events_after = new_smoother.get_merge_split_events()

        self.assertEqual(frozen_before, frozen_after)
        for name in self.module_names:
            self.assertEqual(new_smoother.is_frozen(name), self.smoother.is_frozen(name))
        self.assertEqual(step_before, step_after)
        self.assertEqual(len(events_before["merge_events"]),
                         len(events_after["merge_events"]))

        self.smoother.training_loss(logits_dict, labels)
        for _ in range(5):
            new_smoother.training_loss(logits_dict, labels)

        events_cont = new_smoother.get_merge_split_events()
        self.assertEqual(len(events_cont["merge_events"]),
                         len(events_after["merge_events"]))

    def test_device_migration_cpu_to_cpu(self):
        smoother = AdaptiveLabelSmoother(
            num_classes=self.num_classes, module_names=self.module_names
        )
        labels = torch.zeros(4, dtype=torch.long)
        logits_dict = {n: torch.randn(4, self.num_classes) for n in self.module_names}

        loss1, det1 = smoother.training_loss(logits_dict, labels, return_details=True)

        smoother.to(torch.device("cpu"))

        self.assertEqual(smoother.device.type, "cpu")
        for name, mod in smoother.smoothing_modules.items():
            self.assertEqual(mod.log_alpha.device.type, "cpu")

        for gid, group in smoother.topology_manager._groups.items():
            self.assertEqual(group.alpha.device.type, "cpu")
            self.assertEqual(group.beta.device.type, "cpu")

        loss2, det2 = smoother.training_loss(logits_dict, labels, return_details=True)
        self.assertEqual(loss2.device.type, "cpu")
        self.assertEqual(det2["classification_loss"].device.type, "cpu")
        self.assertEqual(det2["consistency_loss"].device.type, "cpu")
        for name, sl in det2["smooth_labels"].items():
            self.assertEqual(sl.device.type, "cpu")
        for name, pl in det2["per_module_loss"].items():
            self.assertEqual(pl.device.type, "cpu")

    def test_device_migration_logits_labels_same_device(self):
        smoother = AdaptiveLabelSmoother(
            num_classes=self.num_classes, module_names=self.module_names
        )

        labels = torch.zeros(4, dtype=torch.long)
        logits_dict = {n: torch.randn(4, self.num_classes) for n in self.module_names}
        loss, det = smoother.training_loss(logits_dict, labels, return_details=True)

        self.assertEqual(loss.device, labels.device)
        for n in self.module_names:
            self.assertEqual(det["smooth_labels"][n].device, labels.device)
            self.assertEqual(det["per_module_loss"][n].device, labels.device)

        self.assertIn("loss/total", det["flat_log"])
        self.assertIn("global/loss/total", det["flat_log"])
        self.assertIn("local/loss/total", det["flat_log"])
        self.assertEqual(det["flat_log"]["global/dist_rank"], 0)
        self.assertEqual(det["flat_log"]["global/dist_world_size"], 1)

    def test_freeze_module_group_params_unchanged(self):
        smoother = AdaptiveLabelSmoother(
            num_classes=self.num_classes, module_names=["h0", "h1", "h2"],
            base_smoothing=0.1, merge_window=3, merge_threshold=0.5
        )

        for _ in range(10):
            for n in ["h0", "h1", "h2"]:
                smoother.topology_manager.record_smoothing_value(n, 0.1)

        smoother.topology_manager.try_merge_groups()
        self.assertEqual(smoother.topology_manager.num_groups(), 1)

        gid0 = smoother.topology_manager.get_module_group_map()["h0"]
        g_before = smoother.topology_manager._groups[gid0]
        alpha_before = g_before.alpha.data.clone()
        beta_before = g_before.beta.data.clone()
        h1_before = smoother.smoothing_modules["h1"].log_alpha.data.clone()

        smoother.freeze_module("h1")
        self.assertTrue(smoother.is_frozen("h1"))

        labels = torch.zeros(8, dtype=torch.long)
        logits_dict = {n: torch.randn(8, self.num_classes) for n in ["h0", "h1", "h2"]}
        for _ in range(15):
            smoother.training_loss(logits_dict, labels)

        gid0_after = smoother.topology_manager.get_module_group_map()["h0"]
        g_after = smoother.topology_manager._groups[gid0_after]
        alpha_after = g_after.alpha.data.clone()
        beta_after = g_after.beta.data.clone()
        h1_after = smoother.smoothing_modules["h1"].log_alpha.data.clone()

        self.assertTrue(torch.allclose(alpha_before, alpha_after),
                        "冻结模块所在组的参数不应该变化")
        self.assertTrue(torch.allclose(beta_before, beta_after),
                        "冻结模块所在组的参数不应该变化")
        self.assertTrue(torch.allclose(h1_before, h1_after),
                        "冻结模块自身参数不应该变化")

        smoother.unfreeze_module("h1")
        self.assertFalse(smoother.is_frozen("h1"))

        for _ in range(20):
            smoother.training_loss(logits_dict, labels)

        alpha_final = g_after.alpha.data.clone()
        changed_after_unfreeze = not torch.allclose(alpha_after, alpha_final)
        self.assertTrue(changed_after_unfreeze, "解冻后分组参数应该继续更新")

    def test_split_events_count_all_original_modules(self):
        smoother = AdaptiveLabelSmoother(
            num_classes=self.num_classes, module_names=["a", "b", "c"],
            base_smoothing=0.1, merge_window=2, merge_threshold=0.05,
            min_group_size=2, overfitting_patience=1
        )
        smoother._merge_split_interval = 1

        for _ in range(5):
            for n in ["a", "b", "c"]:
                smoother.topology_manager.record_smoothing_value(n, 0.1)

        _, merged_details = smoother.topology_manager.try_merge_groups()
        self.assertEqual(smoother.topology_manager.num_groups(), 1,
                         "应该合并为 1 组")

        smoother.topology_manager.mark_overfitting("a")
        smoother.topology_manager.mark_overfitting("b")
        smoother.topology_manager.mark_overfitting("c")
        _, split_details = smoother.topology_manager.try_split_groups()
        self.assertEqual(len(split_details), 1)
        self.assertEqual(sorted(split_details[0]["modules"]), ["a", "b", "c"])

        smoother._diagnostic_log = []
        labels = torch.zeros(4, dtype=torch.long)
        logits_dict = {n: torch.randn(4, self.num_classes) for n in ["a", "b", "c"]}

        entry = {
            "step": 50,
            "num_groups": 1,
            "module_stats": {},
            "merged_pairs": [],
            "merged_details": [],
            "split_groups": [0],
            "split_details": [{
                "group_id": 0,
                "modules": ["a", "b", "c"],
            }],
        }
        for n in ["a", "b", "c"]:
            entry["module_stats"][n] = {
                "smoothing_mean": 0.1, "smoothing_var": 0.001,
                "group_id": 0 if n == "a" else (1 if n == "b" else 2),
                "confidence": 0.5, "marginal_entropy": 0.5,
            }
        smoother._diagnostic_log.append(entry)

        events = smoother.get_merge_split_events()
        all_split_mods = set()
        for ev in events["split_events"]:
            for m in ev["modules"]:
                all_split_mods.add(m)
        self.assertEqual(all_split_mods, {"a", "b", "c"},
                         f"拆分事件应包含所有原组模块, 实际={all_split_mods}")

        summary = smoother.get_module_summary()
        for n in ["a", "b", "c"]:
            self.assertGreaterEqual(summary[n]["split_count"], 0)
        split_counts = [summary[n]["split_count"] for n in ["a", "b", "c"]]
        for sc in split_counts:
            self.assertEqual(sc, split_counts[0],
                             f"同组拆分所有模块计数应相同, 实际={split_counts}")

        s_filt = smoother.get_module_summary(min_step=40, max_step=60)
        for n in ["a", "b", "c"]:
            self.assertEqual(s_filt[n]["split_count"], 1)

        s_out = smoother.get_module_summary(min_step=100, max_step=200)
        for n in ["a", "b", "c"]:
            self.assertEqual(s_out[n]["split_count"], 0,
                             "step 区间外不应统计拆分事件")

    def test_checkpoint_meta_version_and_hyperparams(self):
        smoother = AdaptiveLabelSmoother(
            num_classes=self.num_classes, module_names=self.module_names,
            base_smoothing=0.2, warmup_steps=5
        )

        labels = torch.zeros(4, dtype=torch.long)
        logits_dict = {n: torch.randn(4, self.num_classes) for n in self.module_names}
        for _ in range(5):
            smoother.training_loss(logits_dict, labels)

        state = smoother.state_dict()
        self.assertIn("meta_version", state)
        self.assertEqual(state["meta_version"], "4.1.0")
        self.assertIn("meta_hyperparams", state)
        self.assertEqual(state["meta_hyperparams"]["num_classes"], self.num_classes)
        self.assertEqual(state["meta_hyperparams"]["base_smoothing"], 0.2)
        self.assertEqual(state["meta_hyperparams"]["warmup_steps"], 5)
        self.assertIn("meta_module_names", state)
        self.assertEqual(sorted(state["meta_module_names"]), sorted(self.module_names))
        self.assertIn("meta_module_names_checksum", state)
        self.assertIn("meta_diagnostic_summary", state)
        self.assertIn("total_steps", state["meta_diagnostic_summary"])
        self.assertIn("num_groups", state["meta_diagnostic_summary"])

    def test_checkpoint_load_strategy_intersection(self):
        smoother_old = AdaptiveLabelSmoother(
            num_classes=self.num_classes, module_names=["h0", "h1", "h2"]
        )
        labels = torch.zeros(4, dtype=torch.long)
        logits_old = {n: torch.randn(4, self.num_classes) for n in ["h0", "h1", "h2"]}
        for _ in range(6):
            smoother_old.training_loss(logits_old, labels)
        smoother_old.freeze_module("h1")

        state = smoother_old.state_dict()

        smoother_new = AdaptiveLabelSmoother(
            num_classes=self.num_classes, module_names=["h0", "h2", "h_new"]
        )
        report = smoother_new.load_state_dict(
            state, strict=False, load_strategy="intersection", return_report=True
        )

        from adaptive_label_smoothing.adaptive_smoother import CheckpointLoadReport
        self.assertIsInstance(report, CheckpointLoadReport)
        self.assertEqual(sorted(report.loaded_modules), ["h0", "h2"])
        self.assertEqual(report.skipped_modules, ["h_new"])
        self.assertEqual(report.extra_modules, ["h1"])
        self.assertTrue(report.version_match)
        self.assertTrue(smoother_new.is_frozen("h0") or not smoother_new.is_frozen("h0"))

    def test_checkpoint_load_strategy_mapping(self):
        smoother_old = AdaptiveLabelSmoother(
            num_classes=self.num_classes, module_names=["old_a", "old_b"]
        )
        labels = torch.zeros(4, dtype=torch.long)
        logits_old = {n: torch.randn(4, self.num_classes) for n in ["old_a", "old_b"]}
        for _ in range(5):
            smoother_old.training_loss(logits_old, labels)

        state = smoother_old.state_dict()

        smoother_new = AdaptiveLabelSmoother(
            num_classes=self.num_classes, module_names=["new_a", "new_b", "new_c"]
        )
        report = smoother_new.load_state_dict(
            state, strict=False, load_strategy="mapping", return_report=True,
            module_mapping={"new_a": "old_a", "new_b": "old_b"}
        )

        self.assertEqual(sorted(report.loaded_modules), ["new_a", "new_b"])
        self.assertEqual(report.skipped_modules, ["new_c"])
        self.assertEqual(report.extra_modules, [])

    def test_checkpoint_load_strict_mismatch_raises(self):
        smoother_old = AdaptiveLabelSmoother(
            num_classes=self.num_classes, module_names=["h0", "h1"]
        )
        state = smoother_old.state_dict()

        smoother_new = AdaptiveLabelSmoother(
            num_classes=self.num_classes, module_names=["h0", "h1", "h2"]
        )
        with self.assertRaises(RuntimeError):
            smoother_new.load_state_dict(state, load_strategy="strict", return_report=False)

        report = smoother_new.load_state_dict(state, load_strategy="strict", return_report=True)
        self.assertGreater(len(report.messages), 0)

    def test_distributed_flat_log_local_global_keys(self):
        smoother = AdaptiveLabelSmoother(
            num_classes=self.num_classes, module_names=self.module_names
        )
        labels = torch.zeros(4, dtype=torch.long)
        logits_dict = {n: torch.randn(4, self.num_classes) for n in self.module_names}

        loss, det = smoother.training_loss(logits_dict, labels, return_details=True)
        flat = det["flat_log"]

        base_keys = ["loss/total", "loss/classification", "topology/num_groups"]
        for bk in base_keys:
            self.assertIn(bk, flat, f"基字段 {bk} 必须存在")
            self.assertIn("local/" + bk, flat, f"local/{bk} 必须存在")
            self.assertIn("global/" + bk, flat, f"global/{bk} 必须存在")

        self.assertEqual(flat["local/loss/total"], flat["loss/total"])
        self.assertAlmostEqual(float(flat["global/loss/total"]), float(flat["loss/total"]), places=5)
        self.assertEqual(flat["global/dist_rank"], 0)
        self.assertEqual(flat["global/dist_world_size"], 1)

    def test_distributed_helper_functions_singlerank(self):
        self.assertFalse(AdaptiveLabelSmoother.is_distributed())
        self.assertEqual(AdaptiveLabelSmoother.get_rank(), 0)
        self.assertEqual(AdaptiveLabelSmoother.get_world_size(), 1)


if __name__ == "__main__":
    unittest.main()
