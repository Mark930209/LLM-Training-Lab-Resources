import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import torch

from exp_eval import eval_lab
from exp_eval.eval_common import (
    build_mini_eval,
    evaluation_set_sha256,
    ngram_contamination,
    predict_generation,
    predict_likelihood,
    score_generation,
    score_likelihood,
)


class LikelihoodTokenizer:
    token_ids = {"P": 1, "Q": 2, "A": 3, "B": 4}

    def encode(self, text):
        return [self.token_ids[character] for character in text]


class PrefixBiasedModel:
    def __call__(self, input_ids):
        logits = torch.full((1, input_ids.shape[1], 5), -100.0)
        for position in range(input_ids.shape[1] - 1):
            target_id = input_ids[0, position + 1].item()
            target_score = -10.0 if position == 0 else (-1.0 if target_id == 3 else -2.0)
            logits[0, position, 0] = 0.0
            logits[0, position, target_id] = target_score
        return type("Output", (), {"logits": logits})()


class UnicodeTokenizer:
    def encode(self, text):
        return [ord(character) for character in text]

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


class ChoiceGenerationModel:
    def __init__(self, answer):
        self.answer = answer
        self.prompt_ids = None

    def generate(self, input_ids, **kwargs):
        self.prompt_ids = input_ids[0].tolist()
        answer = torch.tensor([[ord(self.answer)]], device=input_ids.device)
        return torch.cat((input_ids, answer), dim=1)


class EvalCommonTests(unittest.TestCase):
    def test_classification_answers_are_balanced_by_position(self):
        samples = [sample for sample in build_mini_eval()
                   if sample["task"] == "classification"]
        positions = Counter(sample["answer"] for sample in samples)
        self.assertEqual(positions, Counter({0: 5, 1: 5, 2: 5, 3: 5}))

    def test_likelihood_excludes_prompt_tokens(self):
        sample = {
            "prompt": "PQ",
            "options": ["A", "BB"],
            "answer": 0,
        }
        self.assertEqual(
            score_likelihood(PrefixBiasedModel(), LikelihoodTokenizer(), sample, "cpu"),
            1,
        )

    def test_likelihood_prediction_is_auditable_and_matches_score(self):
        sample = {
            "prompt": "PQ",
            "options": ["A", "BB"],
            "answer": 0,
        }
        prediction = predict_likelihood(
            PrefixBiasedModel(), LikelihoodTokenizer(), sample, "cpu"
        )
        self.assertEqual(prediction["prediction"], 0)
        self.assertEqual(prediction["expected"], 0)
        self.assertEqual(prediction["correct"], 1)
        self.assertEqual(
            score_likelihood(PrefixBiasedModel(), LikelihoodTokenizer(), sample, "cpu"),
            prediction["correct"],
        )

    def test_classification_generation_includes_choices_and_scores_label(self):
        sample = {
            "task": "classification",
            "prompt": "Which color?",
            "options": ["red", "blue", "green", "black"],
            "answer": 2,
        }
        tokenizer = UnicodeTokenizer()
        model = ChoiceGenerationModel("C")
        self.assertEqual(score_generation(model, tokenizer, sample, "cpu"), 1)
        prompt = tokenizer.decode(model.prompt_ids)
        self.assertIn("A. red", prompt)
        self.assertIn("C. green", prompt)
        self.assertIn("只输出选项字母", prompt)

    def test_generation_prediction_records_label_and_raw_output(self):
        sample = {
            "task": "classification",
            "prompt": "Which color?",
            "options": ["red", "blue", "green", "black"],
            "answer": 2,
        }
        prediction = predict_generation(
            ChoiceGenerationModel("C"), UnicodeTokenizer(), sample, "cpu"
        )
        self.assertEqual(prediction["prediction"], "C")
        self.assertEqual(prediction["expected"], "C")
        self.assertEqual(prediction["raw_output"], "C")
        self.assertEqual(prediction["correct"], 1)

    def test_evaluation_set_hash_is_stable_across_dict_key_order(self):
        first = [{"task": "completion", "prompt": "a", "expected": "b"}]
        reordered = [{"expected": "b", "task": "completion", "prompt": "a"}]
        digest = evaluation_set_sha256(first)
        self.assertEqual(digest, evaluation_set_sha256(reordered))
        self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_prompt_shorter_than_ngram_is_unscannable(self):
        result = ngram_contamination(
            [{"task": "completion", "prompt": "abc"}], "abc", n=8
        )
        self.assertEqual(result["n_scannable"], 0)
        self.assertEqual(result["n_unscannable"], 1)
        self.assertEqual(result["samples"][0]["total_ngrams"], 0)
        self.assertFalse(result["samples"][0]["contaminated"])

    def test_seed_variance_is_not_reported_as_confidence_interval(self):
        samples = [
            {"task": "completion", "prompt": "a", "expected": "b"},
            {"task": "arithmetic", "prompt": "1 + 1 = ", "expected": "2"},
        ]

        def score_by_seed(_model, _tokenizer, _sample, _device, seed):
            return seed % 2

        with patch.object(eval_lab, "load_model_and_tokenizer", return_value=(None, None)), \
             patch.object(eval_lab, "build_mini_eval", return_value=samples), \
             patch.object(eval_lab, "score_generation", side_effect=score_by_seed):
            result = eval_lab.mode_variance(SimpleNamespace(ckpt="checkpoint.pt", seeds=3))

        for task_result in result["per_task"].values():
            self.assertNotIn("ci95", task_result)
            self.assertEqual(task_result["seed_range"], [0.0, 1.0])
        self.assertIn("不是总体置信区间", result["note"])

    def test_caliber_report_includes_question_results_and_metadata(self):
        samples = [
            {"task": "classification", "prompt": "p", "options": ["a", "b"], "answer": 1},
            {"task": "completion", "prompt": "c", "expected": "x"},
            {"task": "arithmetic", "prompt": "1+1=", "expected": "2"},
        ]
        generation_results = [
            {"prediction": "B", "expected": "B", "raw_output": "B", "correct": 1},
            {"prediction": "x", "expected": "x", "raw_output": "x", "correct": 1},
            {"prediction": "2", "expected": "2", "raw_output": "2", "correct": 1},
        ]
        model = SimpleNamespace(parameters=lambda: [])
        tokenizer = SimpleNamespace(vocab_size=5)
        with patch.object(eval_lab, "load_model_and_tokenizer", return_value=(model, tokenizer)), \
             patch.object(eval_lab, "build_mini_eval", return_value=samples), \
             patch.object(eval_lab, "predict_likelihood", return_value={
                 "prediction": 1, "expected": 1, "correct": 1,
             }), \
             patch.object(eval_lab, "predict_generation", side_effect=generation_results), \
             patch.object(eval_lab.torch.cuda, "is_available", return_value=False):
            result = eval_lab.mode_caliber(SimpleNamespace(ckpt="checkpoint.pt"))

        self.assertEqual(result["results"]["classification_likelihood"]["acc"], 1.0)
        self.assertEqual(len(result["question_results"]), 4)
        self.assertEqual(result["metadata"]["evaluation_set"]["rows"], 3)
        self.assertEqual(
            result["metadata"]["evaluation_set"]["sha256"],
            evaluation_set_sha256(samples),
        )
        self.assertEqual(result["metadata"]["hardware"]["device"], "CPU")
        self.assertEqual(
            result["metadata"]["peak_memory_mib"],
            {"allocated": None, "reserved": None},
        )
        self.assertEqual(result["metadata"]["tokenizer"]["vocab_size"], 5)
        self.assertIn("eval_lab.py", result["metadata"]["code_sha256"])
        self.assertGreaterEqual(result["metadata"]["timing_seconds"]["total"], 0)

    def test_caliber_report_records_cuda_peak_memory(self):
        samples = [
            {"task": "classification", "prompt": "p", "options": ["a", "b"], "answer": 1},
            {"task": "completion", "prompt": "c", "expected": "x"},
            {"task": "arithmetic", "prompt": "1+1=", "expected": "2"},
        ]
        generation_results = [
            {"prediction": "B", "expected": "B", "raw_output": "B", "correct": 1},
            {"prediction": "x", "expected": "x", "raw_output": "x", "correct": 1},
            {"prediction": "2", "expected": "2", "raw_output": "2", "correct": 1},
        ]
        model = SimpleNamespace(parameters=lambda: [])
        tokenizer = SimpleNamespace(vocab_size=5)
        with patch.object(eval_lab, "load_model_and_tokenizer", return_value=(model, tokenizer)), \
             patch.object(eval_lab, "build_mini_eval", return_value=samples), \
             patch.object(eval_lab, "predict_likelihood", return_value={
                 "prediction": 1, "expected": 1, "correct": 1,
             }), \
             patch.object(eval_lab, "predict_generation", side_effect=generation_results), \
             patch.object(eval_lab.torch.cuda, "is_available", return_value=True), \
             patch.object(eval_lab.torch.cuda, "get_device_name", return_value="Test GPU"), \
             patch.object(eval_lab.torch.cuda, "reset_peak_memory_stats") as reset_peak, \
             patch.object(eval_lab.torch.cuda, "max_memory_allocated", return_value=8 * 1024 ** 2), \
             patch.object(eval_lab.torch.cuda, "max_memory_reserved", return_value=12 * 1024 ** 2), \
             patch.object(eval_lab.torch.cuda, "synchronize"):
            result = eval_lab.mode_caliber(SimpleNamespace(ckpt="checkpoint.pt"))

        reset_peak.assert_called_once_with()
        self.assertEqual(
            result["metadata"]["peak_memory_mib"],
            {"allocated": 8.0, "reserved": 12.0},
        )

    def test_report_fields_match_current_evidence(self):
        with patch("builtins.print"):
            result = eval_lab.mode_report(SimpleNamespace())
        fields = result["fields"]
        self.assertIn("question_results", fields)
        self.assertIn("evaluation_code", fields)
        self.assertIn("peak_memory", fields)
        self.assertIn("timing", fields)
        self.assertIn("样本标准差和观测范围", fields["metrics"])
        self.assertIn("不作为总体置信区间", fields["metrics"])


if __name__ == "__main__":
    unittest.main()