"""Adversarial CPU-only H2-P3 unit tests using no model/tokenizer downloads."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts import rfr_h2_p3_tokenizer_continuation_conformance_cpu as p3


class TokenizerStub:
    """A synthetic tokenizer that gives all three leading-space answers 1 ID."""
    TOKENS = {" False": 15001, " Unknown": 15002, " True": 15003}

    def encode(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is False
        assert truncation is False
        for word, token in self.TOKENS.items():
            if text.endswith("Answer:" + word):
                return [ord(c) + 200 for c in text[:-len(word)]] + [token]
        return [ord(c) + 200 for c in text]


class BoundaryMergeTokenizer(TokenizerStub):
    def encode(self, text, *, add_special_tokens, truncation):
        if text.endswith("Answer: False"):
            return [ord(c) + 200 for c in text[:-len("Answer: False")]] + [15400]
        return super().encode(text, add_special_tokens=add_special_tokens,
                              truncation=truncation)


class MultiTokenTokenizer(TokenizerStub):
    def encode(self, text, *, add_special_tokens, truncation):
        if text.endswith("Answer: Unknown"):
            return [ord(c) + 200 for c in text[:-len(" Unknown")]] + [15300, 15301]
        return super().encode(text, add_special_tokens=add_special_tokens,
                              truncation=truncation)


class CollisionTokenizer(TokenizerStub):
    TOKENS = {" False": 15001, " Unknown": 15001, " True": 15003}


class ContextSensitiveTokenizer(TokenizerStub):
    def encode(self, text, *, add_special_tokens, truncation):
        result = super().encode(text, add_special_tokens=add_special_tokens,
                                truncation=truncation)
        if text.endswith("Answer: True") and "Query: Is B true" in text:
            result[-1] = 15004
        return result


class PrefixMergeTokenizer(TokenizerStub):
    def encode(self, text, *, add_special_tokens, truncation):
        result = super().encode(text, add_special_tokens=add_special_tokens,
                                truncation=truncation)
        if "Query:" in text and "Initial authenticated records" in text:
            result[0] = 24000
        return result


class TestPass(unittest.TestCase):
    def test_all_216_contexts_and_648_continuations(self):
        report = p3.audit_hypothetical_conformance(TokenizerStub(), vocab_size=50280)
        self.assertEqual(report["result"], p3.PASS_STATUS)
        self.assertEqual(report["prompts_checked"], 216)
        self.assertEqual(report["candidate_checks"], 648)
        self.assertEqual(report["hypothetical_boolean_strata"], 8)
        self.assertEqual(report["independently_sampled_roots"], 0)
        self.assertEqual(report["candidate_token_ids"],
                         {"REFUTE": 15001, "NOT_ENTITLED": 15002, "SUPPORT": 15003})
        self.assertEqual(report["complete_token_length_range"][0],
                         report["prompt_token_length_range"][0] + 1)
        self.assertEqual(report["lm_head_loaded"], False)
        self.assertEqual(report["model_evaluation"], False)
        self.assertEqual(report["actual_model_input_budget"], "NOT_VERIFIED")
        self.assertEqual(report["phase3b"], "BLOCKED_UNCHANGED")

    def test_original_prompt_scaffold_is_unchanged(self):
        from scripts import rfr_h2_p1_temporal_serialization_contract_cpu as p1
        self.assertTrue(p1._query_text("A").endswith("Answer: "))
        self.assertEqual(p1.READOUT_SPEC["candidate_continuations"],
                         "UNSET_NO_TOKENIZER")
        self.assertEqual(p3.CONTINUATIONS["REFUTE"], " False")
        self.assertEqual(p3.SCORING_RULE,
                         "ONE_NEXT_TOKEN_LOGIT_AT_FINAL_QUERY_POSITION_NO_EOS_NO_NORMALIZATION")


class TestBlockedMutants(unittest.TestCase):
    def blocked(self, tokenizer, code, vocab_size=50280):
        with self.assertRaisesRegex(p3.ConformanceError, "H2_P3_" + code):
            p3.audit_hypothetical_conformance(tokenizer, vocab_size=vocab_size)

    def test_boundary_merge_rejected(self):
        self.blocked(BoundaryMergeTokenizer(), "ANSWER_BOUNDARY_MISMATCH")

    def test_two_token_verbalizer_rejected(self):
        self.blocked(MultiTokenTokenizer(), "CONTINUATION_NOT_SINGLE_TOKEN")

    def test_duplicate_answer_token_rejected(self):
        self.blocked(CollisionTokenizer(), "CANDIDATE_TOKEN_COLLISION")

    def test_context_driven_answer_id_rejected(self):
        self.blocked(ContextSensitiveTokenizer(), "CANDIDATE_TOKEN_CONTEXT_DRIFT")

    def test_query_prefix_merge_rejected(self):
        self.blocked(PrefixMergeTokenizer(), "TOKEN_ANCHOR_MISMATCH")

    def test_small_model_vocabulary_rejected(self):
        self.blocked(TokenizerStub(), "CANDIDATE_OUTSIDE_MODEL_VOCAB", 15000)

    def test_invalid_vocab_type_rejected(self):
        self.blocked(TokenizerStub(), "VOCAB_SIZE", True)

    def test_special_token_as_answer_rejected(self):
        with self.assertRaisesRegex(p3.ConformanceError, "H2_P3_ANSWER_IS_SPECIAL_TOKEN"):
            p3.audit_hypothetical_conformance(
                TokenizerStub(), vocab_size=50280, reserved_token_ids=(15001,)
            )

    def test_invalid_special_id_declaration_rejected(self):
        with self.assertRaisesRegex(p3.ConformanceError, "H2_P3_SPECIAL_TOKEN_SCHEMA"):
            p3.audit_hypothetical_conformance(
                TokenizerStub(), vocab_size=50280, reserved_token_ids=(True,)
            )

    def test_label_reordering_rejected(self):
        with patch.dict(p3.p1.READOUT_SPEC,
                        {"label_order": ("SUPPORT", "NOT_ENTITLED", "REFUTE")}):
            self.blocked(TokenizerStub(), "P1_LABEL_CONTRACT_DRIFT")

    def test_early_readout_promotion_rejected(self):
        with patch.dict(p3.p1.READOUT_SPEC, {"evaluation_enabled": True}):
            self.blocked(TokenizerStub(), "P1_READOUT_PROMOTION")

    def test_changed_scaffold_rejected(self):
        def corrupt(query):
            return f"Query: {query}\nResponse: "
        with patch.object(p3.p1, "_query_text", side_effect=corrupt):
            self.blocked(TokenizerStub(), "P1_QUERY_SCAFFOLD_CHANGED")

    def test_continuation_contract_rejected(self):
        with patch.dict(p3.CONTINUATIONS, {"REFUTE": "FALSE"}):
            self.blocked(TokenizerStub(), "CONTINUATION_CONTRACT")


if __name__ == "__main__":
    unittest.main()
