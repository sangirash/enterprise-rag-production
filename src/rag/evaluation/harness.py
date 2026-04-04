"""
Evaluation harness: runs a dataset of queries through the RAG pipeline and collects metrics.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import structlog

from .metrics import EvalResult, evaluate_faithfulness, evaluate_context_precision, evaluate_answer_relevance

logger = structlog.get_logger()


@dataclass
class EvalCase:
    query: str
    expected_answer: str | None = None
    context_chunks: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    results: list[EvalResult]

    @property
    def mean_faithfulness(self) -> float:
        return sum(r.faithfulness for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def mean_context_precision(self) -> float:
        return sum(r.context_precision for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def mean_answer_relevance(self) -> float:
        return sum(r.answer_relevance for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def mean_overall(self) -> float:
        return sum(r.overall for r in self.results) / len(self.results) if self.results else 0.0

    def print_table(self) -> None:
        col_q = 42
        header = f"{'Query':<{col_q}} {'Faith':>7} {'Prec':>7} {'Relev':>7} {'Overall':>8}"
        sep = "-" * len(header)
        print(header)
        print(sep)
        for r in self.results:
            q = (r.query[: col_q - 2] + "..") if len(r.query) > col_q else r.query
            print(
                f"{q:<{col_q}} {r.faithfulness:>7.3f} {r.context_precision:>7.3f}"
                f" {r.answer_relevance:>7.3f} {r.overall:>8.3f}"
            )
        print(sep)
        print(
            f"{'MEAN':<{col_q}} {self.mean_faithfulness:>7.3f}"
            f" {self.mean_context_precision:>7.3f}"
            f" {self.mean_answer_relevance:>7.3f}"
            f" {self.mean_overall:>8.3f}"
        )


class EvaluationHarness:
    """Runs eval cases through the pipeline and aggregates metric scores."""

    def __init__(self, pipeline: Any | None = None) -> None:
        if pipeline is None:
            from ..pipeline import RAGPipeline
            pipeline = RAGPipeline()
        self.pipeline = pipeline

    async def run(self, cases: list[EvalCase]) -> EvalReport:
        results: list[EvalResult] = []
        for case in cases:
            try:
                result = await self.pipeline.query(case.query)
                answer: str = result["answer"]
                context_chunks: list[str] = (
                    [s.get("metadata", {}).get("content", "") for s in result.get("sources", [])]
                    or case.context_chunks
                )
                faithfulness = evaluate_faithfulness(answer, context_chunks)
                precision = evaluate_context_precision(case.query, context_chunks)
                relevance = evaluate_answer_relevance(case.query, answer)
                eval_result = EvalResult(
                    query=case.query,
                    answer=answer,
                    faithfulness=faithfulness,
                    context_precision=precision,
                    answer_relevance=relevance,
                )
                results.append(eval_result)
                logger.info(
                    "eval_case_complete",
                    query=case.query,
                    overall=round(eval_result.overall, 3),
                )
            except Exception as exc:
                logger.error("eval_case_failed", query=case.query, error=str(exc))
        return EvalReport(results=results)
