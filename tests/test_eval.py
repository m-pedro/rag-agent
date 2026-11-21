import math
from rag_agent.eval import context_precision_at_k, ndcg_at_k, recall_at_k, reciprocal_rank


def test_ir_and_rag_evaluation_metrics_exact_values():
    retrieved = ["c_irrel", "c_gold1", "c_gold2"]
    relevant = {"c_gold1", "c_gold2"}
    graded = {"c_gold1": 2.0, "c_gold2": 1.0}

    assert math.isclose(recall_at_k(retrieved, relevant, k=2), 0.5)
    assert math.isclose(recall_at_k(retrieved, relevant, k=3), 1.0)
    assert math.isclose(reciprocal_rank(retrieved, relevant), 0.5)  # First hit at rank 2

    # Perfect ordering should yield NDCG == 1.0 and Context Precision == 1.0
    perfect = ["c_gold1", "c_gold2", "c_irrel"]
    assert math.isclose(ndcg_at_k(perfect, graded, k=3), 1.0)
    assert math.isclose(context_precision_at_k(perfect, relevant, k=2), 1.0)
    assert ndcg_at_k(retrieved, graded, k=3) < 1.0
