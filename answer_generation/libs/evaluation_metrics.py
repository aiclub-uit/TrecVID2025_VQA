"""Evaluation metrics for Video Question Answering."""

import numpy as np
from typing import Dict, List, Union

# Import evaluation metrics
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Warning: sentence-transformers not available for STS scoring")

try:
    from nltk.translate.meteor_score import meteor_score
    import nltk

    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
except ImportError:
    print("Warning: NLTK not available for METEOR scoring")

try:
    from evaluate import load

    bertscore_evaluator = load("bertscore")
except ImportError:
    print("Warning: evaluate library not available for BERTScore")
    bertscore_evaluator = None

try:
    from sklearn.metrics import ndcg_score
except ImportError:
    print("Warning: scikit-learn not available for NDCG scoring")

# Additional metrics imports
try:
    from rouge_score import rouge_scorer
except ImportError:
    print("Warning: rouge-score not available for ROUGE metrics")
    rouge_scorer = None


class EvaluationMetrics:
    """Class to handle various evaluation metrics."""

    def __init__(self):
        self.sts_model = None
        self.rouge_scorer = None
        self._init_sts_model()
        self._init_rouge_scorer()

    def _init_sts_model(self):
        """Initialize STS model for semantic similarity."""
        try:
            self.sts_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            print("Warning: Could not initialize STS model")

    def _init_rouge_scorer(self):
        """Initialize ROUGE scorer."""
        try:
            if rouge_scorer:
                self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        except Exception:
            print("Warning: Could not initialize ROUGE scorer")

    def compute_sts(self, predicted: str, reference: str) -> float:
        """Compute Semantic Textual Similarity score."""
        if not self.sts_model:
            return 0.0

        try:
            embeddings = self.sts_model.encode([predicted, reference])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception:
            return 0.0

    def compute_meteor(self, predicted: str, reference: str) -> float:
        """Compute METEOR score."""
        try:
            reference_tokens = reference.lower().split()
            predicted_tokens = predicted.lower().split()
            return meteor_score([reference_tokens], predicted_tokens)
        except Exception:
            return 0.0

    def compute_bertscore(self, predicted: str, reference: str) -> Dict[str, float]:
        """Compute BERTScore (Precision, Recall, F1)."""
        try:
            if bertscore_evaluator is None:
                print("Warning: BERTScore evaluator not available")
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

            # Ensure inputs are valid strings
            if not predicted or not reference:
                print("Warning: Empty predicted or reference text for BERTScore")
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

            # Compute BERTScore using the evaluate library
            results = bertscore_evaluator.compute(
                predictions=[predicted], references=[reference], lang="en"
            )

            # Check if results is None or missing expected keys
            if not results or not all(
                key in results for key in ["precision", "recall", "f1"]
            ):
                print("Warning: Invalid BERTScore results returned")
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

            # Check if the result lists are not empty
            if not results["precision"] or not results["recall"] or not results["f1"]:
                print("Warning: Empty BERTScore result lists")
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

            return {
                "precision": float(results["precision"][0]),
                "recall": float(results["recall"][0]),
                "f1": float(results["f1"][0]),
            }
        except Exception as e:
            print(f"Error computing BERTScore: {str(e)}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def compute_ndcg(
        self,
        predicted: str,
        correct_answer: str,
        incorrect_answers: List[str],
        k: int = 5,
    ) -> float:
        """Compute NDCG score based on answer ranking."""
        try:
            # Create relevance scores (1 for correct, 0 for incorrect)
            all_answers = [correct_answer] + incorrect_answers
            true_relevance = [1] + [0] * len(incorrect_answers)

            # Score predicted answer against all answers using STS
            predicted_scores = []
            for answer in all_answers:
                score = self.compute_sts(predicted, answer)
                predicted_scores.append(score)

            # Reshape for NDCG computation
            true_relevance = np.array([true_relevance])
            predicted_scores = np.array([predicted_scores])

            return float(
                ndcg_score(true_relevance, predicted_scores, k=min(k, len(all_answers)))
            )
        except Exception:
            return 0.0

    def compute_rouge(self, predicted: str, reference: str) -> Dict[str, float]:
        """Compute ROUGE scores."""
        if not self.rouge_scorer:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        try:
            scores = self.rouge_scorer.score(reference, predicted)
            return {
                "rouge1": scores['rouge1'].fmeasure,
                "rouge2": scores['rouge2'].fmeasure,
                "rougeL": scores['rougeL'].fmeasure,
            }
        except Exception:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    def compute_readability(self, text: str) -> Dict[str, float]:
        """Compute readability metrics."""
        try:
            # Import locally to handle missing dependencies
            from textstat import flesch_reading_ease, flesch_kincaid_grade
            return {
                "flesch_reading_ease": flesch_reading_ease(text),
                "flesch_kincaid_grade": flesch_kincaid_grade(text)
            }
        except Exception:
            return {"flesch_reading_ease": 0.0, "flesch_kincaid_grade": 0.0}

    def compute_length_metrics(self, predicted: str, reference: str) -> Dict[str, Union[int, float]]:
        """Compute length-based metrics."""
        pred_words = predicted.split()
        ref_words = reference.split()

        return {
            "predicted_length": len(pred_words),
            "reference_length": len(ref_words),
            "length_ratio": len(pred_words) / len(ref_words) if len(ref_words) > 0 else 0,
            "length_diff": abs(len(pred_words) - len(ref_words))
        }

    def compute_all_metrics(self, predicted: str, correct_answer: str, incorrect_answers: List[str] = None) -> Dict[str, float]:
        """
        Compute all available evaluation metrics for a prediction.
        
        Args:
            predicted: The predicted answer
            correct_answer: The correct reference answer
            incorrect_answers: List of incorrect answers for NDCG computation (optional)
            
        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {}
        
        try:
            # Basic semantic similarity
            metrics["sts_score"] = self.compute_sts(predicted, correct_answer)
            
            # METEOR score
            metrics["meteor_score"] = self.compute_meteor(predicted, correct_answer)
            
            # BERTScore metrics
            bertscore_results = self.compute_bertscore(predicted, correct_answer)
            metrics["bert_precision"] = bertscore_results["precision"]
            metrics["bert_recall"] = bertscore_results["recall"]
            metrics["bert_f1"] = bertscore_results["f1"]
            
            # NDCG score (if incorrect answers provided)
            if incorrect_answers is not None and len(incorrect_answers) > 0:
                metrics["ndcg_score"] = self.compute_ndcg(predicted, correct_answer, incorrect_answers)
            else:
                # Default NDCG calculation with empty incorrect answers list
                metrics["ndcg_score"] = self.compute_ndcg(predicted, correct_answer, [])
                
            # ROUGE scores
            rouge_results = self.compute_rouge(predicted, correct_answer)
            metrics["rouge1"] = rouge_results["rouge1"]
            metrics["rouge2"] = rouge_results["rouge2"]
            metrics["rougeL"] = rouge_results["rougeL"]
            
            # Length metrics
            length_results = self.compute_length_metrics(predicted, correct_answer)
            metrics.update(length_results)
            
            # Readability metrics
            readability_results = self.compute_readability(predicted)
            metrics.update(readability_results)
            
        except Exception as e:
            print(f"Warning: Error computing metrics: {e}")
            # Return basic structure with default values
            default_metrics = {
                "sts_score": 0.0,
                "meteor_score": 0.0,
                "bert_precision": 0.0,
                "bert_recall": 0.0,
                "bert_f1": 0.0,
                "ndcg_score": 0.0,
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
                "predicted_length": 0,
                "reference_length": 0,
                "length_ratio": 0.0,
                "length_diff": 0,
                "flesch_reading_ease": 0.0,
                "flesch_kincaid_grade": 0.0
            }
            metrics.update(default_metrics)
            
        return metrics
