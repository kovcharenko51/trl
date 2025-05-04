import numpy as np
import re

from collections import defaultdict
from typing import Any

from scipy.stats import entropy
from transformers import PreTrainedTokenizerBase
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


class Metric:
    def name(self):
        return re.sub(r'(?<!^)(?=[A-Z])', '_', str(self.__class__.__name__)).lower()

class DiversityMetric(Metric):
    def __init__(self, top_k) -> None:
        self._top_k = top_k

    def compute(self, **kwargs) -> dict:
        tokenizer: PreTrainedTokenizerBase = kwargs.get('tokenizer', None)
        predictions: list[list[str]] = kwargs.get('predictions', None)

        return {self.name(): self.average_token_entropy(predictions, tokenizer, self._top_k).item()}

    def average_token_entropy(
        self, answer_group: list[str], tokenizer: PreTrainedTokenizerBase, top_k: int | None
    ) -> float:
        entropies = [self.token_entropy(answer, tokenizer, top_k) for answer in answer_group]
        if entropies:
            return sum(entropies) / len(entropies)

        return np.nan

    @staticmethod
    def token_entropy(sample: str, tokenizer: PreTrainedTokenizerBase, top_k: int | None) -> float:
        stats: dict[int, Any] = defaultdict(int)
        num_tokens = 0
        tokens = tokenizer.encode(sample)
        
        for t in tokens:
            if t == tokenizer.pad_token_id:
                continue
            stats[t] += 1
            num_tokens += 1
        
        for k in stats.keys():
            stats[k] /= num_tokens

        top_k_stats = list(stats.values())
        if top_k is not None:
            top_k_stats = sorted(top_k_stats, reverse=True)[:top_k]

        return entropy(top_k_stats)


class DistinctnessMetric(Metric):
    def __init__(self, ngram) -> None:
        self._ngram = ngram

    def compute(self, **kwargs) -> dict:
        predictions: list[list[str]] = kwargs.get('predictions', None)
        tokenizer: PreTrainedTokenizerBase = kwargs.get('tokenizer', None)
        
        vocab_size: int = tokenizer.vocab_size

        ans_dist_n = self.distinctness(predictions, vocab_size, self._ngram)
        metric_name = self.name()

        return {
            f"{metric_name}_{label}": value for label, value in ans_dist_n.items()
        }

    @staticmethod
    def distinctness(answers: list[str], vocab_size: int, ngram: int) -> dict[str, float]:
        ngram_sets: list[set] = [set() for _ in range(ngram)]
        total_ngrams: list[int] = [0] * ngram

        for answer in answers:
            words = answer.split(' ')
            ngram_sets[0].update(words)
            total_ngrams[0] += len(words)

            for n in range(1, ngram):
                ngrams = ['_'.join(words[i : i + n + 1]) for i in range(len(words) - n)]
                ngram_sets[n].update(ngrams)
                total_ngrams[n] += len(ngrams)

        result = {}
        for n in range(ngram):
            result[f'dist_{n+1}'] = len(ngram_sets[n]) / total_ngrams[n] if total_ngrams[n] > 0 else 0
            try:
                result[f'ead_dist_{n+1}'] = (
                    len(ngram_sets[n]) / (vocab_size * (1 - ((vocab_size - 1) / vocab_size) ** total_ngrams[n]))
                    if total_ngrams[n] > 0
                    else 0
                )
            except ZeroDivisionError:
                result[f'ead_dist_{n+1}'] = 0

        result['dist_mean'] = sum(result[f'dist_{n+1}'] for n in range(ngram)) / ngram
        result['ead_dist_mean'] = sum(result[f'ead_dist_{n+1}'] for n in range(ngram)) / ngram
        
        return {k: result[k] for k in ['dist_mean', 'ead_dist_mean']}
    

class SelfBleuMetric(Metric):
    def __init__(self, ngram) -> None:
        self._ngram = ngram

    def compute(self, **kwargs) -> dict:
        predictions: list[list[str]] = kwargs.get('predictions', None)

        return {self.name(): self.self_bleu(predictions)}

    def self_bleu(self, answers: list[str]) -> float:
        weight = tuple((1.0 / self._ngram for _ in range(self._ngram)))
        result = []
        sentence_num = len(answers)
        for index in range(sentence_num):
            hypothesis = answers[index]
            other = answers[:index] + answers[index + 1 :]
            result.append(self._calc_bleu(other, hypothesis, weight))
        return np.mean(result).item()

    def _calc_bleu(self, reference: list[str], hypothesis: str, weight: tuple[float, ...]) -> list[float]:
        return sentence_bleu(reference, hypothesis, weight, smoothing_function=SmoothingFunction().method1)