from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import gc


# Benchmark-informed priors from the isolated 5-minute Nasdaq walk-forward run.
# Heavy models are shadow-only by default, so these weights are not allowed to
# alter PAPER trade decisions until shadow mode is deliberately promoted.
WEIGHTS = {
    "technical": 0.35,
    "chronos2": 0.20,
    "timesfm2_5": 0.35,
    "finbert": 0.10,
}

BENCHMARK_PRIORS = {
    "timesfm2_5": {
        "spy_skill_vs_last": 0.20035273058870362,
        "spx_skill_vs_last": 0.23264975083769746,
        "spy_direction_accuracy": 8 / 12,
        "spx_direction_accuracy": 8 / 12,
        "samples_per_symbol": 12,
    },
    "chronos2": {
        "spy_skill_vs_last": 0.15446423546532173,
        "spx_skill_vs_last": 0.144919281527071,
        "spy_direction_accuracy": 7 / 12,
        "spx_direction_accuracy": 7 / 12,
        "samples_per_symbol": 12,
    },
}


@dataclass(frozen=True)
class ModelVote:
    source: str
    score: float
    confidence: float
    detail: str = ""

    def __post_init__(self):
        object.__setattr__(self, "score", max(-1.0, min(1.0, float(self.score))))
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EnsembleResult:
    kind: str
    confidence: float
    score: float
    votes: tuple[ModelVote, ...]
    degraded: bool
    reason: str

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "confidence": self.confidence,
            "score": self.score,
            "votes": [vote.as_dict() for vote in self.votes],
            "degraded": self.degraded,
            "reason": self.reason,
        }


def forecast_vote(source: str, spot: float, predicted: float) -> ModelVote:
    spot, predicted = float(spot), float(predicted)
    if spot <= 0 or predicted <= 0:
        return ModelVote(source, 0.0, 0.0, "invalid forecast")
    change = predicted / spot - 1.0
    confidence = min(1.0, abs(change) * 100.0)
    score = 1.0 if change > 0 else (-1.0 if change < 0 else 0.0)
    return ModelVote(source, score, confidence, f"forecast return {change * 100:.3f}%")


def _technical_score(kind: str) -> float:
    if kind == "call":
        return 1.0
    if kind == "put":
        return -1.0
    return 0.0


def combine_votes(technical_kind: str, technical_strength: float,
                  votes: list[ModelVote] | tuple[ModelVote, ...], threshold: float = 0.18) -> EnsembleResult:
    if technical_kind not in {"call", "put", "hold"}:
        technical_kind = "hold"
    technical_strength = max(0.0, min(1.0, float(technical_strength)))
    ai_votes = tuple(vote for vote in votes if vote.source in WEIGHTS and vote.source != "technical")
    if not ai_votes:
        score = _technical_score(technical_kind) * technical_strength
        return EnsembleResult(technical_kind, technical_strength, score, (), True,
                              "AI models unavailable; preserved technical signal.")

    weighted = WEIGHTS["technical"] * _technical_score(technical_kind) * technical_strength
    active_weight = WEIGHTS["technical"]
    for vote in ai_votes:
        weight = WEIGHTS[vote.source]
        weighted += weight * vote.score * vote.confidence
        active_weight += weight
    score = weighted / active_weight if active_weight else 0.0
    if score >= threshold:
        kind = "call"
    elif score <= -threshold:
        kind = "put"
    else:
        kind = "hold"
    return EnsembleResult(kind, min(1.0, abs(score)), score, ai_votes, False,
                          f"ensemble score={score:.3f} from {len(ai_votes)} AI vote(s)")


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _release_runtime(device: str) -> None:
    gc.collect()
    try:
        import torch
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class Chronos2Adapter:
    source = "chronos2"
    release_after_vote = True
    repo_id = "amazon/chronos-2"

    def __init__(self, cache_dir: Path, device: str):
        self.cache_dir, self.device = Path(cache_dir), _resolve_device(device)
        self.pipeline = None
        self.torch = None

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None

    def release(self):
        self.pipeline = None
        self.torch = None
        _release_runtime(self.device)

    def warmup(self):
        if self.pipeline is None:
            import torch
            from chronos import Chronos2Pipeline
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.pipeline = Chronos2Pipeline.from_pretrained(
                self.repo_id, device_map=self.device, cache_dir=str(self.cache_dir)
            )
            self.torch = torch
        return self

    def vote(self, snapshot: dict) -> ModelVote | None:
        closes = [float(x) for x in snapshot.get("closes_5m", []) if float(x) > 0]
        if len(closes) < 20:
            return None
        self.warmup()
        torch = self.torch
        series = torch.tensor(closes, dtype=torch.float32)
        with torch.no_grad():
            _, mean = self.pipeline.predict_quantiles(
                inputs=[series],
                prediction_length=6,
                quantile_levels=[0.1, 0.5, 0.9],
                batch_size=1,
            )
        values = mean[0].detach().float().cpu().flatten()
        if values.numel() == 0:
            return None
        predicted = float(values[min(5, values.numel() - 1)].item())
        return forecast_vote(self.source, closes[-1], predicted)


class TimesFM25Adapter:
    source = "timesfm2_5"
    release_after_vote = True
    repo_id = "google/timesfm-2.5-200m-transformers"

    def __init__(self, cache_dir: Path, device: str):
        self.cache_dir, self.device = Path(cache_dir), _resolve_device(device)
        self.model = None
        self.torch = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def release(self):
        self.model = None
        self.torch = None
        _release_runtime(self.device)

    def warmup(self):
        if self.model is None:
            import torch
            from transformers import TimesFm2_5ModelForPrediction
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            model = TimesFm2_5ModelForPrediction.from_pretrained(self.repo_id, cache_dir=str(self.cache_dir))
            self.model = model.to(self.device).eval()
            self.torch = torch
        return self

    def vote(self, snapshot: dict) -> ModelVote | None:
        closes = [float(x) for x in snapshot.get("closes_5m", []) if float(x) > 0][-1024:]
        if len(closes) < 20:
            return None
        self.warmup()
        torch = self.torch
        past = torch.tensor(closes, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            outputs = self.model(past_values=[past], forecast_context_len=len(closes))
        values = outputs.mean_predictions[0].detach().float().cpu().flatten()
        if values.numel() == 0:
            return None
        predicted = float(values[min(5, values.numel() - 1)].item())
        return forecast_vote(self.source, closes[-1], predicted)


class FinBERTAdapter:
    source = "finbert"
    release_after_vote = True
    repo_id = "ProsusAI/finbert"

    def __init__(self, cache_dir: Path, device: str):
        self.cache_dir, self.device = Path(cache_dir), _resolve_device(device)
        self.pipe = None

    @property
    def loaded(self) -> bool:
        return self.pipe is not None

    def release(self):
        self.pipe = None
        _release_runtime(self.device)

    def warmup(self):
        if self.pipe is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tokenizer = AutoTokenizer.from_pretrained(self.repo_id, cache_dir=str(self.cache_dir))
            model = AutoModelForSequenceClassification.from_pretrained(self.repo_id, cache_dir=str(self.cache_dir))
            model = model.to(self.device).eval()
            self.pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=self.device)
        return self

    def vote(self, snapshot: dict) -> ModelVote | None:
        headlines = [str(x).strip() for x in snapshot.get("headlines", []) if str(x).strip()][:8]
        if not headlines:
            return None
        self.warmup()
        results = self.pipe(headlines, truncation=True)
        signed = []
        for item in results:
            label = str(item.get("label", "")).casefold()
            confidence = float(item.get("score", 0.0))
            if "positive" in label:
                signed.append(confidence)
            elif "negative" in label:
                signed.append(-confidence)
            else:
                signed.append(0.0)
        if not signed:
            return None
        average = sum(signed) / len(signed)
        score = 1.0 if average > 0 else (-1.0 if average < 0 else 0.0)
        return ModelVote(self.source, score, min(1.0, abs(average)),
                         f"headline sentiment {average:+.3f} across {len(signed)} headline(s)")


class OptionsModelEnsemble:
    def __init__(self, adapters=None, *, shadow_mode: bool = True):
        self.adapters = list(adapters or [])
        self.shadow_mode = bool(shadow_mode)
        self.last_errors: dict[str, str] = {}

    def evaluate(self, snapshot: dict, technical_kind: str, technical_strength: float) -> EnsembleResult:
        votes: list[ModelVote] = []
        self.last_errors = {}
        for adapter in self.adapters:
            source = getattr(adapter, "source", adapter.__class__.__name__)
            try:
                vote = adapter.vote(snapshot)
                if isinstance(vote, ModelVote):
                    votes.append(vote)
            except Exception as exc:
                self.last_errors[str(source)] = f"{exc.__class__.__name__}: {str(exc)[:160]}"
            finally:
                if getattr(adapter, "release_after_vote", False) and hasattr(adapter, "release"):
                    try:
                        adapter.release()
                    except Exception:
                        pass
        return combine_votes(technical_kind, technical_strength, votes)

    def warmup(self) -> dict:
        self.last_errors = {}
        verified: list[str] = []
        for adapter in self.adapters:
            source = getattr(adapter, "source", adapter.__class__.__name__)
            try:
                adapter.warmup()
                verified.append(str(source))
            except Exception as exc:
                self.last_errors[str(source)] = f"{exc.__class__.__name__}: {str(exc)[:160]}"
            finally:
                if getattr(adapter, "release_after_vote", False) and hasattr(adapter, "release"):
                    try:
                        adapter.release()
                    except Exception:
                        pass
        result = self.status()
        result["verified"] = verified
        return result

    def status(self) -> dict:
        return {
            "available": [getattr(adapter, "source", adapter.__class__.__name__) for adapter in self.adapters],
            "loaded": [getattr(adapter, "source", adapter.__class__.__name__)
                       for adapter in self.adapters if bool(getattr(adapter, "loaded", False))],
            "errors": dict(self.last_errors),
            "shadow_mode": self.shadow_mode,
            "mode": "shadow" if self.shadow_mode else "active",
            "benchmark_priors": dict(BENCHMARK_PRIORS),
        }


def build_default_ensemble(cache_dir: Path, device: str = "auto") -> OptionsModelEnsemble:
    cache_dir = Path(cache_dir)
    return OptionsModelEnsemble([
        Chronos2Adapter(cache_dir / "chronos2", device),
        TimesFM25Adapter(cache_dir / "timesfm2_5", device),
        FinBERTAdapter(cache_dir / "finbert", device),
    ], shadow_mode=True)
