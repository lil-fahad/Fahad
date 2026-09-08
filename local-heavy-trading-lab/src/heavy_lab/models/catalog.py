from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    local_name: str
    family: str
    purpose: str


MODEL_CATALOG: dict[str, ModelSpec] = {
    "chronos2": ModelSpec(
        name="chronos2",
        repo_id="amazon/chronos-2",
        local_name="chronos-2",
        family="forecasting",
        purpose="time-series forecasting and native domain fine-tuning",
    ),
    "timesfm25": ModelSpec(
        name="timesfm25",
        repo_id="google/timesfm-2.5-200m-transformers",
        local_name="timesfm-2.5",
        family="forecasting",
        purpose="time-series forecasting with LoRA/full fine-tuning gates",
    ),
    "kronos": ModelSpec(
        name="kronos",
        repo_id="NeoQuasar/Kronos-base",
        local_name="kronos-base",
        family="ohlcv",
        purpose="financial K-line forecasting and fine-tuning",
    ),
    "kronos_tokenizer": ModelSpec(
        name="kronos_tokenizer",
        repo_id="NeoQuasar/Kronos-Tokenizer-base",
        local_name="kronos-tokenizer-base",
        family="ohlcv-tokenizer",
        purpose="paired tokenizer for Kronos-base",
    ),
    "ttm": ModelSpec(
        name="ttm",
        repo_id="ibm-granite/granite-timeseries-ttm-r2",
        local_name="ttm-r2",
        family="forecasting-control",
        purpose="small zero-shot/fine-tuned control model",
    ),
    "finbert": ModelSpec(
        name="finbert",
        repo_id="ProsusAI/finbert",
        local_name="finbert",
        family="sentiment",
        purpose="financial sentiment classification and calibration",
    ),
}
