from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

from heavy_lab.paths import LabPaths


@dataclass(frozen=True)
class BotAction:
    kind: str
    profile: str
    manifest: Path


@dataclass(frozen=True)
class BotReply:
    text: str
    action: BotAction | None = None


class BotCommandProcessor:
    """Pure command layer for the local Telegram controller."""

    def __init__(
        self,
        *,
        root: Path,
        owner_chat_id: int,
        campaign_manifest: Path | None = None,
        doctor_provider: Callable[[], dict[str, Any]] | None = None,
        readiness_provider: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.owner_chat_id = int(owner_chat_id)
        self.campaign_manifest = (
            Path(campaign_manifest).expanduser().resolve()
            if campaign_manifest is not None
            else None
        )
        self._doctor_provider = doctor_provider or self._default_doctor
        self._readiness_provider = readiness_provider or self._default_readiness

    def _default_doctor(self) -> dict[str, Any]:
        from heavy_lab.doctor import doctor_report

        return doctor_report(self.root)

    def _default_readiness(self) -> dict[str, Any]:
        if self.campaign_manifest is None or not self.campaign_manifest.is_file():
            return {"ready": False, "missing": ["campaign_manifest"]}

        try:
            payload = json.loads(self.campaign_manifest.read_text(encoding="utf-8"))
        except Exception:
            return {"ready": False, "missing": ["campaign_manifest_invalid"]}

        from heavy_lab.training.readiness import training_readiness

        base = self.campaign_manifest.parent

        def resolve(name: str) -> Path | None:
            value = str(payload.get(name, "")).strip()
            if not value:
                return None
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = base / path
            return path.resolve()

        return training_readiness(
            root=self.root,
            train_parquet=resolve("train_parquet"),
            validation_parquet=resolve("validation_parquet"),
            finbert_train_jsonl=resolve("finbert_train_jsonl"),
            finbert_validation_jsonl=resolve("finbert_validation_jsonl"),
        )

    @staticmethod
    def _command(text: str) -> tuple[str, list[str]]:
        parts = str(text or "").strip().split()
        if not parts:
            return "", []
        command = parts[0].split("@", 1)[0].lower()
        return command, parts[1:]

    def handle(self, *, chat_id: int, text: str) -> BotReply:
        if int(chat_id) != self.owner_chat_id:
            return BotReply("غير مصرح لهذا الحساب.")

        command, args = self._command(text)

        if command in {"/start", "/help"}:
            return BotReply(
                "Fahad Local AI Bot\n"
                "Research/PAPER control only\n\n"
                "/status - hardware and safety status\n"
                "/trainingready - training readiness\n"
                "/runs - latest local training runs\n"
                "/trainall [smoke|max] - request local training campaign\n"
                "/trainstatus - local training process status\n"
                "/help - this help"
            )

        if command == "/status":
            report = self._doctor_provider()
            hardware = dict(report.get("hardware") or {})
            device = hardware.get("device", "unknown")
            gpu_name = hardware.get("gpu_name") or hardware.get("name") or "none"
            vram = hardware.get("vram_gb")
            profile = report.get("training_profile", "unknown")
            vram_text = "unknown" if vram is None else f"{float(vram):g} GB"
            return BotReply(
                "Fahad Local AI Bot\n"
                "mode: PAPER/RESEARCH\n"
                "LIVE trading: DISABLED\n"
                f"device: {device}\n"
                f"gpu: {gpu_name}\n"
                f"vram: {vram_text}\n"
                f"training profile: {profile}"
            )

        if command == "/trainingready":
            report = self._readiness_provider()
            if report.get("ready"):
                return BotReply("READY: all local heavy-training requirements are satisfied")
            missing = [str(item) for item in report.get("missing", [])]
            suffix = ", ".join(missing) if missing else "unknown requirements"
            return BotReply(f"NOT READY: {suffix}")

        if command == "/runs":
            return BotReply(self._runs_text())

        if command == "/trainall":
            profile = (args[0].strip().lower() if args else "max")
            if profile not in {"smoke", "max"}:
                return BotReply("profile must be smoke or max")
            if self.campaign_manifest is None or not self.campaign_manifest.is_file():
                return BotReply("NOT READY: campaign_manifest")
            readiness = self._readiness_provider()
            if not readiness.get("ready"):
                missing = ", ".join(str(x) for x in readiness.get("missing", []))
                return BotReply(f"NOT READY: {missing or 'training requirements'}")
            return BotReply(
                f"TRAIN REQUEST ACCEPTED: {profile.upper()}",
                BotAction(
                    kind="train-all",
                    profile=profile,
                    manifest=self.campaign_manifest.resolve(),
                ),
            )

        if command == "/trainstatus":
            return BotReply("Training process status is available when the local bot runtime is running.")

        return BotReply("Unknown command. Use /help")

    def _runs_text(self, limit: int = 5) -> str:
        paths = LabPaths.from_root(self.root)
        if not paths.runs.is_dir():
            return "No training runs found."

        items: list[dict[str, Any]] = []
        for state_path in paths.runs.glob("*/state.json"):
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            items.append(state)

        if not items:
            return "No training runs found."

        items.sort(key=lambda item: str(item.get("updated_at_utc", "")), reverse=True)
        lines = ["Latest training runs:"]
        for item in items[:limit]:
            run_id = str(item.get("run_id", "unknown"))
            kind = str(item.get("kind", "unknown"))
            status = str(item.get("status", "unknown"))
            lines.append(f"{run_id} | {kind} | {status}")
        return "\n".join(lines)
