"""Industrial diagnostic Agent runtime with model-authoritative fault labels."""

from .agent import DiagnosticAgent, AgentRuntimeConfig
from .providers import OfflinePlanner, OpenAICompatiblePlanner
from .model_bridge import VersionedForecastModelBridge, StructuredPredictorBridge
from .industrial_bridge import VersionedIndustrialFaultBridge
from .industrial_fault import IndustrialFaultTrainConfig, train_industrial_fault_model
from .model_router import TaskAwareModelRouter

__all__ = [
    "DiagnosticAgent", "AgentRuntimeConfig", "OfflinePlanner", "OpenAICompatiblePlanner",
    "VersionedForecastModelBridge", "StructuredPredictorBridge",
    "VersionedIndustrialFaultBridge", "IndustrialFaultTrainConfig",
    "train_industrial_fault_model", "TaskAwareModelRouter",
]
