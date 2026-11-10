"""ATREA: Autonomous Time-Series Research & Evolution Agent."""
from .trigger import TriggerConfig, TriggerDecision, evaluate_training_trigger
from .store import ExperienceStore
from .registry import ModelRegistry, ForecastAdapter
from .architecture import (ARCHITECTURE_CATALOG, build_model, validate_architecture_spec,
                           run_architecture_sweep, run_nas_search)
from .nas_training import build_etth1_nas_runner, run_etth1_nas
from .nas import (generate_nas_population, mutate_architecture, make_ablation_specs,
                  build_loss, validate_nas_spec, crossover_architectures,
                  infer_structure_hypotheses, build_optimizer, build_scheduler,
                  propose_meta_operator_graph, mutate_meta_operator_graph)
__all__ = ["TriggerConfig", "TriggerDecision", "evaluate_training_trigger",
           "ExperienceStore", "ModelRegistry", "ForecastAdapter",
           "ARCHITECTURE_CATALOG", "build_model", "validate_architecture_spec", "run_architecture_sweep",
           "run_nas_search", "generate_nas_population", "mutate_architecture",
           "make_ablation_specs", "build_loss", "validate_nas_spec",
           "crossover_architectures", "infer_structure_hypotheses",
           "build_optimizer", "build_scheduler", "build_etth1_nas_runner", "run_etth1_nas",
           "propose_meta_operator_graph", "mutate_meta_operator_graph"]
