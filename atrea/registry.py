"""Plug-and-play forecasting model registry with a strict adapter contract."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np

class ForecastAdapter(ABC):
    name="abstract"; version="0"
    @abstractmethod
    def fit(self, X, y, **kwargs): ...
    @abstractmethod
    def predict(self, X): ...

@dataclass
class ModelSpec:
    name: str
    factory: Callable[..., ForecastAdapter]
    tags: list[str]=field(default_factory=list)
    requirements: list[str]=field(default_factory=list)
    source_url: str|None=None

class ModelRegistry:
    def __init__(self): self._models={}
    def register(self,name,factory,**metadata):
        if name in self._models: raise ValueError(f"Model already registered: {name}")
        self._models[name]=ModelSpec(name,factory,**metadata)
    def create(self,name,**kwargs):
        if name not in self._models: raise KeyError(f"Unknown model {name}; available={self.list()}")
        return self._models[name].factory(**kwargs)
    def list(self): return sorted(self._models)
    def metadata(self,name): return self._models[name]

class CallableAdapter(ForecastAdapter):
    """Wrap existing project train/predict functions without refactoring them."""
    def __init__(self,name,fit_fn,predict_fn,version="local",**kwargs):
        self.name=name; self.version=version; self.fit_fn=fit_fn; self.predict_fn=predict_fn
    def fit(self,X,y,**kwargs): self.state=self.fit_fn(X,y,**kwargs); return self
    def predict(self,X): return np.asarray(self.predict_fn(self.state,X))

def default_registry():
    r=ModelRegistry()
    # Register optional adapters only when explicitly requested by the host. This
    # keeps importing ATREA cheap and avoids downloading model weights implicitly.
    return r

def register_optional_chronos2(registry, **kwargs):
    """Register Chronos-2 lazily; model weights load only on create()."""
    registry.register("Chronos-2", lambda: Chronos2Adapter(**kwargs),
                      tags=["foundation", "forecasting"],
                      requirements=["chronos-forecasting"],
                      source_url="https://huggingface.co/amazon/chronos-2")
    return registry

class Chronos2Adapter(ForecastAdapter):
    """Optional zero/few-shot Chronos-2 adapter; install requirements_chronos.txt."""
    name="Chronos-2"; version="2.x"
    def __init__(self, model_id="amazon/chronos-2", device_map="cpu"):
        try: from chronos import Chronos2Pipeline
        except ImportError as e: raise RuntimeError("Install requirements_chronos.txt in a separate env") from e
        self.pipeline=Chronos2Pipeline.from_pretrained(model_id,device_map=device_map)
    def fit(self,X,y,**kwargs):
        raise NotImplementedError("Chronos-2 base adapter is inference-only; use the project's LoRA fine-tuning pipeline")
    def predict(self,X):
        # X: [batch, context_len] univariate target windows
        outputs=[]
        for row in np.asarray(X):
            pred=self.pipeline.predict([row.tolist()],prediction_length=kwargs_prediction_length(self))
            outputs.append(np.asarray(pred[0])[-1].tolist())
        return np.asarray(outputs)

def kwargs_prediction_length(adapter):
    return getattr(adapter,"prediction_length",24)
