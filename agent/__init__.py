"""Public Agent interface; heavy dependencies are loaded only when needed."""
# modify:audit-lazy-import - keep public import compatibility for lightweight RAG/risk use.
__all__ = ["OilTemperatureAgent"]
def __getattr__(name):
    if name == "OilTemperatureAgent":
        from agent.agent import OilTemperatureAgent
        return OilTemperatureAgent
    raise AttributeError(name)
