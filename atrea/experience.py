"""Agent-controlled search tool for the paper-form experience library."""
from __future__ import annotations


class SearchExperienceTool:
    name = "SearchExperience"
    description = (
        "Search the agent's archived experience papers by topic, model, dataset, "
        "failure mode, or method. Call only when prior experience is useful; "
        "the library is never pushed into the agent automatically."
    )

    def __init__(self, store, *, memory_scope="default", trace=None):
        self.store = store
        self.memory_scope = memory_scope
        self.trace = trace
        self.calls = 0
        self.last_results = None

    def __call__(self, query, *, limit=5, dataset=None):
        return self.search_experience(query, limit=limit, dataset=dataset)

    def search_experience(self, query, *, limit=5, dataset=None):
        self.calls += 1
        result = self.store.search_experience_papers(
            query, limit=limit, dataset=dataset, memory_scope=self.memory_scope
        )
        self.last_results = result
        if self.trace:
            self.trace.log(
                phase="act", action="search_experience_papers",
                observation=f"Experience paper search returned {len(result.get('results', []))} result(s)",
                details={"query": str(query), "limit": int(limit),
                         "dataset": dataset, "total_matches": result.get("total_matches", 0)},
            )
        return result
