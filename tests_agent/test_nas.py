import unittest
import numpy as np
try:
    import torch
except Exception:
    torch=None
from atrea.nas import (seed_architecture_population, generate_nas_population,
    mutate_architecture, make_ablation_specs, build_loss, validate_nas_spec,
    crossover_architectures, build_optimizer, build_scheduler, infer_structure_hypotheses,
    pareto_front, propose_operator_innovations, propose_math_operator_graph,
    mutate_math_operator_graph, MATH_EQUATIONS)
from atrea.architecture import build_model, run_nas_search

class NASTests(unittest.TestCase):
    def test_population_search_space_and_mutation(self):
        base=seed_architecture_population()
        self.assertGreaterEqual(len(base),4)
        pop=generate_nas_population(parents=base,population_size=8,seed=7)
        self.assertEqual(len(pop),8)
        for spec in pop: validate_nas_spec(spec)

    @unittest.skipIf(torch is None,"PyTorch unavailable")
    def test_innovation_blocks_and_training_recipe(self):
        for block in ("trend_residual", "frequency_mixer", "cross_var"):
            spec={"model_name":"NAS_Composed","config":{"branches":[[block,"lstm"]],
                "hidden":32,"attention":"none","attention_heads":4,"fusion":"concat",
                "loss":"mae_slope","optimizer":"adamw","learning_rate":0.001}}
            model=build_model(spec,input_len=24,n_features=7,pred_len=6)
            y=model(torch.randn(3,24,7))
            self.assertEqual(tuple(y.shape),(3,6))
            opt=build_optimizer(model,spec["config"])
            self.assertIsNotNone(opt)
            self.assertIsNone(build_scheduler(opt,spec["config"]))
            self.assertTrue(torch.isfinite(build_loss("mae_slope")(y,torch.randn(3,6))).item())

    def test_crossover_and_hypothesis_inference(self):
        a,b=seed_architecture_population()[:2]
        child=crossover_architectures(a,b,seed=8)
        validate_nas_spec(child)
        from atrea.nas import _fingerprint
        fake=[{"parent_architecture_id":_fingerprint(a),"status":"completed",
               "parent_metric":1.0,"metrics":{"mae":0.9},
               "config":{"model_name":"NAS_Composed","config":{"branches":[["lstm"]],
                   "hidden":64,"attention":"none","fusion":"concat","loss":"mae"}}}]
        hypotheses=infer_structure_hypotheses(a,fake)
        self.assertTrue(hypotheses)
        self.assertEqual(hypotheses[0]["next_search_action"],"favor_simplification")

    def test_invalid_topology_rejected(self):
        with self.assertRaises(ValueError):
            validate_nas_spec({"model_name":"NAS_Composed","config":{"branches":[["evil_code"]]}})

    @unittest.skipIf(torch is None,"PyTorch unavailable")
    def test_composed_model_forward_and_loss(self):
        spec={"model_name":"NAS_Composed","config":{
            "branches":[["cnn","lstm"],["lstm","transformer"]],
            "hidden":32,"attention":"fusion","attention_heads":4,
            "fusion":"gated","loss":"huber"}}
        model=build_model(spec,input_len=96,n_features=7,pred_len=24)
        y=model(torch.randn(4,96,7))
        self.assertEqual(tuple(y.shape),(4,24))
        loss=build_loss("huber")(y,torch.randn(4,24))
        self.assertTrue(torch.isfinite(loss).item())

    def test_search_uses_real_runner_and_records_ablation(self):
        calls=[]
        def runner(spec,seed,loss_fn):
            calls.append((spec["config"]["attention"],seed))
            return {"metrics":{"mae":1.0 + spec["config"]["hidden"]/10000}}
        def ablate(spec,seed,loss_fn):
            return {"metrics":{"mae":1.1}}
        result=run_nas_search(train_evaluate=runner,ablation_evaluate=ablate,
            split_hash="frozen-v1",population_size=4,generations=1,seeds=[11])
        self.assertEqual(len(result["experiments"]),4)
        self.assertTrue(result["best_candidate"])
        self.assertGreater(result["ablation_count"],0)
        self.assertIn("next_research_hypotheses", result)
        self.assertTrue(result["next_generation_seed_population"])
        self.assertFalse(result["promotion_allowed"])
        self.assertIn("pareto_candidates", result)
        self.assertEqual(result["search_mode"], "bilevel_differentiable_symbolic_nas_v9")
        self.assertEqual(len(calls),4)


    def test_research_cycle_persists_nas_hypotheses_and_offspring(self):
        import tempfile
        from pathlib import Path
        from atrea.store import ExperienceStore
        from atrea.cycle import run_research_cycle
        with tempfile.TemporaryDirectory() as td:
            store=ExperienceStore(Path(td)/"experience.db")
            def planner(papers, memory):
                return {"statement":"Test whether topology variants improve MAE.",
                    "rationale":"controlled NAS integration test",
                    "plan":{"dataset":"ETTh1","seeds":[11],"primary_metric":"mae",
                            "nas_config":{"population_size":2,"generations":1}}}
            def runner(spec, seed, loss_fn):
                return {"metrics":{"mae":1.0,"rmse":1.2}}
            def ablation(spec, seed, loss_fn):
                return {"metrics":{"mae":1.1,"rmse":1.3}}
            result=run_research_cycle(store=store,train_and_evaluate=lambda p,b:[],
                paper_search=lambda *a,**k:[],hypothesis_builder=planner,
                nas_train_evaluate=runner,nas_ablation_evaluate=ablation,
                nas_split_hash="fixed-split-test",memory_scope="nas-test")
            memory=store.retrieve_experience(memory_scope="nas-test",limit=50)
            self.assertGreater(len(result["experiment_ids"]),0)
            self.assertTrue(any("nas_offspring_seed" in p["scope"].get("kind","") for p in memory["patterns"]))
            self.assertTrue(any("nas_ablation_hypothesis" in p["scope"].get("kind","") for p in memory["patterns"]))

    @unittest.skipIf(torch is None,"PyTorch unavailable")
    def test_amplitude_phase_operator_forward_and_gradient(self):
        spec={"model_name":"NAS_Composed","config":{"branches":[["amplitude_phase","lstm"]],
            "hidden":32,"attention":"none","attention_heads":4,"fusion":"concat","loss":"mae"}}
        model=build_model(spec,input_len=24,n_features=7,pred_len=6)
        x=torch.randn(3,24,7,requires_grad=True)
        y=model(x)
        self.assertEqual(tuple(y.shape),(3,6))
        y.mean().backward()
        self.assertIsNotNone(model.branches[0][0].phase.grad)
        self.assertTrue(torch.isfinite(model.branches[0][0].phase.grad).all().item())

    def test_operator_innovation_is_error_driven_and_validated(self):
        proposals=propose_operator_innovations(weakness="periodic phase drift")
        self.assertTrue(any("amplitude_phase" in str(x["config"]) for x in proposals))
        for spec in proposals:
            validate_nas_spec(spec)

    def test_pareto_front_keeps_accuracy_efficiency_tradeoffs(self):
        rows=[
            {"status":"completed","metrics":{"mae":1.0,"rmse":1.3},"parameter_count":100,"train_seconds":10},
            {"status":"completed","metrics":{"mae":1.1,"rmse":1.4},"parameter_count":50,"train_seconds":5},
            {"status":"completed","metrics":{"mae":1.2,"rmse":1.5},"parameter_count":150,"train_seconds":15},
        ]
        front=pareto_front(rows)
        self.assertEqual(len(front),2)
        self.assertIn(rows[0],front)
        self.assertIn(rows[1],front)
        self.assertNotIn(rows[2],front)

    @unittest.skipIf(torch is None,"PyTorch unavailable")
    def test_v6_meta_operator_graph_compiles_and_backpropagates(self):
        from atrea.nas import propose_meta_operator_graph, mutate_meta_operator_graph
        graph=propose_meta_operator_graph(weakness="periodic phase drift",seed=17)
        spec={"model_name":"NAS_Composed","config":{"branches":[[graph,"lstm"]],
            "hidden":32,"attention":"none","attention_heads":4,"fusion":"concat","loss":"mae"}}
        validate_nas_spec(spec)
        model=build_model(spec,input_len=24,n_features=7,pred_len=6)
        x=torch.randn(3,24,7,requires_grad=True); y=model(x)
        self.assertEqual(tuple(y.shape),(3,6)); y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all().item())
        changed=mutate_meta_operator_graph(graph,seed=23)
        validate_nas_spec({"model_name":"NAS_Composed","config":{**spec["config"],
            "branches":[[changed,"lstm"]]}})

    def test_v6_meta_operator_rejects_untrusted_primitives(self):
        bad={"model_name":"NAS_Composed","config":{"branches":[[{"type":"meta_operator",
            "ops":["__import__"]}]],"hidden":32}}
        with self.assertRaises(ValueError): validate_nas_spec(bad)

    def test_v6_error_driven_search_injects_meta_operator_candidate(self):
        seen=[]
        def runner(spec,seed,loss_fn):
            seen.append(spec)
            return {"metrics":{"mae":1.0,"rmse":1.2},"status":"completed"}
        result=run_nas_search(train_evaluate=runner,split_hash="v6-fixed",
            population_size=4,generations=1,seeds=[11],weakness="periodic phase drift")
        self.assertTrue(any(isinstance(block,dict) and block.get("type")=="meta_operator"
            for spec in seen for branch in spec["config"]["branches"] for block in branch))
        self.assertEqual(result["search_mode"],"bilevel_differentiable_symbolic_nas_v9")

    @unittest.skipIf(torch is None,"PyTorch unavailable")
    def test_v7_mathematical_equation_families_compile_and_backpropagate(self):
        from atrea.nas import MATH_EQUATIONS
        for equation in MATH_EQUATIONS:
            gate_ops = [] if equation == "residual_add" else ["trend_split"]
            graph = {"type":"math_operator", "equation":equation,
                     "ops":["depthwise_conv", "pointwise_mlp"], "gate_ops":gate_ops}
            spec={"model_name":"NAS_Composed","config":{"branches":[[graph,"lstm"]],
                "hidden":32,"attention":"none","attention_heads":4,"fusion":"concat","loss":"mae"}}
            validate_nas_spec(spec)
            model=build_model(spec,input_len=24,n_features=7,pred_len=6)
            x=torch.randn(2,24,7,requires_grad=True)
            y=model(x)
            self.assertEqual(tuple(y.shape),(2,6))
            y.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all().item())

    def test_v7_math_operator_proposal_mutation_and_validation(self):
        graph=propose_math_operator_graph(weakness="periodic phase drift",seed=41)
        self.assertEqual(graph["type"],"math_operator")
        self.assertIn(graph["equation"],MATH_EQUATIONS)
        changed=mutate_math_operator_graph(graph,seed=42)
        spec={"model_name":"NAS_Composed","config":{"branches":[[changed,"lstm"]],"hidden":32}}
        validate_nas_spec(spec)
        invalid={"model_name":"NAS_Composed","config":{"branches":[[{"type":"math_operator",
            "equation":"eval(__import__)","ops":["pointwise_mlp"],"gate_ops":[]}]],"hidden":32}}
        with self.assertRaises(ValueError): validate_nas_spec(invalid)

    def test_v7_innovation_injects_math_equation_candidate(self):
        seen=[]
        def runner(spec,seed,loss_fn):
            seen.append(spec)
            return {"metrics":{"mae":1.0,"rmse":1.2},"status":"completed"}
        result=run_nas_search(train_evaluate=runner,split_hash="v7-fixed",
            population_size=5,generations=1,seeds=[11],weakness="periodic phase drift")
        self.assertTrue(any(isinstance(block,dict) and block.get("type")=="math_operator"
            for spec in seen for branch in spec["config"]["branches"] for block in branch))
        self.assertTrue(any(isinstance(block,dict) and block.get("type")=="darts_operator"
            for spec in seen for branch in spec["config"]["branches"] for block in branch))
        self.assertEqual(result["search_mode"],"bilevel_differentiable_symbolic_nas_v9")

    @unittest.skipIf(torch is None,"PyTorch unavailable")
    def test_v8_darts_mixed_operator_learns_architecture_logits(self):
        from atrea.nas import propose_darts_operator, extract_darts_genotype
        graph = propose_darts_operator(seed=91, ops=["depthwise_conv", "pointwise_mlp", "spectral_gain"])
        spec = {"model_name":"NAS_Composed", "config":{"branches":[[graph,"lstm"]],
            "hidden":32,"attention":"none","attention_heads":4,"fusion":"concat","loss":"mae"}}
        validate_nas_spec(spec)
        model = build_model(spec,input_len=24,n_features=7,pred_len=6)
        x=torch.randn(4,24,7); target=torch.randn(4,6)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
        before=model.branches[0][0].architecture_logits.detach().clone()
        loss=(model(x)-target).abs().mean(); loss.backward(); optimizer.step()
        after=model.branches[0][0].architecture_logits.detach()
        self.assertTrue(torch.isfinite(after).all().item())
        self.assertFalse(torch.equal(before,after))
        discrete=extract_darts_genotype(model,spec)
        self.assertEqual(discrete["config"]["branches"][0][0]["type"],"meta_operator")

    def test_v8_symbolic_regression_finds_time_trend(self):
        from atrea.nas import search_symbolic_residual
        n=200; t=np.linspace(-1,1,n)
        x=np.column_stack([np.sin(2*np.pi*t),np.cos(2*np.pi*t)])
        y=0.7*t + 0.2*np.sin(2*np.pi*t)
        result=search_symbolic_residual(x,y,max_terms=4)
        self.assertTrue(result["expression"])
        self.assertLess(result["validation_mae"],0.15)
        self.assertIn("chronological_validation",result["selection"])

if __name__=="__main__": unittest.main()
