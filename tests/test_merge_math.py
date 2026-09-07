"""CPU unit tests for the dense adapter merging in src/generation.py.

Builds a tiny random Mistral, trains nothing, saves two random rank-r LoRA
adapters and checks that
  * load_adapter_deltas() reproduces PEFT's get_delta_weight();
  * 'linear' equals 0.5*dW_a + 0.5*dW_b exactly;
  * 'ties' equals an independent reference TIES on the dense deltas;
  * apply_deltas() changes the base weights by exactly the merged delta;
  * PEFT's own 'linear' combination has cross terms (documents why it is not used).

Run:  PYTHONPATH=. python -m pytest tests/test_merge_math.py -q
"""

import copy

import pytest
import torch

from src.generation import apply_deltas, combine_deltas, load_adapter_deltas


def _tiny_model():
    from transformers import MistralConfig, MistralForCausalLM
    torch.manual_seed(0)
    cfg = MistralConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, max_position_embeddings=64)
    return MistralForCausalLM(cfg).to(torch.float32)


def _with_two_adapters(model, r=4):
    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=r, lora_alpha=2 * r, target_modules=["q_proj", "v_proj"],
                      lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    pm = get_peft_model(model, lcfg, adapter_name="a0")
    pm.add_adapter("a1", lcfg)
    g = torch.Generator().manual_seed(1)
    for _, module in pm.named_modules():
        if hasattr(module, "lora_A") and "a0" in module.lora_A:
            for a in ("a0", "a1"):
                module.lora_A[a].weight.data = torch.randn(module.lora_A[a].weight.shape, generator=g)
                module.lora_B[a].weight.data = torch.randn(module.lora_B[a].weight.shape, generator=g)
    return pm


PEFT_PREFIX = "base_model.model."


def _lora_layers(pm):
    """LoRA layers keyed by the module path *inside the base model*."""
    return {name[len(PEFT_PREFIX):]: m for name, m in pm.named_modules()
            if hasattr(m, "lora_A") and "a0" in m.lora_A}


def _manual_ties(deltas, weights, density):
    """Reference TIES on dense tensors (trim -> elect sign -> disjoint mean)."""
    trimmed = []
    for d in deltas:
        k = int(density * d.numel())
        idx = torch.topk(d.abs().flatten(), k).indices
        mask = torch.zeros(d.numel(), dtype=torch.bool)
        mask[idx] = True
        trimmed.append(torch.where(mask.reshape(d.shape), d, torch.zeros_like(d)))
    stacked = torch.stack(trimmed)
    sign = torch.where(stacked.sum(0) >= 0, 1.0, -1.0)
    agree = torch.sign(stacked) == sign
    weighted = stacked * torch.tensor(weights).reshape(-1, 1, 1)
    num = (weighted * agree).sum(0)
    den = agree.sum(0).clamp(min=1)
    return num / den


@pytest.fixture()
def saved_adapters(tmp_path):
    pm = _with_two_adapters(_tiny_model())
    pm.save_pretrained(str(tmp_path))  # writes a0/ and a1/
    ref = {
        name: {a: layer.get_delta_weight(a).detach().clone() for a in ("a0", "a1")}
        for name, layer in _lora_layers(pm).items()
    }
    base = copy.deepcopy(pm.get_base_model())
    return tmp_path, ref, base, pm


def test_delta_loader_matches_peft(saved_adapters):
    tmp_path, ref, _, _ = saved_adapters
    for a in ("a0", "a1"):
        deltas = load_adapter_deltas(str(tmp_path / a))
        assert set(deltas) == set(ref), "module paths differ"
        for name, d in deltas.items():
            assert torch.allclose(d, ref[name][a], atol=1e-5), name


def test_linear_is_exact_delta_average(saved_adapters):
    tmp_path, ref, _, _ = saved_adapters
    sets = [load_adapter_deltas(str(tmp_path / a)) for a in ("a0", "a1")]
    merged = combine_deltas(sets, "linear")
    for name in ref:
        expected = 0.5 * ref[name]["a0"] + 0.5 * ref[name]["a1"]
        assert torch.allclose(merged[name], expected, atol=1e-5), name


def test_ties_matches_reference(saved_adapters):
    tmp_path, ref, _, _ = saved_adapters
    sets = [load_adapter_deltas(str(tmp_path / a)) for a in ("a0", "a1")]
    merged = combine_deltas(sets, "ties", density=0.5)
    for name in ref:
        expected = _manual_ties([ref[name]["a0"], ref[name]["a1"]], [0.5, 0.5], 0.5)
        err = (merged[name] - expected).abs().max().item()
        assert torch.allclose(merged[name], expected, atol=1e-5), f"{name}: max err {err:.3e}"
        # TIES must differ from the plain average (otherwise the test is vacuous)
        avg = 0.5 * ref[name]["a0"] + 0.5 * ref[name]["a1"]
        assert not torch.allclose(merged[name], avg, atol=1e-3)


def test_apply_deltas_changes_weights_exactly(saved_adapters):
    tmp_path, ref, base, _ = saved_adapters
    sets = [load_adapter_deltas(str(tmp_path / a)) for a in ("a0", "a1")]
    merged = combine_deltas(sets, "linear")
    before = {n: base.get_submodule(n).weight.detach().clone() for n in merged}
    apply_deltas(base, merged)
    for n in merged:
        after = base.get_submodule(n).weight
        assert torch.allclose(after - before[n], merged[n], atol=1e-5), n


def test_peft_linear_has_cross_terms(saved_adapters):
    """Document why PEFT's add_weighted_adapter('linear') is NOT used."""
    _, ref, _, pm = saved_adapters
    pm.add_weighted_adapter(adapters=["a0", "a1"], weights=[0.5, 0.5],
                            adapter_name="lin", combination_type="linear")
    name, layer = next(iter(_lora_layers(pm).items()))
    merged = layer.get_delta_weight("lin")
    expected = 0.5 * ref[name]["a0"] + 0.5 * ref[name]["a1"]
    assert not torch.allclose(merged, expected, atol=1e-3)


# ---------------------------------------------------------------------------
# Base-model registry helpers (src/base_models.py)
# ---------------------------------------------------------------------------


def test_resolve_target_modules_tiny_mistral():
    from src.base_models import resolve_target_modules
    model = _tiny_model()
    mods = resolve_target_modules(model, ("q_proj", "v_proj"))
    assert len(mods) == 4  # 2 layers x (q_proj, v_proj)
    assert all(m.startswith("model.layers.") for m in mods)
    # explicit full names must be resolvable back to the same Linear modules
    for m in mods:
        assert isinstance(model.get_submodule(m), torch.nn.Linear)


def test_resolve_target_modules_prefers_language_model_subtree():
    import torch.nn as nn
    from src.base_models import resolve_target_modules

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = nn.ModuleDict({"q_proj": nn.Linear(4, 4), "v_proj": nn.Linear(4, 4)})
            self.language_model = nn.ModuleDict({"q_proj": nn.Linear(4, 4), "v_proj": nn.Linear(4, 4)})

    mods = resolve_target_modules(Wrapper(), ("q_proj", "v_proj"))
    assert mods == ["language_model.q_proj", "language_model.v_proj"]


def test_clean_response_gemma4_strips_thought_block():
    from src.base_models import clean_response

    class Tok:
        all_special_tokens = ["<turn|>", "<eos>", "<pad>", "<|channel>", "<channel|>"]

    raw = "<|channel>thought\n<channel|>Yes, the government should.<turn|><pad><pad>"
    assert clean_response(raw, Tok(), "gemma4") == "Yes, the government should."
    raw2 = "<|channel>thought\nsome hidden reasoning<channel|>\nAnswer text.<eos>"
    assert clean_response(raw2, Tok(), "gemma4") == "Answer text."
    # truncated block (cut by max_new_tokens): opening marker removed, text kept
    assert clean_response("<|channel>thought\npartial", Tok(), "gemma4") == "partial"
    # mistral path is a plain strip
    assert clean_response("  hello </s>", Tok(), "mistral") == "hello </s>"


def test_build_instances_carry_base_model_and_root():
    from src.base_models import get_base_model, models_root_for
    from src.generation import build_instances
    spec = get_base_model("gemma4")
    inst = build_instances(spec["seeds"], spec["control_pairs"], models_root_for(spec), "gemma4")
    assert len(inst) == 1 + 3 * 2 + 2 * 2 + 4
    assert all(i["base_model"] == "gemma4" for i in inst)
    sft = next(i for i in inst if i["instance"] == "sft_left_s42")
    assert sft["adapters"] == ["/models/gemma4/sft_left_s42_adapter"]
    dry = build_instances(spec["seeds"], spec["control_pairs"], models_root_for(spec), "gemma4", "_dryrun")
    assert next(i for i in dry if i["instance"] == "sft_left_s42")["adapters"][0].endswith("_adapter_dryrun")
