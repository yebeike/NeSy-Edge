"""Action query-construction helpers."""

from __future__ import annotations

from experiments.project.action.core.noise_v1 import inject_counter_noise, inject_text_noise
from experiments.project.action.core.noise_v2 import build_query_v2
from experiments.project.action.core.schema import ActionCase, ActionQuery, BenchmarkBundle
from experiments.project.action.methods.formal_config import FORMAL_NOISE_VARIANTS


def build_query_v1(case: ActionCase, *, noise_level: float, seed: int) -> ActionQuery:
    return ActionQuery(
        base_case=case,
        noise_level=noise_level,
        seed=seed,
        incident_text=inject_text_noise(
            case.incident_text,
            dataset=case.dataset,
            noise_level=noise_level,
            seed=seed,
        ),
        raw_features={
            name: inject_counter_noise(
                counter,
                dataset=case.dataset,
                noise_level=noise_level,
                seed=seed,
            )
            for name, counter in case.raw_features.items()
        },
        metadata={"noise_level": noise_level, "seed": seed, "noise_variant": "v1"},
    )


def build_query(
    case: ActionCase,
    *,
    noise_level: float,
    seed: int,
    noise_variant: str | None = None,
) -> ActionQuery:
    variant = noise_variant or FORMAL_NOISE_VARIANTS[case.dataset]
    if variant == "v1":
        return build_query_v1(case, noise_level=noise_level, seed=seed)
    if variant == "v2":
        return build_query_v2(case, noise_level=noise_level, seed=seed)
    raise ValueError(f"unsupported noise_variant: {variant}")


def support_cases(bundle: BenchmarkBundle, case: ActionCase) -> list[ActionCase]:
    return [
        item
        for item in bundle.cases
        if item.dataset == case.dataset and item.case_id != case.case_id
    ]
