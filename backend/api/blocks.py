from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api.deps import get_services
from backend.services import AppServices

router = APIRouter(prefix="/blocks", tags=["blocks"])


@router.get("")
def list_blocks(services: AppServices = Depends(get_services)) -> list[dict]:
    return [
        {
            "key": spec.key,
            "name": spec.display_name,
            "aliases": spec.aliases,
            "version": spec.version,
            "category": spec.category,
            "description": spec.description,
            "n_inputs": spec.n_inputs,
            "input_labels": spec.input_labels,
            "output_labels": spec.output_labels,
            "param_schema": [
                {
                    "key": param.key,
                    "type": param.type,
                    "default": param.default,
                    "required": param.required,
                    "description": param.description,
                    "example": param.example,
                    "browse_mode": param.browse_mode,
                }
                for param in spec.param_schema
            ],
            "params": spec.params,
            "param_types": spec.param_types,
            "param_descriptions": spec.param_descriptions,
            "required_params": spec.required_params,
            "param_examples": spec.param_examples,
            "is_custom": spec.is_custom,
            "custom_filename": spec.custom_filename,
            "arity_input_param": spec.arity_input_param,
            "arity_output_param": spec.arity_output_param,
        }
        for spec in services.registry.all_specs()
    ]



