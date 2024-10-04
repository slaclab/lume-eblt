from __future__ import annotations


import pathlib

import numpy as np
import pydantic
import pydantic_core
from pydantic import BaseModel
from pmd_beamphysics.units import pmd_unit

from typing import Any, Sequence, Type, Union, Annotated, Tuple, Dict
AnyPath = Union[pathlib.Path, str]

# Custom Pydantic class to handle numpy ndarray
class _PydanticNDArray:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: Type[Any],
        handler: pydantic.GetCoreSchemaHandler,
    ) -> pydantic_core.core_schema.CoreSchema:
        def serialize(
            obj: np.ndarray, info: pydantic_core.core_schema.SerializationInfo
        ):
            if not isinstance(obj, np.ndarray):
                raise ValueError(
                    f"Only supports numpy ndarray. Got {type(obj).__name__}: {obj}"
                )
            return obj.tolist()

        return pydantic_core.core_schema.with_info_plain_validator_function(
            cls._pydantic_validate,
            serialization=pydantic_core.core_schema.plain_serializer_function_ser_schema(
                serialize, when_used="json-unless-none", info_arg=True
            ),
        )

    @classmethod
    def _pydantic_validate(
        cls,
        value: Union[Any, np.ndarray, Sequence, dict],
        info: pydantic_core.core_schema.ValidationInfo,
    ) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, Sequence):
            return np.asarray(value)
        raise ValueError(f"No conversion from {value!r} to numpy ndarray")


# Annotate numpy arrays to be handled by the custom class
NDArray = Annotated[np.ndarray, _PydanticNDArray]

class _PydanticPmdUnit(BaseModel):
    unitSI: float
    unitSymbol: str
    unitDimension: Tuple[int, ...]

    @staticmethod
    def _from_dict(dct: dict) -> pmd_unit:
        dct = dict(dct)
        dim = dct.pop("unitDimension", None)
        if dim is not None:
            dim = tuple(dim)
        return pmd_unit(**dct, unitDimension=dim)

    def _as_dict(self) -> Dict[str, Any]:
        return {
            "unitSI": self.unitSI,
            "unitSymbol": self.unitSymbol,
            "unitDimension": tuple(self.unitDimension),
        }

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: Type[Any],
        handler: pydantic.GetCoreSchemaHandler,
    ) -> pydantic_core.core_schema.CoreSchema:
        return pydantic_core.core_schema.no_info_plain_validator_function(
            cls._pydantic_validate,
            serialization=pydantic_core.core_schema.plain_serializer_function_ser_schema(
                cls._as_dict, when_used="json-unless-none"
            ),
        )

    @classmethod
    def _pydantic_validate(
        cls, value: Union[Dict[str, Any], pmd_unit, Any]
    ) -> pmd_unit:
        if isinstance(value, pmd_unit):
            return value
        if isinstance(value, dict):
            return cls._from_dict(value)
        raise ValueError(f"No conversion from {value!r} to pmd_unit")

PydanticPmdUnit = Annotated[pmd_unit, _PydanticPmdUnit]
