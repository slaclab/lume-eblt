
## Custom Pydantic class to handle numpy ndarray
## Copied from https://github.com/slaclab/lume-genesis/blob/master/genesis/version4/types.py
#class _PydanticNDArray:
#    @classmethod
#    def __get_pydantic_core_schema__(
#        cls,
#        source: Type[Any],
#        handler: pydantic.GetCoreSchemaHandler,
#    ) -> pydantic_core.core_schema.CoreSchema:
#        def serialize(obj: np.ndarray, info: pydantic_core.core_schema.SerializationInfo):
#            if not isinstance(obj, np.ndarray):
#                raise ValueError(
#                    f"Only supports numpy ndarray. Got {type(obj).__name__}: {obj}"
#                )
#            return obj.tolist()
#
#        return pydantic_core.core_schema.with_info_plain_validator_function(
#            cls._pydantic_validate,
#            serialization=pydantic_core.core_schema.plain_serializer_function_ser_schema(
#                serialize, when_used="json-unless-none", info_arg=True
#            ),
#        )
#
#    @classmethod
#    def _pydantic_validate(
#        cls,
#        value: Union[Any, np.ndarray, Sequence, dict],
#        info: pydantic_core.core_schema.ValidationInfo,
#    ) -> np.ndarray:
#        if isinstance(value, np.ndarray):
#            return value
#        if isinstance(value, Sequence):
#            return np.asarray(value)
#        raise ValueError(f"No conversion from {value!r} to numpy ndarray")
#
## Annotate numpy arrays to be handled by the custom class
#NDArray = Annotated[np.ndarray, _PydanticNDArray]