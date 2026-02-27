# Load the ops shim early to set up _C.ops -> torch.ops.nunchaku mapping
import nunchaku._ops_shim  # noqa: F401, E402

__all__ = [
    "NunchakuFluxTransformer2dModel",
    "NunchakuSanaTransformer2DModel",
    "NunchakuT5EncoderModel",
    "NunchakuFluxTransformer2DModelV2",
    "NunchakuQwenImageTransformer2DModel",
    "NunchakuZImageTransformer2DModel",
]


def __getattr__(name):
    if name in __all__:
        from . import models as _models

        return getattr(_models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
