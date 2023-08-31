from ...utils import (
    _LazyModule,
    OptionalDependencyNotAvailable,
    is_torch_available,
    is_transformers_available,
    is_transformers_version,
)

_import_structure = {}
_dummy_objects = {}


try:
    if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.27.0")):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import (
        AudioLDMPipeline,
    )

    _dummy_objects.update({"AudioLDMPipeline": AudioLDMPipeline})

else:
    _import_structure["pipeline_audioldm"] = ["AudioLDMPipeline"]

import sys

sys.modules[__name__] = _LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
    module_spec=__spec__,
)

for name, value in _dummy_objects.items():
    setattr(sys.modules[__name__], name, value)
