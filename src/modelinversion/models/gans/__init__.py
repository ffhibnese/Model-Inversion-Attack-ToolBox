from .base import BaseImageGenerator, BaseIntermediateImageGenerator
from .simple import (
    SimpleGenerator64,
    SimpleGenerator256,
    GmiDiscriminator64,
    GmiDiscriminator256,
    KedmiDiscriminator64,
    KedmiDiscriminator256,
)
from .cgan import (
    PlgmiGenerator64,
    PlgmiGenerator256,
    PlgmiDiscriminator64,
    PlgmiDiscriminator256,
    LoktDiscriminator64,
    LoktDiscriminator256,
    LoktGenerator64,
    LoktGenerator256,
)
from .stylegan2ada import get_stylegan2ata_generator
