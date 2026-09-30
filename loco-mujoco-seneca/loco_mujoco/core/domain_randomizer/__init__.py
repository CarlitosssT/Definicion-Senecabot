from .base import DomainRandomizer
from .no_randomization import NoDomainRandomization
from .default import DefaultRandomizer
from .slope import SlopeRandomizer    # [SENECA LOCAL CHANGE] slope walking via tilted gravity

# register all domain randomizers
NoDomainRandomization.register()
DefaultRandomizer.register()
SlopeRandomizer.register()
