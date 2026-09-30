from .base import TerminalStateHandler
from .no_terminal import NoTerminalStateHandler
from .height import HeightBasedTerminalStateHandler
from .traj import RootPoseTrajTerminalStateHandler
from .upright import UprightTerminalStateHandler    # [SENECA LOCAL CHANGE] gravity-relative fall detection

# register all terminal state handlers
NoTerminalStateHandler.register()
HeightBasedTerminalStateHandler.register()
RootPoseTrajTerminalStateHandler.register()
UprightTerminalStateHandler.register()
