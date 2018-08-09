"""
Tracker
=======

Tracks people in a video using any pose detector.  Two detectors using
different implementations of OpenPose are provided in the detector
package.

"""

from .tracker import Tracker
from .person import Person
from .track_visualiser import TrackVisualiser
from .track import Track
