"""RD-Fuse research tools for conditional RGB-T detection."""
from .model import RDFuse
from .routing import LatencyProfile, Choice, select_choice
__all__ = ['RDFuse', 'LatencyProfile', 'Choice', 'select_choice']
__version__ = '0.3.0'
