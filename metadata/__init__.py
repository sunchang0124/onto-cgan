"""Metadata module."""

# from sdv.metadata import visualization
# from sdv.metadata.dataset import Metadata
# from sdv.metadata.errors import MetadataError, MetadataNotFittedError
# from sdv.metadata.table import Table

# __all__ = (
#     'Metadata',
#     'MetadataError',
#     'MetadataNotFittedError',
#     'Table',
#     'visualization'
# )


from onto_cgan.metadata import visualization
from onto_cgan.metadata.errors import MetadataNotFittedError
from onto_cgan.metadata.table import Table

__all__ = (
    'MetadataNotFittedError',
    'visualization',
    'Table'
)
