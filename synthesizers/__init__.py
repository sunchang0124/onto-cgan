from onto_cgan.synthesizers.onto_cgan import ONTOCGANSynthesizer

__all__ = (
    'ONTOCGANSynthesizer'
)


def get_all_synthesizers():
    return {
        name: globals()[name]
        for name in __all__
    }
