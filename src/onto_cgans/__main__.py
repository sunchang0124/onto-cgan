import pandas as pd
import pkg_resources
import typer

from onto_cgans import Onto_DP_CGAN
from onto_cgans.ontology_embedding import make_embedding_model

cli = typer.Typer()

BOLD = '\033[1m'
END = '\033[0m'
GREEN = '\033[32m'


@cli.command("gen")
def cli_gen(
    input_file: str,
    embedding_path: str = typer.Option(..., help="Path to the ontology embedding file (.kv)"),
    embedding_size: int = typer.Option(100, help="Size of the ontology embedding vectors"),
    hp_dict: str = typer.Option(..., help="Path to the HP label->IRI dictionary file"),
    rd_dict: str = typer.Option(..., help="Path to the RD label->IRI dictionary file"),
    gen_size: int = typer.Option(100, help="Number of rows in the generated samples file"),
    epochs: int = typer.Option(100, help="Number of epochs"),
    batch_size: int = typer.Option(500, help="Batch size"),
    output: str = typer.Option("synthetic_samples.csv", help="Path to the output"),
    verbose: bool = typer.Option(True, help="Display logs"),
):
    tabular_data = pd.read_csv(input_file)

    embedding = make_embedding_model(
        use_llm=False,
        embedding_path=embedding_path,
        embedding_size=embedding_size,
        hp_dict_fn=hp_dict,
        rd_dict_fn=rd_dict,
    )

    model = Onto_DP_CGAN(
        log_file_path=None,
        embedding=embedding,
        epochs=epochs,
        batch_size=batch_size,
        log_frequency=True,
        verbose=verbose,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        discriminator_lr=2e-4,
        discriminator_steps=1,
        private=False,
        cuda=False,
    )

    if verbose:
        print(f'Model instantiated, fitting...')
    model.fit(tabular_data)

    if verbose:
        print(f'Model fitted, sampling...')
    sample = model.sample(gen_size)

    sample.to_csv(output, index=False)
    if verbose:
        print(f'Samples generated in {BOLD}{GREEN}{output}{END}')


@cli.command("version")
def cli_version():
    print(pkg_resources.get_distribution('onto-cgans').version)


if __name__ == "__main__":
    cli()
