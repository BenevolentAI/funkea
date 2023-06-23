"""File registry for default file paths."""
import zipfile
from pathlib import Path

import pydantic

from funkea.core.utils import version

HOME: Path = Path(__file__).parent.parent.parent.resolve()
RESOURCES: Path = HOME / "core" / "resources"
SCALA_UDF: Path = RESOURCES / f"spark-udf_2.12-{version.__version__}.jar"
FILE_REGISTRY: Path = RESOURCES / "file_registry.json"


class FileRegistry(pydantic.BaseModel):
    r"""Contains the filepaths for commonly used files.

    These filepaths are specified in ``funkea/core/resources/file_registry.json``. These should be
    set by the user (they are ``null`` by default), using the package CLI.

    Notes:
        All of these are optional. Hence, if the paths are not specified, a user will need to pass
        the dataframes or filepaths into the component objects explicitly. We have found it useful
        to use this central object, as it made keeping track easier.
    """
    gtex: str | None = pydantic.Field(
        None,
        title="Default filepath for GTEx tissue gene expression dataset.",
        description="A filepath to the parquet file containing the GTEx expression values, both as "
        "unnormalised (nTPM) and normalised (PEM) values.",
    )
    ld_reference_data: str | None = pydantic.Field(
        None,
        title="Default filepath for LD reference dataset.",
        description="Filepath to the default LD matrix. This is a table in long format, as it "
        "would be output by PLINK, though it by default assumes renamed columns. Also, "
        "it requires ancestral information."
        "See ``funkea.core.data.LDComponent`` for details.",
    )
    depict_null_loci: str | None = pydantic.Field(
        None,
        title="Default filepath for precomputed DEPICT null loci.",
        description="The filepath to the parquet containing the DEPICT-specific null loci. See "
        "``funkea.implementations.depict.NullLoci`` for details on the format. ",
    )
    chromosomes: str | None = pydantic.Field(
        None,
        title="Default filepath for chromosome length information.",
        description="Filepath to parquet containing a map from chromosome identifiers to "
        "chromosome lengths. See ``funkea.core.data.ChromosomeComponent`` for details.",
    )
    snpsea_background_variants: str | None = pydantic.Field(
        None,
        title="Default filepath for the SNPsea background variants.",
        description="The filepath to the variants used in SNPsea to compute the null enrichments "
        "(for significance testing). Should have the minimal sumstats schema "
        "(i.e. rsid, chr, pos, p) and contain LD pruned variants. Moreover, due to the "
        "LD pruning, the variants should also have ancestral information. "
        "See ``funkea.implementations.snpsea.BackgroundVariants`` for details.",
    )
    garfield_control_covariates: str | None = pydantic.Field(
        None,
        title="Default filepath for GARFIELD controlling covariates.",
        description="Filepath to the SNP-level controlling covariates used in the GARFIELD "
        "regression. See ``funkea.implementations.garfield.ControllingCovariates`` for "
        "details.",
    )
    ldsc_controlling_ld_scores: str | None = pydantic.Field(
        None,
        title="Default filepath for LDSC controlling LD scores.",
        description="The filepath to the controlling LD scores used in the LD score regression. "
        "These are LD scores used to control for confounding during enrichment "
        "analysis. This must include ancestry information. See "
        "``funkea.implementations.ldsc.LDScores`` for details.",
    )
    ldsc_weighting_ld_scores: str | None = pydantic.Field(
        None,
        title="Default filepath for LDSC weighting LD scores.",
        description="Filepath to the LD scores used for regression weight computation in LD score "
        "regression. This must contain ancestry information. See "
        "``funkea.implementations.ldsc.LDScores`` for details.",
    )

    @classmethod
    def default(cls) -> "FileRegistry":
        """Loads the default file-registry from disk.

        The default file registry is stored in ``funkea/core/resources/file_registry.json`` and can
        be generated using the CLI.

        Returns:
            The file registry object containing the (optional) default paths.
        """
        filepath = FILE_REGISTRY
        # zipfile will be on top of the funkea package (i.e. the parent)
        # so file path would look like: `funkea.zip/funkea/...`
        if zipfile.is_zipfile(HOME.parent):
            # this is necessary for Python-based UDFs, as otherwise the FileRegistry cannot be
            # initialised in the executors and will cause an error
            import json

            archive = HOME.parent
            with zipfile.ZipFile(archive, "r").open(
                filepath.relative_to(archive).as_posix()
            ) as file:
                obj = json.loads(file.read())

            return cls.parse_obj(obj)
        return cls.parse_file(filepath)


default = FileRegistry.default()
