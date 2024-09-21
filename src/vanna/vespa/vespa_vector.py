from vespa.package import ApplicationPackage, Field, RankProfile

from ..base import VannaBase


class Vespa_VectorStore(VannaBase):
    def __init__(self, config=None):
        super().__init__(config=config)

        if config is None:
            raise ValueError("config is required")

        self.n_results = config.get("n_results", 3)

        app_package = ApplicationPackage(name="testapp")
        app_package.schema.add_fields(
            Field(
                name="text", type="string", indexing=["index", "summary"], index="enable-bm25"
            ),
            Field(
                name="embedding",
                type="tensor<float>(x[384])",
                indexing=["attribute", "summary"],
                attribute=["distance-metric: angular"],
            ),
        )
        app_package.schema.add_rank_profile(
            RankProfile(
                name="default",
                first_phase="closeness(field, embedding)",
                inputs=[("query(query_embedding)", "tensor<float>(x[384])")],
            )
        )
