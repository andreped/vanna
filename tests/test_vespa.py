import os

from dotenv import load_dotenv
from vespa.deployment import VespaDocker
from vespa.package import ApplicationPackage, Field, RankProfile

import vanna
from vanna.openai.openai_chat import OpenAI_Chat
from vanna.vespa import Vespa_VectorStore

load_dotenv()


def setup_vespa():
    app_package = ApplicationPackage(name="text2sql")
    app_package.schema.add_fields(
        Field(
            name="text", type="string", indexing=["index", "summary"], index="enable-bm25",
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

    vespa_docker = VespaDocker(port=8081)
    vespa_app = vespa_docker.deploy(application_package=app_package)

    return vespa_app


class VannaVespa(Vespa_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        Vespa_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
#OPENAI_API_KEY = vanna.get_api_key(email=os.environ['EMAIL'])
print(OPENAI_API_KEY)


# configure vespa
# setup_vespa()


vn_vespa = VannaVespa(config={'api_key': OPENAI_API_KEY, 'model': 'gpt-3.5-turbo'})
vn_vespa.connect_to_sqlite('https://vanna.ai/Chinook.sqlite')

def test_vespa():
    existing_training_data = vn_vespa.get_training_data()
    if len(existing_training_data) > 0:
        for _, training_data in existing_training_data.iterrows():
            vn_vespa.remove_training_data(training_data['id'])

    df_ddl = vn_vespa.run_sql("SELECT type, sql FROM sqlite_master WHERE sql is not null")

    for ddl in df_ddl['sql'].to_list():
        vn_vespa.train(ddl=ddl)

    sql = vn_vespa.generate_sql("What are the top 7 customers by sales?")
    df = vn_vespa.run_sql(sql)
    assert len(df) == 7


test_vespa()
