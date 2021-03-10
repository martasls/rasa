import os
import pytest

from rasa.nlu import registry, train
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Interpreter, Trainer
from tests.nlu.conftest import DEFAULT_DATA_PATH
from tests.train import pipelines_for_tests, pipelines_for_non_windows_tests


def test_all_components_are_in_at_least_one_test_pipeline():
    """There is a template that includes all components to
    test the train-persist-load-use cycle. Ensures that
    really all components are in there.
    """
    all_pipelines = pipelines_for_tests() + pipelines_for_non_windows_tests()
    all_components = [c["name"] for _, p in all_pipelines for c in p]

    for cls in registry.component_classes:
        if "convert" in cls.name.lower():
            # TODO
            #   skip ConveRTTokenizer and ConveRTFeaturizer as the ConveRT model is not
            #   publicly available anymore
            #   (see https://github.com/RasaHQ/rasa/issues/6806)
            continue
        assert (
            cls.name in all_components
        ), "`all_components` template is missing component."


@pytest.mark.timeout(600)
@pytest.mark.parametrize("language, pipeline", pipelines_for_tests())
def test_load_and_persist_without_train(language, pipeline, component_builder, tmpdir):
    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": language})

    trainer = Trainer(_config, component_builder)
    persisted_path = trainer.persist(tmpdir.strpath)

    loaded = Interpreter.load(persisted_path, component_builder)

    assert loaded.pipeline
    assert loaded.parse("Rasa is great!") is not None


@pytest.mark.timeout(600)
@pytest.mark.parametrize("language, pipeline", pipelines_for_non_windows_tests())
@pytest.mark.skip_on_windows
def test_load_and_persist_without_train_non_windows(
    language, pipeline, component_builder, tmpdir
):
    test_load_and_persist_without_train(language, pipeline, component_builder, tmpdir)


async def test_train_model_empty_pipeline(component_builder):
    _config = RasaNLUModelConfig({"pipeline": None, "language": "en"})

    with pytest.raises(ValueError):
        await train(
            _config, data=DEFAULT_DATA_PATH, component_builder=component_builder
        )


async def test_train_named_model(component_builder, tmpdir):
    _config = RasaNLUModelConfig(
        {"pipeline": [{"name": "KeywordIntentClassifier"}], "language": "en"}
    )

    (trained, _, persisted_path) = await train(
        _config,
        path=tmpdir.strpath,
        data=DEFAULT_DATA_PATH,
        component_builder=component_builder,
    )

    assert trained.pipeline

    normalized_path = os.path.dirname(os.path.normpath(persisted_path))
    # should be saved in a dir named after a project
    assert normalized_path == tmpdir.strpath


async def test_handles_pipeline_with_non_existing_component(
    component_builder, pretrained_embeddings_spacy_config
):
    pretrained_embeddings_spacy_config.pipeline.append({"name": "my_made_up_component"})

    with pytest.raises(Exception) as execinfo:
        await train(
            pretrained_embeddings_spacy_config,
            data=DEFAULT_DATA_PATH,
            component_builder=component_builder,
        )
    assert "Cannot find class" in str(execinfo.value)


async def test_train_model_training_data_persisted(component_builder, tmpdir):
    _config = RasaNLUModelConfig(
        {"pipeline": [{"name": "KeywordIntentClassifier"}], "language": "en"}
    )

    (trained, _, persisted_path) = await train(
        _config,
        path=tmpdir.strpath,
        data=DEFAULT_DATA_PATH,
        component_builder=component_builder,
        persist_nlu_training_data=True,
    )

    assert trained.pipeline

    loaded = Interpreter.load(persisted_path, component_builder)

    assert loaded.pipeline
    assert loaded.model_metadata.get("training_data") is not None


async def test_train_model_no_training_data_persisted(component_builder, tmpdir):
    _config = RasaNLUModelConfig(
        {"pipeline": [{"name": "KeywordIntentClassifier"}], "language": "en"}
    )

    (trained, _, persisted_path) = await train(
        _config,
        path=tmpdir.strpath,
        data=DEFAULT_DATA_PATH,
        component_builder=component_builder,
        persist_nlu_training_data=False,
    )

    assert trained.pipeline

    loaded = Interpreter.load(persisted_path, component_builder)

    assert loaded.pipeline
    assert loaded.model_metadata.get("training_data") is None
