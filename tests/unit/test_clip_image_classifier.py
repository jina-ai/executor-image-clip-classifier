from clip_image_classifier import CLIPImageClassifier
from jina import Document, DocumentArray, Executor
import pytest
from pathlib import Path


@pytest.fixture(scope='module')
def classifier(
    classes=['this is a cat', 'this is a dog', 'this is a person']
) -> CLIPImageClassifier:
    return CLIPImageClassifier(classes=classes)


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert len(ex.classes) != 0


def test_no_classes():
    with pytest.raises(
        ValueError, match='parameter is mandatory'
    ):
        url = str(Path(__file__).parents[1] / 'imgs' / 'image1.jpg')
        doc = Document(uri=url)
        doc.load_uri_to_image_blob()
        docs = DocumentArray([doc])
        classifier = CLIPImageClassifier()
        classifier.classify(docs, parameters={})


def test_no_documents(classifier: CLIPImageClassifier):
    docs = DocumentArray()
    classifier.classify(docs=docs, parameters={})
    assert len(docs) == 0


def test_one_image(classifier: CLIPImageClassifier):
    url = str(Path(__file__).parents[1] / 'imgs' / 'image1.jpg')
    doc = Document(uri=url)
    doc.load_uri_to_image_blob()
    docs = DocumentArray([doc])

    classifier.classify(docs, parameters={})
    assert len(docs) == 1
    assert dict(docs[0].tags['class'])['label'] == 'this is a person'


def test_http_image(classifier: CLIPImageClassifier):
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    doc = Document(uri=url)
    doc.load_uri_to_image_blob()
    docs = DocumentArray([doc])

    classifier.classify(docs, parameters={})
    assert len(docs) == 1
    assert dict(docs[0].tags['class'])['label'] == 'this is a cat'
