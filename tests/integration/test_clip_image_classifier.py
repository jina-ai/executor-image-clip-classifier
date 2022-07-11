import pytest
import numpy as np
from pathlib import Path

from jina import Document, DocumentArray, Flow
from clip_image_classifier import CLIPImageClassifier


@pytest.mark.parametrize('request_size', [1, 10, 50, 100])
def test_from_uri(request_size: int):
    doc = Document(uri=str(Path(__file__).parents[1] / 'imgs' / 'image1.png'))
    docs = DocumentArray([doc])

    with Flow().add(
            uses=CLIPImageClassifier,
            uses_with={'classes': ['this is a cat', 'this is a dog', 'this is a person']},
    ) as flow:
        results = flow.post(
            on='/index',
            inputs=docs,
            request_size=request_size
        )

    assert len(results) == 1
    assert dict(results[0].tags['class'])['label'] == 'this is a person'


@pytest.mark.parametrize('request_size', [1, 10, 50, 100])
def test_from_tensor(request_size: int):
    doc = Document(tensor=np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    docs = DocumentArray([doc])

    with Flow().add(
            uses=CLIPImageClassifier,
            uses_with={'classes': ['this is a cat', 'this is a dog', 'this is a person']},
    ) as flow:
        results = flow.post(
            on='/index',
            inputs=docs,
            request_size=request_size,
        )

    assert len(results) == 1
