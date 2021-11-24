from clip_image_classifier import CLIPImageClassifier
import pytest
from pathlib import Path
from jina import Document, DocumentArray, Flow
import numpy as np


@pytest.mark.parametrize('request_size', [1, 10, 50, 100])
def test_integration(request_size: int):
    doc = Document(uri=str(Path(__file__).parents[1] / 'imgs' / 'image1.png'))
    docs = DocumentArray([doc])

    with Flow(return_results=True).add(
        uses=CLIPImageClassifier,
        uses_with={'classes': ['this is a cat', 'this is a dog', 'this is a person']},
    ) as flow:
        resp = flow.post(
            on='/index',
            inputs=docs,
            request_size=request_size,
            return_results=True,
        )

    assert sum(len(resp_batch.docs) for resp_batch in resp) == 1
    for r in resp:
        assert dict(r.docs[0].tags['class'])['label'] == 'this is a person'


@pytest.mark.parametrize('request_size', [1, 10, 50, 100])
def test_integration(request_size: int):
    doc = Document(blob=np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    docs = DocumentArray([doc])

    with Flow(return_results=True).add(
        uses=CLIPImageClassifier,
        uses_with={'classes': ['this is a cat', 'this is a dog', 'this is a person']},
    ) as flow:
        resp = flow.post(
            on='/index',
            inputs=docs,
            request_size=request_size,
            return_results=True,
        )

    assert sum(len(resp_batch.docs) for resp_batch in resp) == 1
