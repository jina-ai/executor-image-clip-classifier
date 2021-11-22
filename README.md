# CLIPImageClassifier

**CLIPImageClassifier** wraps [clip image model from transformers](https://huggingface.co/openai/clip-vit-base-patch32). 

**CLIPImageClassifier** is initialized with the argument `classes`, these are the texts that we want to classify an image to one of them
The executor receives `Documents` with `uri` attribute. Each Document's uri represent the path to an image. The executor will read the image
and classify it to one of the `classes`. 

The result will be saved inside a new tag called `class` within the original document. The `class` tag is a dictionary that contains two things: 

- `label`: the chosen class from `classes`.
- `score`: the confidence score in the chosen class given by the model.


## Usage

Use the prebuilt images from Jina Hub in your Python code, add it to your Flow and classify your images according to chosen classes:

```python
from jina import Flow
classes = ['this is a cat','this is a dog','this is a person']
f = Flow().add(
    uses='jinahub+docker://CLIPImageClassifier',
    uses_with={'classes':classes}
    )
docs = DocumentArray()
doc = Document(uri='/your/image/path')
docs.append(doc)

with f:
    f.post(on='/classify', inputs=docs, on_done=lambda resp: print(resp.docs[0].tags['class']['label']))
```
## Returns

`Document` with `class` tag. This `class` tag which is a `dict`.It contains `label` which is an `str` and a `float` confidence `score` for the image.

## GPU Usage

This executor also offers a GPU version. To use it, make sure to pass `'device'='cuda'`, as the initialization parameter, and `gpus='all'` when adding the containerized Executor to the Flow. See the [Executor on GPU section of Jina documentation for more details](https://docs.jina.ai/tutorials/gpu-executor/).

Here's how you would modify the example above to use a GPU:

```python
from jina import Flow

classes = ['this is a cat','this is a dog','this is a person']	
f = Flow().add(
    uses='jinahub+docker://CLIPImageClassifier',
    uses_with={
    'classes':classes,
    'device':'cuda',
    'gpus':'all'
    }
    )
docs = DocumentArray()
doc = Document(uri='/your/image/path')
docs.append(doc)

with f:
    f.post(on='/classify', inputs=docs, on_done=lambda resp: print(resp.docs[0].tags['class']['label']))
```

## Reference

[CLIP Image model](https://huggingface.co/openai/clip-vit-base-patch32)
