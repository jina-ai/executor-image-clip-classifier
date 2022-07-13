from jina import Executor, DocumentArray, requests
from transformers import CLIPProcessor, CLIPModel
from typing import List, Optional
import torch
from jina.logging.logger import JinaLogger


class CLIPImageClassifier(Executor):
    """Classifies an image given a list of text classes using CLIP model"""

    def __init__(
            self,
            classes: Optional[List[str]] = None,
            pretrained_model_name_or_path: str = 'openai/clip-vit-base-patch32',
            device: str = 'cpu',
            batch_size: int = 32,
            traversal_paths: str = '@r',
            *args,
            **kwargs
    ):
        """
        :param classes: List of string that represents the classes an image can belong to
        e.g. ["this is a cat","this is a dog","this is a person"]
        :param pretrained_model_name_or_path: Can be either:
            - A string, the model id of a pretrained CLIP model hosted
                inside a model repo on huggingface.co, e.g., 'openai/clip-vit-base-patch32'
            - A path to a directory containing model weights saved, e.g.,
                ./my_model_directory/
        :param device: Pytorch device to put the model on, e.g. 'cpu', 'cuda', 'cuda:1'
        :param traversal_paths: Default traversal paths for encoding, used if
            the traversal path is not passed as a parameter with the request.
        :param batch_size: Default batch size for encoding, used if the
            batch size is not passed as a parameter with the request.
        """
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger(getattr(self.metas, 'name', self.__class__.__name__))
        self.batch_size = batch_size
        self.traversal_paths = traversal_paths
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.classes = classes
        self.device = device
        self.preprocessor = CLIPProcessor.from_pretrained(
            self.pretrained_model_name_or_path
        )
        self.model = CLIPModel.from_pretrained(self.pretrained_model_name_or_path)
        self.model.to(self.device).eval()

    @requests
    def classify(self, docs: DocumentArray, parameters: dict, **kwargs):
        classes = parameters.get('classes', self.classes)
        if not classes:
            raise ValueError(
                '``classes`` parameter is mandatory. Pass it in parameters'
                'or in constructor. e.g. ["This is a cat","This is a person", "This is a dog"]'
            )
        for docs_batch in docs[
            parameters.get('traversal_paths', self.traversal_paths)].batch(
            batch_size=parameters.get('batch_size', self.batch_size)):
            image_batch = []  # huggingface's clip feature_extractor requires list-like input
            for each_doc in docs_batch:
                image_batch.append(each_doc.tensor)
            with torch.inference_mode():
                input = self._generate_input_features(classes, image_batch)
                outputs = self.model(**input)
                logits_per_image = (
                    outputs.logits_per_image
                )  # this is the image-text similarity score
                probs = logits_per_image.softmax(dim=1)
            for doc, prob in zip(docs_batch, probs):
                prediction = max(prob)
                doc.tags['class'] = {
                    'label': classes[(prob == prediction).nonzero(as_tuple=True)[0]],
                    'score': float(prediction),
                }

    def _generate_input_features(self, classes, images):
        inputs = self.preprocessor(
            text=classes, images=images, return_tensors='pt', padding=True
        )
        inputs = {k: v.to(torch.device(self.device)) for k, v in inputs.items()}
        return inputs