# executor-image-clip-classifier


## Usage

#### via Docker image (recommended)

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://executor-image-clip-classifier')
```

#### via source code

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://executor-image-clip-classifier')
```

- To override `__init__` args & kwargs, use `.add(..., uses_with: {'key': 'value'})`
- To override class metas, use `.add(..., uses_metas: {'key': 'value})`
