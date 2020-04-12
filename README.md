# ML Tour
A small library of curated machine learning algorithms implemented using TensorFlow 2.

## Installation

```sh
pip install git+https://github.com/swghosh/ml-tour.git
```

## Usage

```python
from ml_tour import KMeans
kmeans = KMeans(k=2)
clusters = kmeans.fit_predict(X)
```