# ML Tour
A small library of curated machine learning algorithms implemented using TensorFlow 2.

## Installation

```sh
$ pip install git+https://github.com/swghosh/ml-tour.git
```

## Usage

```python
from ml_tour.kmeans import Means
kmeans = KMeans(k=2)
clusters = kmeans.fit_predict(X)
```

```python
from ml_tour.pca import PCA
pca = PCA(k=10)
X_transformed = pca.fit_transform(X)
```
