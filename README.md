# Image Classification with Deep Learning

A compact PyTorch CNN recognizes one handwritten digit, from 0 through 9. The included model classified **268 of 269 held-out images correctly (99.63%)**. The one error was a handwritten 1 classified as a 7.

## Try the trained model

Run these commands from the project directory with Python 3.13.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m digits predict examples/digit-6.png
```

<img src="examples/digit-6.png" alt="A handwritten six from the test set" width="96" height="96">

The sample is image 492 in the dataset. The command predicts **6** and prints the three highest softmax scores. For your own image, crop closely around one digit and keep it upright. Light strokes on a dark background work by default. For dark handwriting on light paper, run `python -m digits predict your-digit.png --dark-ink`. The loader converts the image to grayscale, fits it into an 8 × 8 square while preserving aspect ratio, and applies the training normalization.

## Training and evaluation

```sh
python -m digits train --output runs/retrained.pt
python -m digits evaluate --checkpoint runs/retrained.pt --split validation
python -m digits evaluate --checkpoint runs/retrained.pt --split test --json runs/test.json
python -m digits benchmark --checkpoint runs/retrained.pt --json runs/benchmark.json
```

The commands use one CPU thread by default. Each accepts `--help` for options. Evaluation, prediction, and benchmarking load the included [checkpoint](artifacts/digits.pt) by default.

### Data and model

1. Load 1,797 grayscale, 8 × 8 images from scikit-learn's bundled handwritten digits dataset. NumPy shuffles each class with seed 42 and assigns 1,259 images to training, 269 to validation, and 269 to testing. The split is by image. This experiment does not measure accuracy on an independent group of writers. [Dataset reference](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
2. Scale intensities from 0–16 to 0–1. Compute the mean and standard deviation on the training pixels and reuse both for validation, testing, and prediction. During training, independently rotate each digit by up to 10 degrees and translate it by up to 0.48 pixels on each axis.
3. Train two 3 × 3 convolution blocks with 24 and 48 channels. Each uses batch normalization, ReLU, and 2 × 2 max pooling. A 64-unit hidden layer with 20% dropout feeds the 10-class output. The model has 23,730 trainable parameters.
4. Minimize cross-entropy with AdamW, an initial learning rate of 0.002, weight decay of 0.01, and batches of 64. Reduce the learning rate when validation loss plateaus. Keep the lowest-loss weights and stop after 10 consecutive epochs without an improvement, up to 100 epochs.

### Measured accuracy

| Split | Correct | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Validation | 269 / 269 | 100.00% | 1.0000 |
| Test | 268 / 269 | 99.63% | 0.9963 |

Training stopped at epoch 52 and selected epoch 42. The checkpoint includes its weights, split indices, dataset fingerprint, normalization statistics, run configuration, package versions, and per-epoch history. Evaluation verifies the dataset fingerprint before using the saved indices. Both reports include precision, recall, F1, support per digit, and a confusion matrix with true digits in rows and predictions in columns. [Validation report](artifacts/validation.json) · [Test report](artifacts/test.json)

### Measured CPU performance

| Measurement | Reference | Optimized |
| --- | ---: | ---: |
| Training throughput | 3,414 images/s at batch size 1 | 34,588 images/s at batch size 64 |
| Single-image forward pass, median | 0.0474 ms | 0.0412 ms |
| Single-image forward pass, 95th percentile | 0.0529 ms | 0.0465 ms |

These measurements used Python 3.13.12 and PyTorch 2.15.0 on an arm64 Mac with one PyTorch CPU thread. The benchmark runs 10 warm-up iterations and 200 timed iterations per configuration. Training measures in-memory batches through augmentation, normalization, forward pass, backward pass, and optimizer update.

For inference, the predictor folds each batch normalization layer into its preceding convolution and runs under `torch.inference_mode()`. The latency figures time one forward pass on an already-normalized image with the model in memory. The benchmark checks the fused model's class choices and scores against the original on 64 images before timing. [Benchmark report](artifacts/benchmark.json)
