// Fused 3×3 convolution, ReLU and 2×2 max pooling in channel-first order.
function block(input, size, channels, layer) {
  const half = size / 2;
  const output = new Float32Array(layer.b.length * half * half);
  for (let out = 0; out < layer.b.length; out++) {
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        let value = layer.b[out];
        for (let c = 0; c < channels; c++) {
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              const row = y + dy, col = x + dx;
              if (row < 0 || col < 0 || row >= size || col >= size) continue;
              const k = ((out * channels + c) * 3 + dy + 1) * 3 + dx + 1;
              value += layer.w[k] * input[(c * size + row) * size + col];
            }
          }
        }
        const index = (out * half + (y >> 1)) * half + (x >> 1);
        output[index] = Math.max(output[index], value);
      }
    }
  }
  return output;
}
function dense(input, layer) {
  return layer.b.map((bias, out) => {
    let value = bias;
    for (let i = 0; i < input.length; i++) value += input[i] * layer.w[out * input.length + i];
    return value;
  });
}
export function validateModel(model) {
  const shapes = [[216,24], [10368,48], [12288,64], [640,10]];
  if (!Number.isFinite(model?.mean) || !Number.isFinite(model.std) || model.std <= 0 ||
      !Array.isArray(model.layers) || model.layers.length !== shapes.length)
    throw Error('Invalid digit model metadata.');
  for (const [index, [weights, biases]] of shapes.entries()) {
    const layer = model.layers[index];
    if (!Array.isArray(layer?.w) || layer.w.length !== weights || !layer.w.every(Number.isFinite) ||
        !Array.isArray(layer.b) || layer.b.length !== biases || !layer.b.every(Number.isFinite))
      throw Error('Invalid digit model weights.');
  }
  for (let i=0; i<10; i++) {
    const pixels = model.samples?.[i];
    if (!Array.isArray(pixels) || pixels.length !== 64 ||
        !pixels.every(x => Number.isFinite(x) && x >= 0 && x <= 1))
      throw Error('Invalid digit model sample.');
  }
}

export function probabilities(pixels, model) {
  const [conv1, conv2, fc1, fc2] = model.layers;
  const input = Float32Array.from(pixels, x => (x - model.mean) / model.std);
  const features = block(block(input, 8, 1, conv1), 4, 24, conv2);
  const logits = dense(dense(features, fc1).map(x => Math.max(0, x)), fc2);
  const max = Math.max(...logits);
  const exp = logits.map(x => Math.exp(x - max));
  const sum = exp.reduce((a,b) => a+b);
  return exp.map(x => x / sum);
}
