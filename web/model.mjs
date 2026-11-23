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
