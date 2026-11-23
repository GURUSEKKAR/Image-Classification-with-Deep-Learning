// Convert the decoded RGBA image into the transparent ink used by the pad.
export function prepareUpload(data, width, height, color) {
  const border = [], visible = [];
  for (let i = 0; i < data.length; i += 4) {
    const value = Math.round(.299 * data[i] + .587 * data[i+1] + .114 * data[i+2]);
    data[i] = value;
    if (data[i+3] < 128) continue;
    visible.push(value);
    const x = (i / 4) % width, y = Math.floor(i / 4 / width);
    if (!x || !y || x === width-1 || y === height-1) border.push(value);
  }
  const median = values => values.sort((a,b) => a-b)[Math.floor(values.length/2)];
  const transparent = border.length < (width+height)/2;
  const background = transparent ? 0 : median(border);
  const dark = transparent ? median(visible) < 128 : background >= 128;
  let low = 255, high = 0;
  for (let i = 0; i < data.length; i += 4) {
    const contrast = transparent ? (dark ? 255-data[i] : data[i])
                                 : Math.max(0, dark ? background-data[i] : data[i]-background);
    const ink = contrast * data[i+3] / 255;
    low = Math.min(low, ink);
    high = Math.max(high, ink);
    data[i] = color[0]; data[i+1] = color[1]; data[i+2] = color[2]; data[i+3] = ink;
  }
  if (high-low < 8) throw Error('Choose an image with a visible digit.');
}
