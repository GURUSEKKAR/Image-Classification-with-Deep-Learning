// Input: JSON rows with 64 normalized pixels and their digit label.
import fs from 'node:fs';
import {probabilities} from '../web/model.mjs';
import {prepareUpload} from '../web/upload.mjs';

const model = JSON.parse(fs.readFileSync(new URL('../web/model.json',import.meta.url)));
const rows = JSON.parse(fs.readFileSync(0));
function predict(pixels) {
  const p = probabilities(pixels, model);
  return p.indexOf(Math.max(...p));
}
for (const [paper, pen] of [[255,0],[205,165],[55,95]]) {
  let before = 0, after = 0;
  for (const {pixels, label} of rows) {
    // Add a one-pixel background margin. Hold digit geometry fixed in both comparisons.
    const rgba = new Uint8ClampedArray(10*10*4);
    const previous = [];
    for (let y=0; y<10; y++) for (let x=0; x<10; x++) {
      const inside = x>0 && x<9 && y>0 && y<9;
      const ink = inside ? pixels[(y-1)*8+x-1] : 0;
      const value = Math.round(paper + ink*(pen-paper));
      rgba.set([value,value,value,255],(y*10+x)*4);
      if (inside) previous.push((paper>=128 ? 255-value : value)/255);
    }
    prepareUpload(rgba,10,10,[0,0,0]);
    const corrected = pixels.map((_,i)=>rgba[((1+(i>>3))*10+1+i%8)*4+3]/255);
    before += predict(previous) === label;
    after += predict(corrected) === label;
  }
  console.log(JSON.stringify({paper,pen,total:rows.length,before,after}));
}
