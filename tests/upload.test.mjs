import test from 'node:test';
import assert from 'node:assert/strict';
import {prepareUpload} from '../web/upload.mjs';

function tile(background, stroke, alpha = 255, borderAlpha = 255) {
  const data = new Uint8ClampedArray(36);
  for (let pixel = 0; pixel < 9; pixel++) {
    const v = pixel === 4 ? stroke : background;
    data.set([v,v,v,pixel === 4 ? alpha : borderAlpha], 4*pixel);
  }
  return data;
}
test('normalize faint strokes on both light and dark paper', () => {
  for (const [bg, pen] of [[205,171],[54,90],[100,64],[165,204]]) {
    const data = tile(bg,pen);
    prepareUpload(data,3,3,[37,66,143]);
    assert.equal(data[19],255);
    assert.equal(data[3],0);
  }
});
test('detect translucent strokes on transparent margins', () => {
  for (const pen of [0,255]) {
    const data = tile(0,pen,80,0);
    prepareUpload(data,3,3,[0,0,0]);
    assert.equal(data[19],255);
    assert.equal(data[3],0);
  }
});
test('reject uniform images', () => {
  for (const alpha of [0,255]) {
    assert.throws(() => prepareUpload(tile(180,180,alpha,alpha),3,3,[0,0,0]), /visible digit/);
  }
});
