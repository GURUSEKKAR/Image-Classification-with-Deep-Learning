import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

test('new samples cancel older gestures and failed uploads preserve the reading', async () => {
  const model = JSON.parse(readFileSync(new URL('../web/model.json', import.meta.url)));
  const html = readFileSync(new URL('../web/index.html', import.meta.url), 'utf8');
  class Element {
    width = 288;
    children = [];
    style = {};
    setAttribute() {}
    setPointerCapture() {}
    getBoundingClientRect() { return {left:0,top:0,width:288,height:288}; }
    getContext() { return {clearRect(){}, fillRect(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}}; }
    append(...items) { this.children.push(...items); }
    prepend(item) { this.children.unshift(item); }
    replaceChildren() { this.children = []; }
  }
  const nodes = Object.fromEntries([...html.matchAll(/ id="([^"]+)"/g)].map(m=>[m[1], new Element()]));
  globalThis.document = {getElementById:id=>nodes[id], createElement:()=>new Element()};
  globalThis.fetch = async () => ({ok:true,json:async()=>model});
  await import('../web/app.mjs');
  assert.equal(nodes.digit.textContent,6);
  const pointer = {pointerId:1,isPrimary:true,button:0,clientX:30,clientY:40};
  nodes.pad.onpointerdown(pointer);
  assert.equal(nodes.ranks.children.length,0);
  nodes.samples.children[6].onclick();
  nodes.pad.onlostpointercapture(pointer);
  assert.equal(nodes.digit.textContent,6);
  await nodes.file.onchange({target:{files:[{size:11*1024*1024}]}});
  assert.match(nodes.status.textContent,/10 MB/);
  assert.equal(nodes.ranks.children.length,3);
  nodes.clear.onclick();
  assert.equal(nodes.digit.textContent,'–');
});
