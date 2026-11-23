import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {validateModel, probabilities} from '../web/model.mjs';
const model = JSON.parse(readFileSync(new URL('../web/model.json',import.meta.url)));
test('shipped model produces a normalized distribution', () => {
  validateModel(model);
  const scores = probabilities(model.samples[6],model);
  assert.equal(scores.indexOf(Math.max(...scores)),6);
  assert.ok(Math.abs(scores.reduce((a,b)=>a+b)-1)<1e-8);
});
test('reject incomplete models before scoring', () => {
  for(const fields of [{std:0},{layers:[]},{samples:{}},{layers:[{},...model.layers.slice(1)]}])
    assert.throws(()=>validateModel({...model,...fields}));
});
