import {probabilities} from './model.mjs';
import {prepareUpload} from './upload.mjs';
const $ = id => document.getElementById(id);
const pad = $('pad'), ctx = pad.getContext('2d', {willReadFrequently:true});
const side = pad.width;
let model, drawing = false, replace = true, revision = 0;
const inkColor = [37, 66, 143];
function renderPixels(canvas, pixels, grayscale = false) {
  const context = canvas.getContext('2d');
  context.clearRect(0, 0, 8, 8);
  pixels.forEach((value, index) => {
    const shade = Math.round(value * 255);
    context.fillStyle = grayscale ? `rgb(${shade},${shade},${shade})` : `rgba(${inkColor},${value})`;
    context.fillRect(index % 8, Math.floor(index / 8), 1, 1);
  });
}
function resetResult(message) {
  $('input-preview').getContext('2d').clearRect(0, 0, 8, 8);
  $('digit').textContent = '–';
  $('ranks').replaceChildren();
  $('status').textContent = message;
}
function selectSample(key) {
  for (const button of $('samples').children)
    button.setAttribute('aria-pressed', String(button.textContent === key));
}
function score(pixels, message) {
  if (!model) return;
  renderPixels($('input-preview'), pixels, true);
  const ranks = probabilities(pixels, model).map((p,d) => ({p,d})).sort((a,b) => b.p-a.p);
  $('digit').textContent = ranks[0].d;
  $('status').textContent = message;
  $('ranks').replaceChildren();
  for (const {p,d} of ranks.slice(0,3)) {
    const row = document.createElement('div'), label = document.createElement('span');
    const track = document.createElement('div'), fill = document.createElement('div');
    const value = document.createElement('span');
    row.className = 'row'; track.className = 'track'; fill.className = 'fill';
    label.textContent = d; value.textContent = (100*p).toFixed(1)+'%';
    fill.style.width = 100*p+'%'; track.append(fill); row.append(label,track,value);
    $('ranks').append(row);
  }
}
function showSample(key) {
  ++revision; replace = true; selectSample(key);
  ctx.clearRect(0,0,side,side);
  model.samples[key].forEach((v,i) => {
    ctx.fillStyle = `rgba(${inkColor},${v})`;
    ctx.fillRect(48+(i%8)*24,48+Math.floor(i/8)*24,24,24);
  });
  score(model.samples[key], 'Sample '+key+' analyzed');
}
function analyze(message) {
  const rgba = ctx.getImageData(0,0,side,side).data;
  let left=side, top=side, right=-1, bottom=-1;
  for(let y=0;y<side;y++) for(let x=0;x<side;x++) {
    if(rgba[4*(y*side+x)+3]<28) continue;
    left=Math.min(left,x);right=Math.max(right,x);top=Math.min(top,y);bottom=Math.max(bottom,y);
  }
  if(right<left) return resetResult('Add a visible digit to the canvas.');
  const tile=document.createElement('canvas');tile.width=tile.height=8;
  const tiny=tile.getContext('2d'), w=right-left+1,h=bottom-top+1;
  const scale=Math.min(7/w,8/h);
  tiny.imageSmoothingQuality='high';
  tiny.drawImage(pad,left,top,w,h,(8-w*scale)/2,(8-h*scale)/2,w*scale,h*scale);
  const data=tiny.getImageData(0,0,8,8).data;
  score(Array.from({length:64},(_,i)=>data[i*4+3]/255),message);
}
function point(event) {
  const box=pad.getBoundingClientRect();
  return [(event.clientX-box.left)*side/box.width,(event.clientY-box.top)*side/box.height];
}
pad.onpointerdown = event => {
  if(!model || !event.isPrimary || event.button!==0) return;
  ++revision; selectSample(''); if(replace) ctx.clearRect(0,0,side,side);
  replace=false;drawing=true;pad.setPointerCapture(event.pointerId);
  resetResult('Drawing…');
  ctx.strokeStyle=`rgb(${inkColor})`;ctx.lineWidth=22;ctx.lineCap='round';ctx.lineJoin='round';
  const [x,y]=point(event);ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+.01,y);ctx.stroke();
};
pad.onpointermove = event => {
  if(drawing && event.isPrimary){ctx.lineTo(...point(event));ctx.stroke();}
};
pad.onpointerup = pad.onpointercancel = pad.onlostpointercapture = event => {
  if(drawing && event.isPrimary){drawing=false;analyze('Drawing analyzed');}
};
$('clear').onclick = () => {
  ++revision;drawing=false;replace=false;ctx.clearRect(0,0,side,side);selectSample('');
  resetResult('Waiting for a digit');
};
$('file').onchange = async event => {
  const file=event.target.files[0];event.target.value='';if(!file || !model)return;
  const ticket=++revision;
  let url;
  try {
    if(file.size>10*1024*1024) throw Error('Choose an image smaller than 10 MB.');
    const image=new Image();url=URL.createObjectURL(file);image.src=url;await image.decode();
    if(ticket!==revision)return;
    const scale=Math.min(side/image.width,side/image.height);
    const temp=document.createElement('canvas');temp.width=Math.max(1,Math.round(image.width*scale));
    temp.height=Math.max(1,Math.round(image.height*scale));
    const t=temp.getContext('2d');t.drawImage(image,0,0,temp.width,temp.height);
    const bitmap=t.getImageData(0,0,temp.width,temp.height), data=bitmap.data;
    prepareUpload(data, temp.width, temp.height, inkColor);
    t.putImageData(bitmap,0,0);ctx.clearRect(0,0,side,side);
    ctx.drawImage(temp,(side-temp.width)/2,(side-temp.height)/2);
    replace=true;selectSample('');analyze('Uploaded image analyzed');
  } catch(error){if(ticket===revision)resetResult(error.name==='EncodingError'?'Choose a readable image file.':error.message);}
  finally{if(url)URL.revokeObjectURL(url);}
};
try {
  const response=await fetch('./model.json');if(!response.ok)throw Error();
  model=await response.json();
  for(const key of Object.keys(model.samples)){
    const button=document.createElement('button');button.textContent=key;
    const thumb = document.createElement('canvas');
    thumb.width = thumb.height = 8;
    thumb.setAttribute('aria-hidden', 'true');
    renderPixels(thumb, model.samples[key]);
    button.prepend(thumb);
    button.setAttribute('aria-label','Load sample '+key);button.onclick=()=>showSample(key);
    $('samples').append(button);
  }
  showSample('6');
} catch {resetResult('Model unavailable. Reload the page to try again.');}
