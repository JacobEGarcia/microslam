// microSLAM - monocular visual SLAM in the browser.
// Pipeline: GFTT corners -> pyramidal KLT tracking (fwd/bwd verified) ->
// 5-point essential matrix + RANSAC -> pose recovery -> median-depth scale
// propagation -> keyframe triangulation -> sparse 3D map (three.js).

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const W = 640, H = 360;
const MAX_CORNERS = 500, MIN_TRACKS = 110, TOPUP = 320;
const KF_PARALLAX_PX = 28, KF_MAX_AGE = 40, MIN_PARALLAX = 1.6, NEWLM_PARALLAX = 2.6;
const MAX_LANDMARKS = 80000;

const $ = id => document.getElementById(id);
const logEl = $('log');
let logCount = 0;
function log(msg, cls = 'log-ok') {
  const d = document.createElement('div');
  d.className = cls; d.textContent = msg;
  logEl.prepend(d);
  if (++logCount > 90) logEl.removeChild(logEl.lastChild);
}

// ---------- small math helpers ----------
const dot3 = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
const sub3 = (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const add3 = (a, b) => [a[0]+b[0], a[1]+b[1], a[2]+b[2]];
const mul3s = (a, s) => [a[0]*s, a[1]*s, a[2]*s];
const norm3 = a => Math.hypot(a[0], a[1], a[2]);
function mul33(A, B) { // 3x3 * 3x3, row-major
  const C = new Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    C[i*3+j] = A[i*3]*B[j] + A[i*3+1]*B[3+j] + A[i*3+2]*B[6+j];
  return C;
}
const mul3v = (A, v) => [
  A[0]*v[0] + A[1]*v[1] + A[2]*v[2],
  A[3]*v[0] + A[4]*v[1] + A[5]*v[2],
  A[6]*v[0] + A[7]*v[1] + A[8]*v[2]];
const tr3 = A => [A[0],A[3],A[6], A[1],A[4],A[7], A[2],A[5],A[8]];
const I3 = [1,0,0, 0,1,0, 0,0,1];
// CV coords (y down, z fwd) -> three coords (y up)
const cv2three = p => [p[0], -p[1], -p[2]];
const three2cv = p => [p[0], -p[1], -p[2]];

function buildK(f) { return [f,0,W/2, 0,f,H/2, 0,0,1]; }

// ---------- OpenCV loader ----------
function loadOpenCV() {
  return new Promise((resolve, reject) => {
    // NB: only detection + KLT are used from OpenCV (this build omits calib3d
    // pose functions, which is why geometry.js implements them from scratch).
    const urls = [
      'https://docs.opencv.org/4.x/opencv.js',
      'https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.10.0-release.1/dist/opencv.js'
    ];
    let i = 0;
    const tryNext = () => {
      if (i >= urls.length) return reject(new Error('opencv.js failed to load'));
      const s = $('opencv-loader');
      s.src = urls[i++];
      s.onerror = tryNext;
      window.Module = { onRuntimeInitialized() { resolve(); } };
    };
    tryNext();
  });
}

// ---------- SLAM engine ----------
class MonoSLAM {
  constructor(K) {
    this.K = K;
    this.reset();
  }
  reset() {
    this.tracks = [];            // {id, x, y, kx, ky, age, lm}
    this.nextTrackId = 1;
    this.landmarks = new Map();  // id -> {p:[x,y,z], c:[r,g,b]}
    this.nextLmId = 1;
    this.prevGray = null; this.kfGray = null;
    this.Rw = I3.slice();        // camera->world rotation
    this.C = [0,0,0];            // camera center in map frame
    this.kfRw = I3.slice(); this.kfC = [0,0,0];
    this.lastScale = 1;
    this.kfAge = 0; this.kfCount = 0;
    this.distance = 0;
    this.path = [];              // estimated centers (cv coords)
    this.kfPath = [];
    this.frame = 0;
    this.lastInlierRatio = 0; this.lastInliers = 0;
    this.state = 'INIT';
    this.trails = new Map();     // track id -> recent [x,y] for overlay
  }

  detect(gray, existing) {
    const corners = new cv.Mat();
    cv.goodFeaturesToTrack(gray, corners, MAX_CORNERS, 0.01, 9, new cv.Mat(), 3, false, 0.04);
    const out = [];
    const d = corners.data32F;
    for (let i = 0; i < corners.rows; i++) {
      const x = d[2*i], y = d[2*i+1];
      if (existing && existing.some(t => (t.x-x)*(t.x-x) + (t.y-y)*(t.y-y) < 81)) continue;
      out.push({ x, y });
    }
    corners.delete();
    return out;
  }

  process(imgData) {
    // returns per-frame info for overlay + stats
    const info = { inliers: [], outliers: [], fresh: [], accepted: false, kfChanged: false };
    const garbage = [];
    const rgba = cv.matFromImageData(imgData);
    const gray = new cv.Mat();
    garbage.push(rgba, gray);
    cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
    this.frame++;

    try {
      if (!this.kfGray) {
        // bootstrap keyframe
        const pts = this.detect(gray, null);
        for (const p of pts) this.tracks.push({ id: this.nextTrackId++, x: p.x, y: p.y, kx: p.x, ky: p.y, age: 0, lm: 0 });
        this.kfGray = gray.clone();
        this.prevGray = gray.clone();
        this.kfCount = 1;
        this.kfPath.push(this.C.slice());
        this.state = 'TRACKING';
        info.fresh = pts; info.accepted = true; info.kfChanged = true;
        log(`kf 001 | bootstrapped with ${pts.length} corners`, 'log-hi');
        return info;
      }

      // --- KLT track prev -> cur ---
      const n = this.tracks.length;
      const p0 = new cv.Mat(n, 1, cv.CV_32FC2);
      garbage.push(p0);
      const arr = new Float32Array(2*n);
      this.tracks.forEach((t, i) => { arr[2*i] = t.x; arr[2*i+1] = t.y; });
      p0.data32F.set(arr);
      const p1 = new cv.Mat(), st = new cv.Mat(), err = new cv.Mat();
      garbage.push(p1, st, err);
      const crit = new cv.TermCriteria(cv.TermCriteria_COUNT | cv.TermCriteria_EPS, 30, 0.01);
      cv.calcOpticalFlowPyrLK(this.prevGray, gray, p0, p1, st, err, new cv.Size(21,21), 3, crit);

      // forward-backward verification
      const p0r = new cv.Mat(), st2 = new cv.Mat(), err2 = new cv.Mat();
      garbage.push(p0r, st2, err2);
      cv.calcOpticalFlowPyrLK(gray, this.prevGray, p1, p0r, st2, err2, new cv.Size(21,21), 3, crit);

      const d1 = p1.data32F, s1 = st.data, d0r = p0r.data32F;
      const kept = [];
      for (let i = 0; i < n; i++) {
        const t = this.tracks[i];
        const ok = s1[i] === 1;
        const nx = d1[2*i], ny = d1[2*i+1];
        const fb = Math.hypot(d0r[2*i] - t.x, d0r[2*i+1] - t.y);
        if (ok && fb < 1.2 && nx >= 4 && ny >= 4 && nx < W-4 && ny < H-4) {
          // trail history for overlay
          let tr = this.trails.get(t.id);
          if (!tr) { tr = []; this.trails.set(t.id, tr); }
          tr.push([t.x, t.y]); if (tr.length > 7) tr.shift();
          t.x = nx; t.y = ny; t.age++;
          kept.push(t);
        } else {
          this.trails.delete(t.id);
        }
      }
      this.tracks = kept;
      this.prevGray.delete(); this.prevGray = gray.clone();
      this.kfAge++;

      const m = this.tracks.length;
      if (m < 30) {
        // tracking collapsed: relocalize on next frame by forcing a new keyframe attempt
        this.state = 'RELOCALIZING';
        log(`tracking degraded (${m} tracks) - forcing new keyframe`, 'log-warn');
        this.newKeyframe(gray, garbage);
        return info;
      }

      // --- essential matrix between keyframe and current ---
      let medFlow = 0; const flows = [];
      this.tracks.forEach(t => flows.push(Math.hypot(t.x - t.kx, t.y - t.ky)));
      flows.sort((a,b) => a-b); medFlow = flows[m >> 1];

      const pose = SLAMGEO.estimatePose(
        this.tracks.map(t => [t.kx, t.ky]),
        this.tracks.map(t => [t.x, t.y]),
        this.K);
      if (!pose) { this.state = 'TRACKING'; return info; }
      const Rrel = pose.R, trel = pose.t;
      const maskData = pose.mask;
      const inlIdx = [];
      for (let i = 0; i < m; i++) if (maskData[i]) inlIdx.push(i);
      this.lastInliers = pose.inliers;
      this.lastInlierRatio = pose.inliers / m;
      info.inliers = inlIdx.map(i => this.tracks[i]);
      info.outliers = this.tracks.filter((t, i) => !maskData[i]);

      // reject degenerate updates
      if (pose.inliers < 25 || this.lastInlierRatio < 0.35 || medFlow < 0.35) {
        this.state = 'TRACKING';
        return info;
      }

      // --- triangulate inlier tracks (keyframe frame) ---
      const R = Rrel, Rt = tr3(Rrel);
      const triIdx = inlIdx.filter(i => {
        const t = this.tracks[i];
        const par = Math.hypot(t.x - t.kx, t.y - t.ky);
        return par > MIN_PARALLAX;
      });

      let Xkf = null; // Float64Array 3 per triIdx entry (unscaled, keyframe frame)
      if (triIdx.length >= 6) {
        const N = triIdx.length;
        Xkf = new Float64Array(3*N);
        let valid = 0;
        for (let i = 0; i < N; i++) {
          const t = this.tracks[triIdx[i]];
          const X = SLAMGEO.triangulatePx([t.kx, t.ky], [t.x, t.y], this.K, R, trel);
          if (!X) continue;
          const z2 = R[6]*X[0] + R[7]*X[1] + R[8]*X[2] + trel[2];
          if (X[2] > 0.02 && z2 > 0.02 && X[2] < 200) {
            Xkf[3*i] = X[0]; Xkf[3*i+1] = X[1]; Xkf[3*i+2] = X[2]; valid++;
          }
        }
        if (valid < 6) Xkf = null;
      }

      // --- scale from existing landmarks (median depth ratio) ---
      let scale = this.lastScale;
      if (Xkf) {
        const ratios = [];
        const Rwf = tr3(this.kfRw); // world -> kf rotation
        for (let i = 0; i < triIdx.length; i++) {
          const t = this.tracks[triIdx[i]];
          const z = Xkf[3*i+2];
          if (!t.lm || z <= 0) continue;
          const L = this.landmarks.get(t.lm);
          if (!L) continue;
          const rel = sub3(L.p, this.kfC);
          const depthPrev = Rwf[6]*rel[0] + Rwf[7]*rel[1] + Rwf[8]*rel[2];
          if (depthPrev > 0.02) ratios.push(depthPrev / z);
        }
        if (ratios.length >= 6) {
          ratios.sort((a,b) => a-b);
          scale = ratios[ratios.length >> 1];
          if (!isFinite(scale) || scale <= 0 || scale > 1e4) scale = this.lastScale;
        }
      }
      this.lastScale = scale;

      // --- pose update ---
      const Rwt = mul3v(mul33(this.kfRw, Rt), trel); // Rw_kf * R^T * t
      const Cold = this.C.slice();
      this.C = sub3(this.kfC, mul3s(Rwt, scale));
      this.Rw = mul33(this.kfRw, Rt);
      this.distance += norm3(sub3(this.C, Cold));
      info.accepted = true;
      this.state = 'TRACKING';

      // --- landmark creation ---
      if (Xkf && this.landmarks.size < MAX_LANDMARKS) {
        let created = 0;
        const px = imgData.data;
        for (let i = 0; i < triIdx.length; i++) {
          const t = this.tracks[triIdx[i]];
          if (t.lm) continue;
          const z = Xkf[3*i+2];
          if (z <= 0) continue;
          const par = Math.hypot(t.x - t.kx, t.y - t.ky);
          if (par < NEWLM_PARALLAX) continue;
          const Xw = add3(this.kfC, mul3v(this.kfRw, mul3s([Xkf[3*i], Xkf[3*i+1], z], scale)));
          const sx = Math.min(W-1, Math.max(0, t.x|0)), sy = Math.min(H-1, Math.max(0, t.y|0));
          const o = 4*(sy*W + sx);
          const id = this.nextLmId++;
          this.landmarks.set(id, { p: Xw, c: [px[o], px[o+1], px[o+2]] });
          t.lm = id; created++;
        }
        info.newLandmarks = created;
      }

      // --- keyframe decision ---
      if (medFlow > KF_PARALLAX_PX || this.tracks.length < MIN_TRACKS || this.kfAge > KF_MAX_AGE) {
        this.newKeyframe(gray, garbage);
        info.kfChanged = true;
      }
      return info;
    } finally {
      for (const g of garbage) { try { g.delete(); } catch (e) {} }
    }
  }

  newKeyframe(gray, garbage) {
    if (this.kfGray) this.kfGray.delete(); this.kfGray = gray.clone();
    this.kfRw = this.Rw.slice(); this.kfC = this.C.slice();
    this.kfAge = 0; this.kfCount++;
    for (const t of this.tracks) { t.kx = t.x; t.ky = t.y; }
    // top up features
    const pts = this.detect(gray, this.tracks);
    let added = 0;
    for (const p of pts) {
      if (this.tracks.length >= MAX_CORNERS) break;
      this.tracks.push({ id: this.nextTrackId++, x: p.x, y: p.y, kx: p.x, ky: p.y, age: 0, lm: 0 });
      added++;
    }
    this.kfPath.push(this.C.slice());
    log(`kf ${String(this.kfCount).padStart(3,'0')} | +${added} corners | ${this.tracks.length} tracked | map ${this.landmarks.size}`, 'log-hi');
  }
}

// ---------- synthetic factory sequence ----------
function makeFloorTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 512;
  const g = c.getContext('2d');
  g.fillStyle = '#454a52'; g.fillRect(0,0,512,512);
  // concrete speckle
  for (let i = 0; i < 2600; i++) {
    g.fillStyle = `rgba(${30+Math.random()*60|0},${30+Math.random()*60|0},${34+Math.random()*60|0},0.35)`;
    g.fillRect(Math.random()*512, Math.random()*512, 2, 2);
  }
  // expansion joints every 128px (1 m)
  g.strokeStyle = '#2c3036'; g.lineWidth = 3;
  for (let i = 0; i <= 4; i++) {
    g.beginPath(); g.moveTo(i*128, 0); g.lineTo(i*128, 512); g.stroke();
    g.beginPath(); g.moveTo(0, i*128); g.lineTo(512, i*128); g.stroke();
  }
  // painted lane
  g.strokeStyle = '#c9a227'; g.lineWidth = 8; g.setLineDash([46, 30]);
  g.beginPath(); g.moveTo(0, 256); g.lineTo(512, 256); g.stroke();
  g.setLineDash([]);
  // stencil numbers
  g.fillStyle = 'rgba(230,230,225,0.75)';
  g.font = 'bold 84px Arial'; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText('04', 128, 384); g.fillText('05', 384, 128);
  // hazard patch
  for (let i = 0; i < 8; i++) {
    g.fillStyle = i % 2 ? '#c9a227' : '#23262b';
    g.fillRect(300 + i*16, 440, 16, 40);
  }
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(15, 15);
  t.anisotropy = 4;
  return t;
}
function makeHazardTexture() {
  const c = document.createElement('canvas'); c.width = 256; c.height = 256;
  const g = c.getContext('2d');
  g.fillStyle = '#5a5f66'; g.fillRect(0,0,256,256);
  for (let i = 0; i < 256; i += 32) {
    g.fillStyle = (i/32) % 2 ? '#c9a227' : '#23262b';
    g.save(); g.translate(0, 200); g.rotate(-0.5);
    g.fillRect(i - 40, -20, 22, 90); g.restore();
  }
  g.fillStyle = 'rgba(230,230,225,0.85)'; g.font = 'bold 40px Arial'; g.textAlign = 'center';
  g.fillText('P-' + (1 + Math.random()*40|0), 128, 60);
  return new THREE.CanvasTexture(c);
}
function makeCrateTexture(id, base) {
  const c = document.createElement('canvas'); c.width = c.height = 256;
  const g = c.getContext('2d');
  g.fillStyle = base; g.fillRect(0,0,256,256);
  g.strokeStyle = 'rgba(0,0,0,0.45)'; g.lineWidth = 10; g.strokeRect(8,8,240,240);
  g.strokeStyle = 'rgba(255,255,255,0.25)'; g.lineWidth = 3; g.strokeRect(24,24,208,208);
  g.fillStyle = 'rgba(255,255,255,0.9)'; g.font = 'bold 64px Arial'; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText(id, 128, 118);
  g.font = 'bold 22px Arial'; g.fillText('MFG // ZH', 128, 176);
  for (let i = 0; i < 5; i++) { g.fillStyle = 'rgba(0,0,0,0.6)'; g.fillRect(60 + i*28, 216, 14, 20); }
  return new THREE.CanvasTexture(c);
}
function makeBillboard(text, sub, bg) {
  const c = document.createElement('canvas'); c.width = 512; c.height = 256;
  const g = c.getContext('2d');
  g.fillStyle = bg; g.fillRect(0,0,512,256);
  g.fillStyle = 'rgba(255,255,255,0.92)'; g.font = 'bold 92px Arial'; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText(text, 256, 108);
  g.font = 'bold 30px Arial'; g.fillStyle = 'rgba(255,255,255,0.55)';
  g.fillText(sub, 256, 196);
  return new THREE.CanvasTexture(c);
}

class SynthFactory {
  constructor(canvas) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    this.renderer.setSize(W, H, false);
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x14171c);
    this.scene.fog = new THREE.FogExp2(0x14171c, 0.016);
    this.cam = new THREE.PerspectiveCamera(55, W/H, 0.1, 300);
    this.fov = 55;
    this.t = 0;
    this.gtSamples = [];

    const hemi = new THREE.HemisphereLight(0xbfd0e0, 0x30343a, 1.15);
    this.scene.add(hemi);
    const dir = new THREE.DirectionalLight(0xffffff, 1.4);
    dir.position.set(12, 22, 8); this.scene.add(dir);

    // floor
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(60, 60),
      new THREE.MeshLambertMaterial({ map: makeFloorTexture() }));
    floor.rotation.x = -Math.PI/2; this.scene.add(floor);

    // ceiling with light strips
    const ceil = new THREE.Mesh(
      new THREE.PlaneGeometry(60, 60),
      new THREE.MeshBasicMaterial({ color: 0x0c0e11 }));
    ceil.rotation.x = Math.PI/2; ceil.position.y = 7; this.scene.add(ceil);
    const stripMat = new THREE.MeshBasicMaterial({ color: 0xfff3d0 });
    for (let x = -24; x <= 24; x += 8) {
      const strip = new THREE.Mesh(new THREE.PlaneGeometry(0.7, 56), stripMat);
      strip.rotation.x = Math.PI/2; strip.position.set(x, 6.95, 0);
      this.scene.add(strip);
    }

    // pillars
    const hazardTex = makeHazardTexture();
    for (let gx = -2; gx <= 2; gx++) for (let gz = -2; gz <= 2; gz++) {
      if (Math.abs(gx) < 1 && Math.abs(gz) < 1) continue;
      const p = new THREE.Mesh(
        new THREE.CylinderGeometry(0.35, 0.4, 7, 14),
        new THREE.MeshLambertMaterial({ map: hazardTex }));
      p.position.set(gx*9 + (Math.random()-0.5), 3.5, gz*9 + (Math.random()-0.5));
      this.scene.add(p);
    }

    // crates
    const palette = ['#8a4a3a', '#3a5a8a', '#7a7a3a', '#5a6a72', '#6a4a7a', '#8a7a5a'];
    let cid = 1;
    for (let i = 0; i < 42; i++) {
      const w = 0.8 + Math.random()*1.4, h = 0.7 + Math.random()*1.5, d = 0.8 + Math.random()*1.4;
      const tex = makeCrateTexture('C-' + String(cid++).padStart(2,'0'), palette[i % palette.length]);
      const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), new THREE.MeshLambertMaterial({ map: tex }));
      const ang = Math.random()*Math.PI*2, r = 5 + Math.random()*22;
      b.position.set(Math.cos(ang)*r, h/2, Math.sin(ang)*r);
      b.rotation.y = Math.random()*Math.PI;
      this.scene.add(b);
      if (Math.random() < 0.35) { // stacked crate
        const tex2 = makeCrateTexture('C-' + String(cid++).padStart(2,'0'), palette[(i+3) % palette.length]);
        const b2 = new THREE.Mesh(new THREE.BoxGeometry(w*0.8, h*0.7, d*0.8), new THREE.MeshLambertMaterial({ map: tex2 }));
        b2.position.copy(b.position); b2.position.y = h + h*0.35;
        b2.rotation.y = b.rotation.y + 0.3;
        this.scene.add(b2);
      }
    }

    // wall billboards
    const texts = [['SECTOR 7','assembly - north','#274a35'], ['BAY 04','logistics','#24394f'],
      ['DOCK 12','inbound freight','#4f2f24'], ['QA LINE','inspection','#3f3f28'],
      ['GRID 9','storage','#2c3a4a'], ['LIFT','level 2','#463031']];
    texts.forEach(([tx, sub, bg], i) => {
      const tex = makeBillboard(tx, sub, bg);
      const bb = new THREE.Mesh(new THREE.PlaneGeometry(9, 4.5), new THREE.MeshLambertMaterial({ map: tex }));
      const side = i % 4;
      const off = -20 + (i >> 2) * 20 + (i % 2) * 12;
      if (side === 0) { bb.position.set(off, 3.4, -29.5); }
      if (side === 1) { bb.position.set(29.5, 3.4, off); bb.rotation.y = -Math.PI/2; }
      if (side === 2) { bb.position.set(-off, 3.4, 29.5); bb.rotation.y = Math.PI; }
      if (side === 3) { bb.position.set(-29.5, 3.4, -off); bb.rotation.y = Math.PI/2; }
      this.scene.add(bb);
    });

    // closed patrol path
    const pts = [
      [-20,-10], [-8,-15], [6,-14], [16,-16], [22,-8], [20,4],
      [22,14], [10,16], [0,12], [-12,16], [-22,10], [-16,0]
    ].map(([x,z]) => new THREE.Vector3(x, 1.7, z));
    this.curve = new THREE.CatmullRomCurve3(pts, true, 'centripetal', 0.6);
    this.pathLen = this.curve.getLength();
    this.speed = 1.0; // m/s
  }

  step(dt) {
    this.t = (this.t + dt * this.speed / this.pathLen) % 1;
    const pos = this.curve.getPointAt(this.t);
    const tan = this.curve.getTangentAt(this.t);
    const wob = this.t * Math.PI * 2;
    const look = pos.clone().add(tan.clone().multiplyScalar(5));
    look.x += Math.sin(wob * 3.1) * 1.6;
    look.y += Math.sin(wob * 1.7) * 0.35;
    look.z += Math.cos(wob * 2.3) * 1.6;
    this.cam.position.copy(pos);
    this.cam.lookAt(look);
    this.cam.updateMatrixWorld();
    this.renderer.render(this.scene, this.cam);
  }

  // ground-truth camera center relative to frame 0, in cam0's CV frame
  gtRelCV() {
    const e = this.cam.matrixWorld.elements;
    const R3 = [e[0],e[4],e[8], e[1],e[5],e[9], e[2],e[6],e[10]]; // row-major 3x3
    const C3 = [e[12], e[13], e[14]];
    if (!this.R30inv) {
      this.R30 = R3; this.C30 = C3;
      this.R30inv = tr3(R3);
      return [0,0,0];
    }
    const d = sub3(C3, this.C30);
    const rel3 = mul3v(this.R30inv, d);   // in cam0 three-frame
    return three2cv(rel3);                 // cam0 CV frame
  }
}

// ---------- 3D map view ----------
class MapView {
  constructor(canvas) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b0d10);
    this.cam = new THREE.PerspectiveCamera(60, 1, 0.05, 500);
    this.cam.position.set(6, 5, 8);
    this.controls = new OrbitControls(this.cam, canvas);
    this.controls.enableDamping = true;
    this.follow = true;

    this.scene.add(new THREE.GridHelper(60, 60, 0x2a3038, 0x1a1f26));
    const axes = new THREE.AxesHelper(1.5); this.scene.add(axes);

    // point cloud (preallocated)
    this.maxPts = MAX_LANDMARKS;
    this.pPos = new Float32Array(this.maxPts * 3);
    this.pCol = new Float32Array(this.maxPts * 3);
    const pg = new THREE.BufferGeometry();
    pg.setAttribute('position', new THREE.BufferAttribute(this.pPos, 3).setUsage(THREE.DynamicDrawUsage));
    pg.setAttribute('color', new THREE.BufferAttribute(this.pCol, 3).setUsage(THREE.DynamicDrawUsage));
    this.cloud = new THREE.Points(pg, new THREE.PointsMaterial({ size: 0.05, vertexColors: true, sizeAttenuation: true }));
    this.cloud.frustumCulled = false;
    this.scene.add(this.cloud);

    // estimated path
    this.maxPath = 20000;
    this.pathPos = new Float32Array(this.maxPath * 3);
    const lg = new THREE.BufferGeometry();
    lg.setAttribute('position', new THREE.BufferAttribute(this.pathPos, 3).setUsage(THREE.DynamicDrawUsage));
    this.pathLine = new THREE.Line(lg, new THREE.LineBasicMaterial({ color: 0x46e08a }));
    this.pathLine.frustumCulled = false;
    this.scene.add(this.pathLine);

    // ground truth path
    this.gtPos = new Float32Array(this.maxPath * 3);
    const gg = new THREE.BufferGeometry();
    gg.setAttribute('position', new THREE.BufferAttribute(this.gtPos, 3).setUsage(THREE.DynamicDrawUsage));
    this.gtLine = new THREE.Line(gg, new THREE.LineBasicMaterial({ color: 0x8a919c, transparent: true, opacity: 0.7 }));
    this.gtLine.frustumCulled = false;
    this.scene.add(this.gtLine);

    // keyframe markers
    this.kfGroup = new THREE.Group(); this.scene.add(this.kfGroup);

    // camera frustum
    const fg = new THREE.BufferGeometry();
    this.fPos = new Float32Array(8 * 3 * 2 + 12); // 8 edges
    fg.setAttribute('position', new THREE.BufferAttribute(this.fPos, 3).setUsage(THREE.DynamicDrawUsage));
    this.frustum = new THREE.LineSegments(fg, new THREE.LineBasicMaterial({ color: 0x5ad1ff }));
    this.frustum.frustumCulled = false;
    this.scene.add(this.frustum);

    window.addEventListener('resize', () => this.resize());
    this.resize();
  }
  resize() {
    const el = this.renderer.domElement.parentElement;
    if (!el) return;
    const w = el.clientWidth, h = el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.cam.aspect = w / h; this.cam.updateProjectionMatrix();
  }
  updateCloud(landmarks) {
    let i = 0;
    for (const L of landmarks.values()) {
      if (i >= this.maxPts) break;
      const p = cv2three(L.p);
      this.pPos[3*i] = p[0]; this.pPos[3*i+1] = p[1]; this.pPos[3*i+2] = p[2];
      this.pCol[3*i] = L.c[0]/255; this.pCol[3*i+1] = L.c[1]/255; this.pCol[3*i+2] = L.c[2]/255;
      i++;
    }
    this.cloud.geometry.setDrawRange(0, i);
    this.cloud.geometry.attributes.position.needsUpdate = true;
    this.cloud.geometry.attributes.color.needsUpdate = true;
  }
  updatePath(line, attr, positions, sAlign) {
    const n = Math.min(positions.length, this.maxPath);
    for (let i = 0; i < n; i++) {
      const p = cv2three(mul3s(positions[i], sAlign));
      attr[3*i] = p[0]; attr[3*i+1] = p[1]; attr[3*i+2] = p[2];
    }
    line.geometry.setDrawRange(0, n);
    line.geometry.attributes.position.needsUpdate = true;
  }
  updateKfs(kfPath, sAlign) {
    while (this.kfGroup.children.length < kfPath.length) {
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.09, 8, 8),
        new THREE.MeshBasicMaterial({ color: 0xffc454 }));
      this.kfGroup.add(m);
    }
    this.kfGroup.children.forEach((m, i) => {
      const p = cv2three(mul3s(kfPath[i], sAlign));
      m.position.set(p[0], p[1], p[2]);
    });
  }
  updateFrustum(Rw, C, K) {
    const f = K[0], cx = K[2], cy = K[5];
    const corners = [[0,0],[W,0],[W,H],[0,H]];
    const len = 0.6;
    const pts3 = [];
    const Cc = cv2three(C);
    for (const [u,v] of corners) {
      let d = [(u-cx)/f, (v-cy)/f, 1];
      d = mul3s(d, len / norm3(d));
      const w = add3(C, mul3v(Rw, d));
      pts3.push(cv2three(w));
    }
    let o = 0;
    const edge = (a, b) => {
      this.fPos[o++] = a[0]; this.fPos[o++] = a[1]; this.fPos[o++] = a[2];
      this.fPos[o++] = b[0]; this.fPos[o++] = b[1]; this.fPos[o++] = b[2];
    };
    for (const p of pts3) edge(Cc, p);
    for (let i = 0; i < 4; i++) edge(pts3[i], pts3[(i+1)%4]);
    this.frustum.geometry.attributes.position.needsUpdate = true;
  }
  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.cam);
  }
}

// ---------- controller ----------
const feedCanvas = $('feed-canvas');
const overlay = $('overlay-canvas');
const octx = overlay.getContext('2d');
const snapCanvas = document.createElement('canvas');
snapCanvas.width = W; snapCanvas.height = H;
const sctx = snapCanvas.getContext('2d', { willReadFrequently: true });

let slam = null, mapView = null, factory = null;
let mode = 'synth';
let running = false, cvReady = false;
let video = null, stream = null;
let K = buildK((W/2) / Math.tan(55 * Math.PI/360));
let gtSamples = [];       // ground truth rel positions (cv frame)
let sAlign = 1;
let lastT = 0, fpsEMA = 0;
let alignError = null;

function setState(txt, cls) {
  const c = $('state-chip'); c.textContent = txt; c.className = cls || '';
}

function computeAlignment() {
  // similarity scale between est path and gt samples (origins coincide by construction)
  const n = Math.min(gtSamples.length, slam.path.length);
  if (n < 20) return;
  let num = 0, den = 0;
  for (let i = 1; i < n; i++) {
    num += dot3(gtSamples[i], slam.path[i]);
    den += dot3(slam.path[i], slam.path[i]);
  }
  if (den < 1e-9) return;
  sAlign = num / den;
  let err = 0;
  for (let i = 1; i < n; i++) err += norm3(sub3(gtSamples[i], mul3s(slam.path[i], sAlign)));
  alignError = err / (n - 1);
}

function drawOverlay(info) {
  octx.clearRect(0, 0, W, H);
  // trails
  octx.lineWidth = 1;
  for (const tr of slam.trails.values()) {
    if (tr.length < 2) continue;
    octx.strokeStyle = 'rgba(90,209,255,0.35)';
    octx.beginPath();
    octx.moveTo(tr[0][0], tr[0][1]);
    for (const p of tr) octx.lineTo(p[0], p[1]);
    octx.stroke();
  }
  const dot = (x, y, col, r) => { octx.fillStyle = col; octx.beginPath(); octx.arc(x, y, r, 0, 7); octx.fill(); };
  if (info) {
    for (const t of info.outliers) dot(t.x, t.y, 'rgba(255,85,102,0.85)', 2.2);
    for (const t of info.inliers) dot(t.x, t.y, 'rgba(70,224,138,0.9)', 2);
    for (const p of info.fresh) dot(p.x, p.y, 'rgba(90,209,255,0.9)', 2);
  } else {
    for (const t of slam.tracks) dot(t.x, t.y, 'rgba(70,224,138,0.9)', 2);
  }
  if (slam.state === 'RELOCALIZING') {
    octx.fillStyle = 'rgba(255,196,84,0.95)';
    octx.font = 'bold 15px ui-monospace, monospace';
    octx.fillText('RELOCALIZING', 12, H - 14);
  }
}

function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.1, (now - lastT) / 1000) || 0.016;
  lastT = now;
  if (!cvReady || !running || !slam) return;

  // 1. produce source pixels
  if (mode === 'synth') {
    factory.step(dt);
    sctx.drawImage(feedCanvas, 0, 0, W, H);
  } else if (video && video.readyState >= 2) {
    sctx.drawImage(video, 0, 0, W, H);
  } else {
    return;
  }
  const imgData = sctx.getImageData(0, 0, W, H);

  // 2. SLAM step
  const t0 = performance.now();
  let info = null;
  try {
    info = slam.process(imgData);
  } catch (e) {
    console.error(e);
    log('pipeline exception: ' + e.message, 'log-warn');
  }
  slam.path.push(slam.C.slice());
  if (slam.path.length > 19000) { slam.path.splice(0, 6000); gtSamples.splice(0, 6000); }
  const procMs = performance.now() - t0;
  const fps = 1000 / Math.max(1, now - (frame._p || now - 16));
  frame._p = now;
  fpsEMA = fpsEMA ? fpsEMA*0.9 + fps*0.1 : fps;

  // 3. ground truth + alignment (synthetic)
  if (mode === 'synth') {
    gtSamples.push(factory.gtRelCV());
    if (slam.frame % 15 === 0) computeAlignment();
  }

  // 4. draw feed overlay
  drawOverlay(info);

  // 5. map view (throttle cloud rebuild)
  if (slam.frame % 10 === 0 || (info && info.newLandmarks)) mapView.updateCloud(slam.landmarks);
  mapView.updatePath(mapView.pathLine, mapView.pathPos, slam.path, 1);
  if (mode === 'synth') mapView.updatePath(mapView.gtLine, mapView.gtPos, gtSamples, sAlign);
  mapView.updateKfs(slam.kfPath, 1);
  mapView.updateFrustum(slam.Rw, slam.C, slam.K);
  if (mapView.follow && slam.path.length > 1) {
    const p = cv2three(slam.C);
    mapView.controls.target.set(p[0], p[1], p[2]);
  }
  mapView.render();

  // 6. stats
  if (slam.frame % 6 === 0) {
    $('st-fps').textContent = fpsEMA.toFixed(0) + ' (' + procMs.toFixed(0) + ' ms)';
    $('st-tracks').textContent = slam.tracks.length;
    $('st-inliers').textContent = slam.lastInliers + ' (' + (slam.lastInlierRatio*100).toFixed(0) + '%)';
    $('st-landmarks').textContent = slam.landmarks.size;
    $('st-kfs').textContent = slam.kfCount;
    $('st-dist').textContent = slam.distance.toFixed(2) + (mode === 'synth' ? ' m*' : ' u');
    $('st-pos').textContent = slam.C.map(v => v.toFixed(2)).join(', ');
    $('st-scale').textContent = slam.lastScale.toFixed(3);
    if (mode === 'synth' && alignError !== null) {
      $('stat-ate').style.display = '';
      $('st-ate').textContent = alignError.toFixed(3) + ' m (scale ' + sAlign.toFixed(3) + ')';
    }
    setState(slam.state, slam.state === 'TRACKING' ? '' : 'warn');
  }
}

// ---------- source switching ----------
function stopVideo() {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  if (video) { video.pause(); video.srcObject = null; video.src = ''; }
}
function activate(which) {
  mode = which;
  for (const b of ['btn-synth','btn-webcam','btn-video']) $(b).classList.remove('active');
  $({ synth: 'btn-synth', webcam: 'btn-webcam', video: 'btn-video' }[which]).classList.add('active');
  stopVideo();
  gtSamples = []; sAlign = 1; alignError = null;
  $('stat-ate').style.display = 'none';
  mapView.gtLine.geometry.setDrawRange(0, 0);

  if (which === 'synth') {
    K = buildK((W/2) / Math.tan(factory.fov * Math.PI/360));
    slam.K = K;
    factory.R30inv = null;
    log('source: synthetic factory floor (known intrinsics, ground truth on)', 'log-ok');
  } else if (which === 'webcam') {
    K = buildK(1.15 * W); slam.K = K;
    navigator.mediaDevices.getUserMedia({ video: { width: W, height: H } })
      .then(s => {
        stream = s;
        if (!video) video = document.createElement('video');
        video.srcObject = s; video.play();
        log('source: webcam live', 'log-ok');
      })
      .catch(e => { log('webcam denied: ' + e.message, 'log-warn'); setState('NO CAMERA', 'err'); });
  }
  slam.reset();
  mapView.updateCloud(slam.landmarks);
  mapView.updatePath(mapView.pathLine, mapView.pathPos, [], 1);
  mapView.updateKfs([], 1);
  setState('TRACKING');
}

$('btn-synth').onclick = () => activate('synth');
$('btn-webcam').onclick = () => activate('webcam');
$('btn-video').onclick = () => {
  if (mode !== 'video') { mode = 'video'; }
  $('file-input').click();
};
$('file-input').onchange = e => {
  const f = e.target.files[0];
  if (!f) return;
  stopVideo();
  if (!video) video = document.createElement('video');
  video.src = URL.createObjectURL(f);
  video.loop = true; video.muted = true; video.play();
  for (const b of ['btn-synth','btn-webcam','btn-video']) $(b).classList.remove('active');
  $('btn-video').classList.add('active');
  K = buildK(1.15 * W); slam.K = K;
  gtSamples = []; sAlign = 1; alignError = null; $('stat-ate').style.display = 'none';
  slam.reset(); mapView.updateCloud(slam.landmarks);
  log('source: video file "' + f.name + '" (intrinsics estimated)', 'log-ok');
  setState('TRACKING');
};
$('btn-pause').onclick = () => {
  running = !running;
  $('btn-pause').textContent = running ? 'Pause' : 'Resume';
  setState(running ? 'TRACKING' : 'PAUSED', running ? '' : 'warn');
};
$('btn-reset').onclick = () => {
  slam.reset(); gtSamples = []; sAlign = 1; alignError = null;
  if (mode === 'synth') factory.R30inv = null;
  mapView.updateCloud(slam.landmarks);
  mapView.updatePath(mapView.pathLine, mapView.pathPos, [], 1);
  mapView.updateKfs([], 1);
  log('map reset', 'log-warn');
};

// map view click toggles follow
$('map-canvas').addEventListener('pointerdown', () => { if (mapView) mapView.follow = false; });

// ---------- boot ----------
(async () => {
  mapView = new MapView($('map-canvas'));
  factory = new SynthFactory(feedCanvas);
  slam = new MonoSLAM(K);
  requestAnimationFrame(frame);
  try {
    await loadOpenCV();
    cvReady = true;
    $('cv-status').style.display = 'none';
    log('opencv.js ready - pipeline live', 'log-hi');
    log('source: synthetic factory floor (known intrinsics, ground truth on)', 'log-ok');
    running = true;
    setState('TRACKING');
  } catch (e) {
    $('cv-status-text').textContent = 'opencv.js failed to load - check connection';
    setState('CV LOAD FAILED', 'err');
  }
  // expose for automated verification
  window.__SLAM = {
    stats: () => ({
      frame: slam.frame, tracks: slam.tracks.length, landmarks: slam.landmarks.size,
      kfs: slam.kfCount, distance: slam.distance, ate: alignError, sAlign,
      inliers: slam.lastInliers, inlierRatio: slam.lastInlierRatio, state: slam.state, mode
    })
  };
})();
