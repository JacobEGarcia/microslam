// geometry.js - pure-JS two-view geometry for microSLAM.
// Normalized 8-point essential matrix (RANSAC + Sampson scoring),
// rank-2 projection, 4-fold pose decomposition with cheirality test,
// and per-point DLT triangulation. Linear algebra via Jacobi
// eigendecomposition of small symmetric matrices. No dependencies.
"use strict";
window.SLAMGEO = (() => {

  const dot3 = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
  const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
  const mul3s = (a, s) => [a[0]*s, a[1]*s, a[2]*s];
  const add3 = (a, b) => [a[0]+b[0], a[1]+b[1], a[2]+b[2]];
  function mul33(A, B) {
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
  const det3 = A =>
    A[0]*(A[4]*A[8]-A[5]*A[7]) - A[1]*(A[3]*A[8]-A[5]*A[6]) + A[2]*(A[3]*A[7]-A[4]*A[6]);

  // Jacobi eigendecomposition of a symmetric n x n matrix (row-major).
  // Returns eigenvalues ascending and eigenvectors as columns of V.
  function jacobiEigen(Ain, n, maxSweeps = 40) {
    const A = Ain.slice();
    const V = new Array(n*n).fill(0);
    for (let i = 0; i < n; i++) V[i*n+i] = 1;
    for (let s = 0; s < maxSweeps; s++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p+1; q < n; q++) off += A[p*n+q]*A[p*n+q];
      if (off < 1e-24) break;
      for (let p = 0; p < n-1; p++) for (let q = p+1; q < n; q++) {
        const apq = A[p*n+q];
        if (Math.abs(apq) < 1e-16) continue;
        const app = A[p*n+p], aqq = A[q*n+q];
        const phi = 0.5 * Math.atan2(2*apq, aqq - app);
        const c = Math.cos(phi), sn = Math.sin(phi);
        for (let i = 0; i < n; i++) {
          const aip = A[i*n+p], aiq = A[i*n+q];
          A[i*n+p] = c*aip - sn*aiq; A[i*n+q] = sn*aip + c*aiq;
        }
        for (let i = 0; i < n; i++) {
          const api = A[p*n+i], aqi = A[q*n+i];
          A[p*n+i] = c*api - sn*aqi; A[q*n+i] = sn*api + c*aqi;
        }
        for (let i = 0; i < n; i++) {
          const vip = V[i*n+p], viq = V[i*n+q];
          V[i*n+p] = c*vip - sn*viq; V[i*n+q] = sn*vip + c*viq;
        }
      }
    }
    const d = [];
    for (let i = 0; i < n; i++) d.push(A[i*n+i]);
    const idx = d.map((v,i) => i).sort((a,b) => d[a]-d[b]);
    const Vs = new Array(n*n);
    for (let col = 0; col < n; col++) for (let r = 0; r < n; r++) Vs[r*n+col] = V[r*n+idx[col]];
    return { d: idx.map(i => d[i]), V: Vs };
  }

  function nullVecN(rows, n) {
    const AtA = new Array(n*n).fill(0);
    for (const r of rows)
      for (let i = 0; i < n; i++) {
        const ri = r[i];
        for (let j = 0; j <= i; j++) AtA[i*n+j] += ri * r[j];
      }
    for (let i = 0; i < n; i++) for (let j = i+1; j < n; j++) AtA[i*n+j] = AtA[j*n+i];
    const { V } = jacobiEigen(AtA, n);
    const out = [];
    for (let i = 0; i < n; i++) out.push(V[i*n]); // first column = smallest eigenvalue
    return out;
  }

  // Normalized 8-point: pts are [[x,y,1],...] in normalized camera coords.
  function eightPoint(x1, x2) {
    if (x1.length < 8) return null;
    const rows = [];
    for (let i = 0; i < x1.length; i++) {
      const [a, b] = [x1[i], x2[i]];
      rows.push([b[0]*a[0], b[0]*a[1], b[0], b[1]*a[0], b[1]*a[1], b[1], a[0], a[1], 1]);
    }
    const e = nullVecN(rows, 9);
    const nrm = Math.hypot(...e);
    if (nrm < 1e-12) return null;
    const E = e.map(v => v / nrm);
    return projectRank2(E);
  }

  // project E to essential manifold: singular values (s,s,0)
  function projectRank2(E) {
    const dec = svdE(E);
    if (!dec) return null;
    const { U, V } = dec;
    // E' = u1 v1' + u2 v2'  (both singular values = 1; overall scale irrelevant)
    const out = new Array(9).fill(0);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
      out[i*3+j] = U[i*3]*V[j*3] + U[i*3+1]*V[j*3+1];
    return out;
  }

  // partial SVD of a 3x3 E with sigma3 ~ 0: returns right-handed U,V
  // whose first two columns span the rank-2 decomposition.
  function svdE(E) {
    const EtE = mul33(tr3(E), E);
    const { d, V } = jacobiEigen(EtE, 3); // ascending sigma^2
    const s2 = Math.sqrt(Math.max(d[2], 0)), s1 = Math.sqrt(Math.max(d[1], 0));
    if (s2 < 1e-12 || s1 < 1e-12) return null;
    const vcol = c => [V[c], V[3+c], V[6+c]];
    const v1 = vcol(2), v2 = vcol(1);
    let u1 = mul3s(mul3v(E, v1), 1/s2);
    let u2 = mul3s(mul3v(E, v2), 1/s1);
    // orthonormalize defensively
    u1 = mul3s(u1, 1/Math.hypot(...u1));
    u2 = mul3s(u2, 1/Math.hypot(...u2));
    const u3 = cross(u1, u2);
    const v3 = cross(v1, v2);
    const U = [u1[0],u2[0],u3[0], u1[1],u2[1],u3[1], u1[2],u2[2],u3[2]]; // columns
    const Vm = [v1[0],v2[0],v3[0], v1[1],v2[1],v3[1], v1[2],v2[2],v3[2]];
    return { U, V: Vm };
  }

  // E -> 4 candidate (R,t) with unit t
  function decomposeE(E) {
    const dec = svdE(E);
    if (!dec) return [];
    const { U, V } = dec;
    const W = [0,-1,0, 1,0,0, 0,0,1];
    const Wt = tr3(W);
    const Vt = tr3(V);
    let R1 = mul33(U, mul33(W, Vt));
    let R2 = mul33(U, mul33(Wt, Vt));
    if (det3(R1) < 0) R1 = R1.map(v => -v);
    if (det3(R2) < 0) R2 = R2.map(v => -v);
    const t = [U[2], U[5], U[8]]; // third column of U
    return [
      { R: R1, t },
      { R: R1, t: mul3s(t, -1) },
      { R: R2, t },
      { R: R2, t: mul3s(t, -1) }
    ];
  }

  // Sampson error of E for correspondence (a, b), normalized coords
  function sampson(E, a, b) {
    const Ea = mul3v(E, a), Etb = mul3v(tr3(E), b);
    const r = dot3(b, Ea);
    return (r*r) / (Ea[0]*Ea[0] + Ea[1]*Ea[1] + Etb[0]*Etb[0] + Etb[1]*Etb[1] + 1e-18);
  }

  // DLT triangulation, normalized coords, P1 = [I|0], P2 = [R|t]
  function triangulateNorm(a, b, R, t) {
    const rows = [
      [-1, 0, a[0], 0],
      [0, -1, a[1], 0],
      [b[0]*R[6]-R[0], b[0]*R[7]-R[1], b[0]*R[8]-R[2], b[0]*t[2]-t[0]],
      [b[1]*R[6]-R[3], b[1]*R[7]-R[4], b[1]*R[8]-R[5], b[1]*t[2]-t[1]]
    ];
    const h = nullVecN(rows, 4);
    if (Math.abs(h[3]) < 1e-12) return null;
    return [h[0]/h[3], h[1]/h[3], h[2]/h[3]];
  }

  function normPts(pts, K) {
    const f = K[0], cx = K[2], cy = K[5];
    return pts.map(p => [(p[0]-cx)/f, (p[1]-cy)/f, 1]);
  }

  // Full pose estimation: pixel-coordinate correspondences -> {R, t, mask, inliers}
  function estimatePose(pts1, pts2, K, opts = {}) {
    const n = pts1.length;
    if (n < 8) return null;
    const maxIter = opts.maxIter || 350;
    const f = K[0];
    const th2 = Math.pow((opts.threshPx || 1.6) / f, 2);
    const x1 = normPts(pts1, K), x2 = normPts(pts2, K);

    let bestMask = null, bestCount = 0, bestE = null;
    const idx = new Array(n);
    for (let i = 0; i < n; i++) idx[i] = i;

    for (let it = 0; it < maxIter; it++) {
      // sample 8 distinct indices
      const s = new Set();
      while (s.size < 8) s.add((Math.random()*n) | 0);
      const si = [...s];
      const E = eightPoint(si.map(i => x1[i]), si.map(i => x2[i]));
      if (!E) continue;
      let count = 0; const mask = new Array(n);
      for (let i = 0; i < n; i++) { const inl = sampson(E, x1[i], x2[i]) < th2; mask[i] = inl ? 1 : 0; if (inl) count++; }
      if (count > bestCount) { bestCount = count; bestMask = mask; bestE = E; }
      if (count > 0.72*n && it > 60) break; // early exit on strong consensus
    }
    if (!bestE || bestCount < 12) return null;

    // least-squares refit on inliers
    const inl1 = [], inl2 = [];
    bestMask.forEach((m, i) => { if (m) { inl1.push(x1[i]); inl2.push(x2[i]); } });
    const Eref = eightPoint(inl1, inl2) || bestE;
    let count = 0; const mask = new Array(n);
    for (let i = 0; i < n; i++) { const inl = sampson(Eref, x1[i], x2[i]) < th2; mask[i] = inl ? 1 : 0; if (inl) count++; }

    // cheirality disambiguation on a subset of inliers
    const sel = [];
    mask.forEach((m, i) => { if (m && sel.length < 48) sel.push(i); });
    let best = null, bestPos = -1;
    for (const { R, t } of decomposeE(Eref)) {
      let pos = 0;
      for (const i of sel) {
        const X = triangulateNorm(x1[i], x2[i], R, t);
        if (!X) continue;
        if (X[2] > 0 && add3(mul3v(R, X), t)[2] > 0) pos++;
      }
      if (pos > bestPos) { bestPos = pos; best = { R, t }; }
    }
    if (!best) return null;
    return { R: best.R, t: best.t, mask, inliers: count, E: Eref };
  }

  // Triangulate one correspondence in pixel coords given K, R, t (unit t)
  function triangulatePx(p1, p2, K, R, t) {
    const a = normPts([p1], K)[0], b = normPts([p2], K)[0];
    return triangulateNorm(a, b, R, t);
  }

  return { estimatePose, triangulatePx, triangulateNorm, decomposeE, eightPoint, jacobiEigen, _test: { mul33, mul3v, tr3, det3, sampson } };
})();
