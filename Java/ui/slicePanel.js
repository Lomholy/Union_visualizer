import * as THREE from 'three';

// ----- constants -----
const TEXELS_PER_OBJECT = 7;   // must match shader packing
const MAX_OBJECTS = 256;       // shader loop upper bound - increase if needed

// ----- helpers: type mapping -----
function typeToInt(typeString){
  switch(typeString){
    case 'Union_sphere': return 0;
    case 'Union_cylinder': return 1;
    case 'Union_box': return 2;
    case 'Union_cone': return 3;
    default: return -1;
  }
}

// Build a DataTexture packing all objects (one row per object)
function buildObjectDataTexture(objects){
  const H = Math.max(1, objects.length);
  const W = TEXELS_PER_OBJECT;
  const floatCount = W * H * 4;
  const arr = new Float32Array(floatCount);

  const rowSize = W * 4; // floats per row
  for(let i=0;i<objects.length && i < MAX_OBJECTS;i++){
    const obj = objects[i];
    obj.updateMatrixWorld(true);

    const inv = new THREE.Matrix4().copy(obj.matrixWorld).invert(); // inverse world matrix
    const m = inv.elements; // 16 floats (column-major as three.js stores them)

    const base = i * rowSize;

    // texel 0: type, priority, unused, unused
    arr[base + 0] = typeToInt(obj.userData.type || '');
    arr[base + 1] = obj.userData.priority || 0;
    arr[base + 2] = 0;
    arr[base + 3] = 0;

    // texel 1..4: 16 floats of inverse matrix (4 floats per texel)
    // We'll store them sequentially: texel1 = m[0..3], texel2 = m[4..7], ...
    for(let t=0;t<4;t++){
      const texelOffset = base + (1 + t) * 4;
      const mi = t * 4;
      arr[texelOffset + 0] = m[mi + 0];
      arr[texelOffset + 1] = m[mi + 1];
      arr[texelOffset + 2] = m[mi + 2];
      arr[texelOffset + 3] = m[mi + 3];
    }

    // texel 5: shape params (radius, height, width, depth)
    const pOff = base + 5 * 4;
    arr[pOff + 0] = obj.userData.radius || obj.userData.radiu_bottom || 0;
    arr[pOff + 1] = obj.userData.yheight || 0;
    arr[pOff + 2] = obj.userData.xwidth || obj.userData.radius_top || 0;
    arr[pOff + 3] = obj.userData.zdepth || 0;

    // texel 6: color.r,g,b,alpha
    const cOff = base + 6 * 4;
    const col = obj.material && obj.material.color ? obj.material.color : new THREE.Color(0x000000);
    arr[cOff + 0] = col.r;
    arr[cOff + 1] = col.g;
    arr[cOff + 2] = col.b;
    arr[cOff + 3] = (obj.material && obj.material.opacity !== undefined) ? obj.material.opacity : 1.0;
  }

  const tex = new THREE.DataTexture(arr, W, H, THREE.RGBAFormat, THREE.FloatType);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;
  return tex;
}

// ----- GLSL -----
const vertexShader = `
    // RawShaderMaterial: we must declare attributes explicitly
    attribute vec3 position;
    attribute vec2 uv;

    varying vec2 vUv;

    void main(){
    vUv = uv;
    gl_Position = vec4(position, 1.0); // use full vec3 position
    }

`;

// fragment shader does the per-pixel per-object loop and picks highest priority
const fragmentShader = `
  precision highp float;
  varying vec2 vUv;

  uniform sampler2D u_dataTex;
  uniform vec2 u_texSize;    // (TEXELS_PER_OBJECT, objectCount)
  uniform float u_planeWidth;
  uniform float u_planeHeight;
  uniform float u_planeY;
  uniform int u_objectCount;

  // packing constants (must match JS)
  const int TEXELS_PER_OBJECT = ${TEXELS_PER_OBJECT};
  const int MAX_OBJECTS = ${MAX_OBJECTS};

  // helper to fetch exact texel (nearest)
  vec4 fetchTex(int row, int col){
    // convert integer coords to normalized UV into the data texture
    float fx = (float(col) + 0.5) / u_texSize.x;
    float fy = (float(row) + 0.5) / u_texSize.y;
    return texture2D(u_dataTex, vec2(fx, fy));
  }

  // shape tests in object local space
  bool sphereContains(vec3 local, float radius){
    return (dot(local, local) <= radius * radius);
  }
  bool cylinderContains(vec3 local, float radius, float height){
    float r = length(vec2(local.x, local.z));
    return (r <= radius && local.y >= -height*0.5 && local.y <= height*0.5);
  }
  bool boxContains(vec3 local, float w, float h, float d){
    return (abs(local.x) <= w*0.5 && abs(local.y) <= h*0.5 && abs(local.z) <= d*0.5);
  }
    bool coneContains(vec3 local, float bottomR, float topR, float height){
        // compute fraction along the cone axis
        float yBottom = -height * 0.5;
        float yTop = height * 0.5;

        // check if point is inside Y bounds
        if(local.y < yBottom || local.y > yTop) return false;

        // linear interpolation of radius at this Y
        float t = (local.y - yBottom) / height; // 0 at bottom, 1 at top
        float rAtY = mix(bottomR, topR, t);

        // radial distance in XZ plane
        float r = length(local.xz);

        return r <= rAtY;
    }


  void main(){
    // map vUv -> world plane coordinates (same mapping as your CPU code)
    float worldX = vUv.x * u_planeWidth - u_planeWidth * 0.5;
    float worldZ = u_planeHeight * 0.5 - vUv.y * u_planeHeight;
    vec3 worldPos = vec3(worldX, u_planeY, worldZ);

    vec3 outColor = vec3(0.0);
    float outAlpha = 0.0;
    float bestPriority = -1.0;

    // loop objects (bounded)
    for(int i = 0; i < MAX_OBJECTS; i++){
      if(i >= u_objectCount) break;

      int row = i;
      // read texel 0: type, priority
      vec4 t0 = fetchTex(row, 0);
      int type = int(t0.    x + 0.5);
      float priority = t0.y;

      if(priority < bestPriority) {
        // skip: this object's priority is lower than current best
        // (we still need to test >= in case equal might overwrite, adjust if you prefer)
      }

      // read inverse matrix from texels 1..4 (each texel is a vec4)
      vec4 c0 = fetchTex(row, 1);
      vec4 c1 = fetchTex(row, 2);
      vec4 c2 = fetchTex(row, 3);
      vec4 c3 = fetchTex(row, 4);
      // We stored matrix elements column-major across these 4 vec4s.
      mat4 invMat = mat4(c0, c1, c2, c3);

      // transform worldPos to local space
      vec4 local4 = invMat * vec4(worldPos, 1.0);
      vec3 local = local4.xyz;

      // read params (texel 5)
      vec4 params = fetchTex(row, 5);
      float radius = params.x;
      float height = params.y;
      float width = params.z;
      float depth = params.w;

      // read color (texel 6)
      vec4 col = fetchTex(row, 6);

      bool hit = false;
      if(type == 0){
        // sphere
        hit = sphereContains(local, radius);
      } else if(type == 1){
        hit = cylinderContains(local, radius, height);
      } else if(type == 2){
        hit = boxContains(local, width, height, depth);
      } else if(type == 3){
        hit = coneContains(local, radius, width, height);
      }

      if(hit){
        // if this object has equal-or-higher priority, overwrite
        if(priority >= bestPriority){
          bestPriority = priority;
          outColor = col.rgb;
          outAlpha = col.a;
        }
      }
    }

    gl_FragColor = vec4(outColor, outAlpha);
  }
`;

// ----- Main GPU slice drawing function -----
export function init_slicePanel(context){
    // DOM wiring (like your original)
    const toggleBtn = document.getElementById('toggleSlicePanel');
    const slicePanel = document.getElementById('slicePanel');
    const planeHeightSlider = document.getElementById('planeHeight');
    const planeCanvas = document.getElementById('planeCanvas');
    const planeWidthSlider = document.getElementById('planeWidth');
    const planeHeightSizeSlider = document.getElementById('planeHeightSize');

  let xyPlane = null;
  let planeWidth = 2;
  let planeHeight = 2;

  // Create a small three.js scene that renders only our fullscreen quad into planeCanvas
  const sliceRenderer = new THREE.WebGLRenderer({ canvas: planeCanvas, antialias: false, alpha: true });
  sliceRenderer.setPixelRatio(window.devicePixelRatio || 1);

  // Basic scene for full-screen pass
  const sliceScene = new THREE.Scene();
  const orthoCam = new THREE.OrthographicCamera(-1,1,1,-1,0,1);

  // Shader material - uniforms will be filled after we build data texture
  const shaderUniforms = {
    u_dataTex: { value: null },
    u_texSize: { value: new THREE.Vector2(TEXELS_PER_OBJECT, Math.max(1, (context.objects || []).length)) },
    u_planeWidth: { value: planeWidth },
    u_planeHeight: { value: planeHeight },
    u_planeY: { value: parseFloat(planeHeightSlider.value || 0) },
    u_objectCount: { value: Math.min(MAX_OBJECTS, (context.objects || []).length) }
  };

  const material = new THREE.RawShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms: shaderUniforms,
    depthWrite: false,
    depthTest: false
  });

  // fullscreen quad
  const quadGeo = new THREE.PlaneGeometry(2,2);
  const quad = new THREE.Mesh(quadGeo, material);
  sliceScene.add(quad);

  // build and upload data texture once (and when objects change)
  let dataTexture = null;
  function uploadObjects(){
    if(!context.objects || context.objects.length === 0){
      shaderUniforms.u_dataTex.value = null;
      shaderUniforms.u_objectCount.value = 0;
      shaderUniforms.u_texSize.value.set(TEXELS_PER_OBJECT, 1);
      return;
    }
    const tex = buildObjectDataTexture(context.objects);
    dataTexture = tex;
    shaderUniforms.u_dataTex.value = tex;
    shaderUniforms.u_texSize.value.set(TEXELS_PER_OBJECT, context.objects.length);
    shaderUniforms.u_objectCount.value = Math.min(MAX_OBJECTS, context.objects.length);
  }

  // resize helper
  function resizeCanvasToDisplaySize() {
    const w = planeCanvas.clientWidth;
    const h = planeCanvas.clientHeight;
    sliceRenderer.setSize(w, h, false);
    // If you want pixel-perfect mapping between plane world units and canvas pixels:
    // we compute scale factors using renderer.domElement width/height in physical pixels:
    shaderUniforms.u_planeWidth.value = planeWidth;
    shaderUniforms.u_planeHeight.value = planeHeight;
  }

    function renderSlice() {
    resizeCanvasToDisplaySize();
    shaderUniforms.u_planeY.value = parseFloat(planeHeightSlider.value || 0);

    if (shaderUniforms.u_dataTex.value == null) {
        const gl = sliceRenderer.getContext();
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        return;
    }

    sliceRenderer.render(sliceScene, orthoCam);
    }


  // initial upload
  uploadObjects();

  // wire up toggle button and slider
  toggleBtn.addEventListener('click', () => {
    slicePanel.style.display = (slicePanel.style.display === 'none') ? 'block' : 'none';

    if (!xyPlane) {
      const geometry = new THREE.PlaneGeometry(planeWidth, planeHeight);
      const mat = new THREE.MeshStandardMaterial({
        color: 0xaaaaaa,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.5
      });
      xyPlane = new THREE.Mesh(geometry, mat);
      xyPlane.rotation.x = -Math.PI / 2;
      xyPlane.position.set(0, 0, 0);
    } else {
      xyPlane.visible = !xyPlane.visible;
    }

    if (typeof context.scene !== 'undefined') {
      context.scene.add(xyPlane);
      console.log('XY plane added to scene');
    } else {
      console.warn('scene is not defined');
    }
    let frameCount = 0;
    const UPDATE_INTERVAL = 5; // update every 5th frame (tweak as needed)

        // ---- animation loop ----
    function animate() {
        requestAnimationFrame(animate);

        if (frameCount % UPDATE_INTERVAL === 0) {
            uploadObjects();
        }

        // Render the slice
        renderSlice();
        }

    // Start loop
    animate();
  });

  // slider -> move plane Y + redraw
  planeHeightSlider.addEventListener('input', () => {
    if (xyPlane) {
      xyPlane.position.y = parseFloat(planeHeightSlider.value);
    }
    renderSlice();
  });

  // If objects can change over time, give a simple API to refresh the texture
  // (call context.sliceRefreshObjects() after you change geometry/material/userData)
  context.sliceRefreshObjects = () => {
    uploadObjects();
    renderSlice();
  };

  // initial draw if panel visible
  if(slicePanel.style.display !== 'none'){
    renderSlice();
  }
  // initialize with default values
    updatePlaneSize();

    planeWidthSlider.addEventListener('input', updatePlaneSize);
    planeHeightSizeSlider.addEventListener('input', updatePlaneSize);
    function updatePlaneSize() {
        planeWidth = parseFloat(planeWidthSlider.value);
        planeHeight = parseFloat(planeHeightSizeSlider.value);

        // update uniforms for GPU shader
        shaderUniforms.u_planeWidth.value = planeWidth;
        shaderUniforms.u_planeHeight.value = planeHeight;

        // update XY plane mesh (rebuild geometry so its size matches)
        if (xyPlane) {
            xyPlane.geometry.dispose();
            xyPlane.geometry = new THREE.PlaneGeometry(planeWidth, planeHeight);
        }
    }
}


