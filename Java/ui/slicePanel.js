import * as THREE from 'three';



export function init_slicePanel(context){// Add slice panel 
    // Assuming you already have these references
    const toggleBtn = document.getElementById('toggleSlicePanel');
    const slicePanel = document.getElementById('slicePanel');
    const planeHeightSlider = document.getElementById('planeHeight');
    // Track if the plane has been added
    let xyPlane = null;
    let planeWidth = 2;
    let planeHeight = 2;


    toggleBtn.addEventListener('click', () => {
    // Toggle panel visibility
    slicePanel.style.display = (slicePanel.style.display === 'none') ? 'block' : 'none';

    // Add XY plane if it doesn't exist
    if (!xyPlane) {
        // Create a plane
     
        const geometry = new THREE.PlaneGeometry(planeWidth, planeHeight);
        const material = new THREE.MeshStandardMaterial({
        color: 0xaaaaaa,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.5
        });
        xyPlane = new THREE.Mesh(geometry, material);

        xyPlane.rotation.x = -Math.PI / 2; // Make it XY plane
        xyPlane.position.set(0, 0, 0);
    } else {
        xyPlane.visible = !xyPlane.visible; // toggle visibility
    }
        // Make sure 'scene' is in scope
    if (typeof context.scene !== 'undefined') {
        context.scene.add(xyPlane);
        console.log('XY plane added to scene');
    } else {
        console.warn('scene is not defined');
    }
    // make the first plot
    drawPrioritySlice(context.objects, planeHeightSlider.value, planeWidth, planeHeight);
    });
    // Move plane up/down when slider changes
    planeHeightSlider.addEventListener('input', () => {
        if (xyPlane) {
            xyPlane.position.y = parseFloat(planeHeightSlider.value);
        }
    });
    // Update when slider moves
    planeHeightSlider.addEventListener('input', () => {
        drawPrioritySlice(context.objects, planeHeightSlider.value, planeWidth, planeHeight);
    });
}


function drawPrioritySlice(objects, planeY, planeWidth, planeHeight) {
    const canvas = document.getElementById('planeCanvas');
    
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
     // Compute scale factors (pixels per world unit)
    const scaleX = canvas.width / planeWidth;
    const scaleY = canvas.height / planeHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let topPriority = -Infinity;
    let color = [0,0,0];
    const imageData = ctx.createImageData(canvas.width, canvas.height);
    const data = imageData.data;

    objects.forEach(obj => {
        // Skip objects that do not intersect planeY (local-space check)
        const localPlane = new THREE.Vector3(0, planeY, 0);
        obj.worldToLocal(localPlane);
        // If object has a height, check bounding box
        if (obj.userData.heigh){
            const halfHeight = (obj.userData.height || 0) / 2;
            if (localPlane.y < -halfHeight || localPlane.y > halfHeight) return;
        } else {
            if (Math.abs(localPlane.y > obj.userData.radius)) return;
        }

        const color = [
            Math.floor(obj.material.color.r * 255),
            Math.floor(obj.material.color.g * 255),
            Math.floor(obj.material.color.b * 255)
        ];
        const priority = obj.userData.priority || 0;
        const vec = new THREE.Vector3(0, planeY, 0);
        // Loop over pixels that the object projects onto
        for (let i = 0; i <= canvas.width; i++) {
            for (let j = 0; j <= canvas.height; j++) {
                // Map pixel to world coordinates
                
                vec.set((i / scaleX - planeWidth/2), planeY, (planeHeight / 2) - (j / scaleY));

                if (!shapeContainsPoint(obj, vec)) continue;

                const idx = (j * canvas.width + i) * 4;
                const p = obj.userData.priority;
                // Only overwrite if this object has higher priority
                if (p >= topPriority) {
                    data[idx] = color[0];
                    data[idx + 1] = color[1];
                    data[idx + 2] = color[2];
                    data[idx + 3] = 255; // fully opaque
                }
            }
        }
    });
    ctx.putImageData(imageData, 0, 0);
}
function shapeContainsPoint(obj, vec){
    // ensure world matrices are up-to-date
    obj.updateMatrixWorld(true);

    // copy the world point and transform it into the object's local space
    const local = vec.clone();
    obj.worldToLocal(local); 
    switch (obj.userData.type) {
        case 'Union_cylinder':
            return cylinderContainsPoint(obj, local);
        case 'Union_sphere':
            return sphereContainsPoint(obj, local);
        case 'Union_box':
            return boxContainsPoint(obj, local);
        case 'Union_cone':
            return coneContainsPoint(obj, local);
        default:
        return false;

    }
}

function cylinderContainsPoint(obj, local){
    // cylinder aligned with local Y
    const radius = obj.userData.radius;
    const height = obj.userData.height;

    // radial distance in XZ plane and local Y as height
    const r = Math.hypot(local.x, local.z); // equivalent to sqrt(x*x + z*z)
    const h = local.y;

    // If cylinder origin is centered, check [-height/2, +height/2]
    return (r >= 0) &&(r <= radius) && (h >= -height/2) && (h <= height/2);
}

function sphereContainsPoint(obj, local) {
    const radius = obj.userData.radius;

    const r = Math.hypot(local.x, local.y, local.z);
    return r <= radius;
}

function boxContainsPoint(obj, local) {
    const w = obj.userData.width / 2;
    const h = obj.userData.height / 2;
    const d = obj.userData.depth / 2;

    return (
    local.x >= -w && local.x <= w &&
    local.y >= -h && local.y <= h &&
    local.z >= -d && local.z <= d
    );
}

function coneContainsPoint(obj, local) {
  const height = obj.userData.height;
  const bottomRadius = obj.userData.radius_bottom;
  const topRadius = obj.userData.radius_top;
    
  const h = local.y;
  if (h < -height/2 || h > height/2) return false;
 
  // Linearly interpolate radius at this height
  const t = h / height; // 0 at base, 1 at top
  const rAtH = bottomRadius * (1 - t) + topRadius * t;

  const r = Math.hypot(local.x, local.z);
    // console.log(bottomRadius, topRadius, height)
  return r <= rAtH;
}