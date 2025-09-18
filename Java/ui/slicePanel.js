import * as THREE from 'three';
import { atan, cos } from 'three/tsl';



export function init_slicePanel(context){// Add slice panel 
    // Assuming you already have these references
    const toggleBtn = document.getElementById('toggleSlicePanel');
    const slicePanel = document.getElementById('slicePanel');
    const planeHeightSlider = document.getElementById('planeHeight');
    // Track if the plane has been added
    let xyPlane = null;

    toggleBtn.addEventListener('click', () => {
    // Toggle panel visibility
    slicePanel.style.display = (slicePanel.style.display === 'none') ? 'block' : 'none';

    // Add XY plane if it doesn't exist
    if (!xyPlane) {
        // Create a plane
        const size = 2; // adjust as needed
        const geometry = new THREE.PlaneGeometry(size, size);
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
    });
    // Move plane up/down when slider changes
    planeHeightSlider.addEventListener('input', () => {
        if (xyPlane) {
            xyPlane.position.y = parseFloat(planeHeightSlider.value);
        }
    });
    // Update when slider moves
    planeHeightSlider.addEventListener('input', () => {
        drawPrioritySlice(context.objects, planeHeightSlider.value);
    });
}


function drawPrioritySlice(objects, planeY) {
    const canvas = document.getElementById('planeCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const width = canvas.width;
    const height = canvas.height;

    const scale = 1000;         // world units → pixels
    const offsetX = width / 2;
    const offsetZ = height / 2;
    console.log('New planeHeight Check')
    for (let i = 0; i < width; i++) {
        for (let j = 0; j < height; j++) {
            // Map canvas pixel to world coordinates (X,Z)
            const x = (i - offsetX) / scale;
            const z = (offsetZ - j) / scale;
            const vec = new THREE.Vector3(x, planeY, z)

            let topShape = null;
            let topPriority = -Infinity;

            objects.forEach(obj => {
            if (obj.userData.type==='Union_cylinder'){
                if (cylinderContainsPoint(obj, vec)){
                    
                    const p = obj.userData.priority || 0;
                    console.log(p)
                    if (p >= topPriority) {
                        topPriority = p;
                        topShape = obj;
                    }
                }
            }});

            if (topShape) {
                ctx.fillStyle = topShape.material.color.getStyle();
                ctx.fillRect(i, j, 1, 1);
            }
        }
    }
}
function shapeContainsPoint(obj, vec){
    return false
}


function cylinderContainsPoint(obj, vec){
    // console.log(obj)
    const max_rad = obj.userData.radius;
    const height = obj.userData.height;
    // Get the rotation matrix
    const matrix4 = new THREE.Matrix4().makeRotationFromEuler(obj.rotation);

    // Extract 3x3 rotation matrix
    const ROT = new THREE.Matrix3().setFromMatrix4(matrix4);
    // Make a copy of ROT and invert it
    const ROT_inv = new THREE.Matrix3().copy(ROT).invert();

    // Apply inverse rotation to X
    const localVec = vec.clone().applyMatrix3(ROT_inv);

    const dx = localVec.x - obj.position.x;
    const dz = localVec.z - obj.position.z;
    const r = Math.sqrt(dx*dx + dz*dz);
    const h = localVec.y - obj.position.y;
    
    
    // Return true if r and h are within max rad & height
    if (r <= max_rad && r>=0 && h >= -height/2 && h <= height/2){
        console.log(r,h)
        return true
    }

    return false
}



