import * as THREE from 'three';
import { atan, cos } from 'three/tsl';



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

    // Match canvas size to container
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;

    const scaleX = canvas.width / planeWidth;
    const scaleY = canvas.height / planeHeight;
    const centerX = canvas.width / 2;
    const centerZ = canvas.height / 2;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Sort objects by priority (low -> high)
    objects.sort((a,b) => (a.userData.priority || 0) - (b.userData.priority || 0));

    objects.forEach(obj => {
        const color = obj.material.color.getStyle();
        ctx.fillStyle = color;

        switch(obj.userData.type) {
            case 'Union_cylinder':
                const halfHeight = obj.userData.height / 2;
                if (planeY < obj.position.y - halfHeight || planeY > obj.position.y + halfHeight) break;

                const r = obj.userData.radius;
                const x = centerX + obj.position.x * scaleX;
                const z = centerZ - obj.position.z * scaleY;

                ctx.beginPath();
                ctx.arc(x, z, r * ((scaleX + scaleY)/2), 0, 2*Math.PI);
                ctx.fill();
                break;

            case 'Union_sphere':
                const sphereR = obj.userData.radius;
                const dy = planeY - obj.position.y;
                if (Math.abs(dy) > sphereR) break;

                const sliceR = Math.sqrt(sphereR*sphereR - dy*dy); // radius at slice
                const sx = centerX + obj.position.x * scaleX;
                const sz = centerZ - obj.position.z * scaleY;

                ctx.beginPath();
                ctx.arc(sx, sz, sliceR * ((scaleX + scaleY)/2), 0, 2*Math.PI);
                ctx.fill();
                break;

            case 'Union_box':
                const w = obj.userData.width / 2;
                const d = obj.userData.depth / 2;
                const h = obj.userData.height / 2;
                if (planeY < obj.position.y - h || planeY > obj.position.y + h) break;

                const bx = centerX + (obj.position.x - w) * scaleX;
                const bz = centerZ - (obj.position.z + d) * scaleY;
                ctx.fillRect(bx, bz, w*2*scaleX, d*2*scaleY);
                break;

            case 'Union_cone':
                const coneH = obj.userData.height;
                const bottomR = obj.userData.radius_bottom;
                const topR = obj.userData.radius_top;
                const halfH = coneH / 2;
                if (planeY < obj.position.y - halfH || planeY > obj.position.y + halfH) break;

                // Linear interpolation of radius at planeY
                const t = (planeY - (obj.position.y - halfH)) / coneH;
                const coneR = bottomR * (1 - t) + topR * t;

                const cx = centerX + obj.position.x * scaleX;
                const cz = centerZ - obj.position.z * scaleY;
                ctx.beginPath();
                ctx.arc(cx, cz, coneR * ((scaleX + scaleY)/2), 0, 2*Math.PI);
                ctx.fill();
                break;
        }
    });
}
