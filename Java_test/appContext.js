// appContext.js
const context = {
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    objects: [],
    sliderListeners: {},
    inputListeners: {},
    selectedObject: null,
    raycaster: null,
    mouse: null
  };
  
  export default context;