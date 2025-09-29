/*
 * 
 * Function for exporting instruments loaded into the context.objects
 * into a copy friendly text in a new window
 * */


export function writeInstr(context){
  // Build the content string (same as before)
  let fileContent = '';

 
   const sortedObjects = [...context.objects].sort((a, b) => a.userData.priority - b.userData.priority);
   sortedObjects.forEach(obj => {
     const parameters = buildParameters(obj);
     const position = `(${obj.position.x.toFixed(2)}, ${obj.position.y.toFixed(2)}, ${obj.position.z.toFixed(2)})`;
     const rotation = `(${
       (obj.rotation.x * 180 / Math.PI).toFixed(2)}, ${
       (obj.rotation.y * 180 / Math.PI).toFixed(2)}, ${
       (obj.rotation.z * 180 / Math.PI).toFixed(2)})`;
     fileContent += `COMPONENT ${obj.name} = ${obj.userData.type}(\n${parameters}\n) AT ${position} RELATIVE sample_arm\nROTATED ${rotation} RELATIVE sample_arm\n\n`;
   });


  // Open a new tab
  const newWindow = window.open('', '_blank');
  // TODO: Make this also write out in mcstasscript
  // Write the content as HTML using <pre> for formatting
  newWindow.document.write(`
    <html>
      <head>
        <title>Exported INSTR</title>
        <style>
          body { font-family: monospace; padding: 20px; white-space: pre; background: #f5f5f5; }
          pre { background: #fff; padding: 10px; border-radius: 6px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        </style>
      </head>
      <body>
        <h2>Generated McStas Union Content</h2>
        <pre>${fileContent}</pre>
      </body>
    </html>
  `);

  newWindow.document.close();
}

// Helper function to build the parameters string based on geometry type
function buildParameters(obj) {
    let params = '';
    // Get the material and priority from userData, defaulting to "default" for material and 0 for priority
    const material = obj.userData.materialName || 'default';
    const priority = obj.userData.priority || 0;

    switch (obj.userData.type) {
        case 'Union_cylinder':
            params += `radius=${obj.userData.radius ?? 
                                 obj.userData.radiusTop ?? 
                                 obj.geometry.parameters.radiusTop ?? 
                                 obj.userData.radiusBottom ?? 
                                 obj.geometry.parameters.radiusBottom ?? 1},
   yheight=${obj.userData.height ?? obj.geometry.parameters.height ?? 1},
   material_string=${material}, priority=${priority}`;
            break;
    
        case 'Union_box':
            params += `xwidth=${obj.userData.width ?? obj.geometry.parameters.width ?? 1},
   yheight=${obj.userData.height ?? obj.geometry.parameters.height ?? 1},
   zdepth=${obj.userData.depth ?? obj.geometry.parameters.depth ?? 1},
   material_string=${material}, 
   priority=${priority}`;
            break;
    
        case 'Union_sphere':
            params += `radius=${obj.userData.radius ?? obj.geometry.parameters.radius ?? 1},
   material_string=${material},
   priority=${priority}`;
            break;
    
        case 'Union_cone':
            params += `radius_top=${obj.userData.radiusTop ?? obj.geometry.parameters.radiusTop ?? 1},
   radius_bottom=${obj.userData.radiusBottom ?? obj.geometry.parameters.radiusBottom ?? 1},
   yheight=${obj.userData.height ?? obj.geometry.parameters.height ?? 1},
   material_string=${material},
   priority=${priority}`;
            break;
    
        default:
            console.warn(`Unsupported geometry type: ${obj.type}`);
            break;
    }
    return params;
}
