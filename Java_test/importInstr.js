
function parseInstrFile(text) {
    const componentRegex = /COMPONENT\s+(\w+)\s*=\s*(\w+)\(([\s\S]*?)\)\s*AT\s*\(([^)]+)\)/g;
  
    let match;
    while ((match = componentRegex.exec(text)) !== null) {
      const [_, name, type, paramsBlock, positionStr] = match;
  
      const params = parseParams(paramsBlock);
      const position = positionStr.split(',').map(s => parseFloat(s.trim()));
      if (!type.startsWith("Union_")) {continue;}
      spawnFromInstr({
        name,
        type,
        params,
        position
      });
    }
  }