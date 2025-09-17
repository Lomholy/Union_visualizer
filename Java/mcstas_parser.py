# server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
import mcstasscript as ms
from typing import cast
from libpyvinyl.Parameters.Parameter import Parameter
from mcstasscript.interface.instr import McCode_instr
from mcstasscript.helper.mcstas_objects import Component, DeclareVariable
from mcstasscript.instr_reader.control import InstrumentReader  # <-- this is your existing parser code
import json




def parse_mcstas(file):
    instr = InstrumentReader(file)
    instr.generate_py_version('temp.py')
    # Change instrument name
    instrument = cast(McCode_instr,instr.Instr)
    instrument.name = file
    return instrument_to_json(instrument)


def instrument_to_json(instr: McCode_instr):
    data = {
        "name": instr.name,
        "parameters": {p.name: p.value for p in instr.parameters},
        "components": []
    }
    
    for comp in instr.component_list:
        # Only use the component if they are of Union_geometry type
        geometries = ['Union_cone', 'Union_cylinder', 'Union_box', 'Union_sphere']
        if comp.component_name not in geometries:
            continue

        component_parameters = {}
        for key in comp.parameter_names:
            val = getattr(comp, key)
            if val is None:
                if comp.parameter_defaults[key] is None:
                    raise NameError("Required parameter named "
                                    + key
                                    + " in component named "
                                    + comp.name
                                    + " not set.")
                else:
                    continue
            elif isinstance(val, (Parameter, DeclareVariable)):
                # Extract the parameter name
                val = val.name

            component_parameters[key] = val
        comp = cast(Component,comp)


        comp_info = {
            "name": comp.name,
            "type": comp.component_name,
            "parameters": component_parameters
        }
        data["components"].append(comp_info)

    return json.dumps(data, indent=2)

# file = '../../template.instr'
# parse_mcstas(file)

app = FastAPI()


# Allow requests from Vite frontend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173"] for stricter control
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define request schema
class ParseRequest(BaseModel):
    path: str
@app.post("/parse")
async def parse_file(request: ParseRequest):
    result = parse_mcstas(request.path)
    return {"parsed": result}
