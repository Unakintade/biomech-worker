import numpy as np
import trimesh
import xml.etree.ElementTree as ET
import os

def update_mujoco_from_trellis_and_smpl(mesh_path, xml_path, smpl_betas, height_cm):
    """
    Refined updater using Trellis mesh geometry and mmhuman3d SMPL shape parameters.
    smpl_betas: list of 10 shape parameters from mmhuman3d/SMPL
    """
    # 1. LOAD TRELLIS MESH
    mesh = trimesh.load(mesh_path)
    mesh_height = mesh.bounds[1][1] - mesh.bounds[0][1]
    scale_factor = (height_cm / 100.0) / mesh_height
    mesh.apply_scale(scale_factor)
    
    # 2. EXTRACT BIO-ACCURATE DIMENSIONS
    # Using SMPL beta[0] (stature/scale) and beta[1] (weight/thickness)
    # to refine the MuJoCo geom sizes (thickness of capsules)
    thickness_scale = 1.0 + (smpl_betas[1] * 0.1) 

    bounds = mesh.bounds
    thigh_len = abs(bounds[1][1] * 0.5 - bounds[1][1] * 0.25)
    shank_len = abs(bounds[1][1] * 0.25 - bounds[0][1])
    
    # Map of geom name to (length, thickness_multiplier)
    specs = {
        "r_thigh_geom": (thigh_len, thickness_scale * 1.1), # Sprinters have > avg quads
        "r_shank_geom": (shank_len, thickness_scale),
        "r_foot_geom": (height_cm * 0.0015, thickness_scale)
    }

    # 3. INJECT INTO MUJOCO XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for geom in root.iter('geom'):
        name = geom.get('name')
        if name in specs:
            length, thick_mult = specs[name]
            
            # Update Length via 'fromto'
            if 'fromto' in geom.attrib:
                ft = [float(x) for x in geom.get('fromto').split()]
                ft[5] = ft[2] - length 
                geom.set('fromto', f"{ft[0]} {ft[1]} {ft[2]} {ft[3]} {ft[4]} {ft[5]}")
            
            # Update Thickness via 'size'
            if 'size' in geom.attrib:
                current_size = float(geom.get('size'))
                geom.set('size', str(round(current_size * thick_mult, 4)))
                
            print(f"Refined {name}: Length={round(length,3)}m, Thickness Multiplier={round(thick_mult,2)}")

    tree.write(xml_path)
    return specs

if __name__ == "__main__":
    # Example: SMPL Betas usually come from mmhuman3d inference
    # mock_betas = [0.5, 1.2, -0.3, 0.1, 0, 0, 0, 0, 0, 0]
    # update_mujoco_from_trellis_and_smpl("run_mesh.obj", "app/models/sprinter.xml", mock_betas, 182)
    pass
