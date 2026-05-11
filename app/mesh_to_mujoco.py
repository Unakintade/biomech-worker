import numpy as np
import trimesh
import xml.etree.ElementTree as ET
import os

def update_mujoco_from_trellis_and_smpl(mesh_path, xml_path, smpl_betas, height_cm):
    """
    The bridge script: Uses Trellis geometry and SMPL shape (from Colab) 
    to scale the MuJoCo physics model on Render/Mac.
    """
    # 1. Load and Scale the AI-generated mesh
    # If the mesh wasn't saved locally, this would need to handle bytes from the API
    if not os.path.exists(mesh_path):
        print(f"Warning: Mesh file {mesh_path} not found. Using height-only scaling.")
        mesh_height = 1.0 # Placeholder
    else:
        mesh = trimesh.load(mesh_path)
        mesh_height = mesh.bounds[1][1] - mesh.bounds[0][1]
    
    scale_factor = (height_cm / 100.0) / mesh_height
    
    # 2. Use SMPL Betas (Shape Parameters)
    # beta[1] typically controls 'fullness' or thickness
    thickness_scale = 1.0 + (smpl_betas[1] * 0.1) 

    # Estimated limb lengths based on scaled mesh height/proportions
    # These are then used to update the MuJoCo 'fromto' attributes
    thigh_len = (height_cm / 100.0) * 0.25 
    shank_len = (height_cm / 100.0) * 0.24
    
    specs = {
        "r_thigh_geom": (thigh_len, thickness_scale),
        "r_shank_geom": (shank_len, thickness_scale),
        "r_foot_geom": (0.18 * (height_cm/175.0), thickness_scale)
    }

    # 3. Inject updates into the MuJoCo XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for geom in root.iter('geom'):
        name = geom.get('name')
        if name in specs:
            length, thick_mult = specs[name]
            
            # Update Bone Length
            if 'fromto' in geom.attrib:
                ft = [float(x) for x in geom.get('fromto').split()]
                ft[5] = ft[2] - length 
                geom.set('fromto', f"{ft[0]} {ft[1]} {ft[2]} {ft[3]} {ft[4]} {ft[5]}")
            
            # Update Muscle/Segment Thickness
            if 'size' in geom.attrib:
                current_size = float(geom.get('size'))
                geom.set('size', str(round(current_size * thick_mult, 4)))
                
    # Save the updated model for the solver to use
    tree.write(xml_path)
    print(f"MuJoCo Model updated successfully at {xml_path}")
    return specs
