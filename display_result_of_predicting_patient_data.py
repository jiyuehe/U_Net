# NOTE:
# Because the main.py needs the docker enviroment for the MinkowskiEngine,
# the display_result.py is separated to avoid the dependency issue.

#%%
import os
from pathlib import Path
script_dir = os.path.dirname(os.path.abspath(__file__)) # get the path of the current script
os.chdir(script_dir) # change the working directory
script_dir = Path(script_dir)

# add the workspace root to Python path
import sys
workspace_root = Path().resolve().parent # Path().resolve() returns an absolute path, the full path
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))
import common

import plotly.graph_objects as go # pip install plotly
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = 'browser'

import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import find_peaks

# load prediction results
result_folder = script_dir / 'result'

name_prefix = '103_1-lagood'

predicted_data = np.load(result_folder / f'predictions_{name_prefix}.npy')

geometry_file_name = script_dir.parent / '0_data' / f'{name_prefix}_processed_map_refined.npz'
data = np.load(geometry_file_name, allow_pickle=True)
geometry_data = {k: data[k] for k in data.files}
nodes = geometry_data['voxel3mm_1mm_spacing']  # shape (n_node, 3)

#%%
# predicted data for rhythm 0
sample_id = 0
rhythm_id = 0
data_predicted = predicted_data[sample_id][rhythm_id,:]

data_min_pred = np.nanmin(data_predicted)
data_max_pred = np.nanmax(data_predicted)
data_threshold_pred = data_min_pred-0.1
converted_color_pred = common.convert_data_to_color.execute(data_predicted, data_min_pred, data_max_pred, data_threshold_pred)

#%%
# process the mix rhythm map
##########
clinical_electrogram = geometry_data['clinical_electrogram_unipolar_refined']

# file_path = script_dir.parent / '0_data' / 'simulation_results_6890_20931.npy'
# simulation_data = np.load(file_path, allow_pickle=True).item()
# clinical_electrogram = simulation_data['electrogram_unipolar'].T

lat_electrode = geometry_data['activation_uni'].astype(float)
lat_electrode[lat_electrode == 0] = np.nan

# # compute local activation time
# for e_id in range(clinical_electrogram.shape[0]):
#     egm = clinical_electrogram[e_id,:]

#     # find peaks in the -dv/dt
#     derivative_uni = -np.diff(egm, prepend=egm[0])

#     signal_abs = np.abs(derivative_uni)
#     med = np.median(signal_abs) # median
#     mad = np.median(np.abs(signal_abs - med)) + 1e-12 # mad: median absolute deviation
#     peak_height_threshold = med + 4.0 * mad
#     peaks_egm_uni, _ = find_peaks(derivative_uni, height=peak_height_threshold, distance=80)

#     if len(peaks_egm_uni) != 0:
#         lat_electrode[e_id] = peaks_egm_uni[0]
#     elif len(peaks_egm_uni) == 0:
#         lat_electrode[e_id] = np.nan

#     debug_plot = 0
#     if debug_plot == 1:
#         plt.figure()
#         plt.plot(derivative_uni, label='-dv/dt')
#         plt.plot(egm, label='egm')
#         plt.axhline(peak_height_threshold, color='green', linestyle='--', label='threshold')
#         plt.scatter(peaks_egm_uni, egm[peaks_egm_uni], color='red', label='peaks')
#         plt.legend()
#         plt.title(f'Electrode {e_id}, LAT: {lat_electrode[e_id]}')

#         plt.show()

#%%
'''# interpolate electrode data to node
electrode_node_id = geometry_data['electrode_node_id']
electrode_node = node[electrode_node_id,:]
valid_mask = ~np.isnan(lat_electrode)

# Find nodes within distance 10 of any electrode_node
from scipy.spatial import cKDTree
distance_threshold = 10.0
tree = cKDTree(electrode_node[valid_mask])
distances, _ = tree.query(node)
nodes_id_within_distance = np.where(distances <= distance_threshold)[0]

from scipy.interpolate import Rbf
rbf_interp = Rbf(electrode_node[valid_mask][:, 0], 
                 electrode_node[valid_mask][:, 1], 
                 electrode_node[valid_mask][:, 2], 
                 lat_electrode[valid_mask], 
                 function='multiquadric',  # more robust
                 smooth=0.1)  # add smoothing to avoid singularity
lat_node = rbf_interp(node[nodes_id_within_distance, 0], node[nodes_id_within_distance, 1], node[nodes_id_within_distance, 2])
lat_node = np.clip(lat_node, np.nanmin(lat_electrode[valid_mask]), np.nanmax(lat_electrode[valid_mask]))

lat_node_full = np.full(node.shape[0], np.nan)
lat_node_full[nodes_id_within_distance] = lat_node
'''
#%%
# prepare electrode data
##########
data_electrode = lat_electrode
electrode_node_id = geometry_data['voxel3mm_id_for_electrode']
# data_electrode = lat_electrode[electrode_node_id]

# filter out NaN values
valid_mask = ~np.isnan(data_electrode)
data_electrode_valid = data_electrode[valid_mask]
electrode_node_id_valid = electrode_node_id[valid_mask]

data_min_electrode = np.nanmin(data_electrode_valid)
data_max_electrode = np.nanmax(data_electrode_valid)
data_threshold_electrode = data_min_electrode-0.01
converted_color_electrode = common.convert_data_to_color.execute(data_electrode_valid, data_min_electrode, data_max_electrode, data_threshold_electrode)

electrode_node = geometry_data['voxel3mm_1mm_spacing'][electrode_node_id,:][valid_mask,:]

'''
# prepare node data
data_node = lat_node_full

data_min_node = np.nanmin(data_node)
data_max_node = np.nanmax(data_node)
data_threshold_node = data_min_node-0.01
converted_color_node = common.convert_data_to_color.execute(data_node, data_min_node, data_max_node, data_threshold_node)
'''
#%%
def create_voxel_mesh(centers, colors=None, color=None, opacity=1.0, name='', voxel_size=1.0):
    """Create a Mesh3d trace of cubes for gap-free voxel visualization."""
    n = len(centers)
    s = voxel_size / 2
    # 8 vertex offsets for a cube
    offsets = np.array([
        [-s, -s, -s], [+s, -s, -s], [+s, +s, -s], [-s, +s, -s],
        [-s, -s, +s], [+s, -s, +s], [+s, +s, +s], [-s, +s, +s],
    ])
    # all vertices: (n*8, 3)
    verts = (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    # 12 triangle faces per cube (2 per face, 6 faces)
    face_i = np.array([0, 0, 4, 4, 3, 3, 0, 0, 0, 0, 1, 1])
    face_j = np.array([2, 3, 5, 6, 6, 7, 1, 5, 4, 7, 2, 6])
    face_k = np.array([1, 2, 6, 7, 2, 6, 5, 4, 7, 3, 6, 5])
    # offset face indices for each cube
    cube_offsets = (np.arange(n) * 8).reshape(-1, 1)  # (n, 1)
    all_i = (face_i[None, :] + cube_offsets).ravel()
    all_j = (face_j[None, :] + cube_offsets).ravel()
    all_k = (face_k[None, :] + cube_offsets).ravel()
    # face colors: repeat each voxel's color 12 times
    if colors is not None:
        # colors is (n, 3) RGB float array → convert to rgb strings, repeat per face
        rgb_strings = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in colors]
        facecolor = np.repeat(rgb_strings, 12).tolist()
    else:
        facecolor = None
    mesh = go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=all_i, j=all_j, k=all_k,
        facecolor=facecolor,
        color=color,
        opacity=opacity,
        flatshading=True,
        lighting=dict(ambient=1, diffuse=0, specular=0, roughness=1, fresnel=0),
        name=name,
    )
    return mesh

#%%
# create combined figure with 2 subplots
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scene'}, {'type': 'scene'}]],
    subplot_titles=('Clinical Map', 'Prediction'),
    horizontal_spacing=0.005
)

# separate non-electrode nodes (gray) from electrode nodes to avoid overlap
non_electrode_mask = np.ones(len(nodes), dtype=bool)
non_electrode_mask[electrode_node_id_valid] = False
non_electrode_nodes = nodes[non_electrode_mask]

# add gray geometry voxels (non-electrode positions only)
fig.add_trace(create_voxel_mesh(
    non_electrode_nodes, color='lightgray', opacity=0.1, name='Geometry'
), row=1, col=1)

# add electrode data voxels
fig.add_trace(create_voxel_mesh(
    electrode_node, colors=converted_color_electrode, name='Electrode'
), row=1, col=1)

# add predicted data voxels
fig.add_trace(create_voxel_mesh(
    nodes, colors=converted_color_pred, name='Predicted'
), row=1, col=2)

# set common camera view for synchronized rotation
camera = dict(
    eye=dict(x=1.5, y=1.5, z=1.5)
)

# update layout with synchronized camera and scene settings
fig.update_layout(
    scene=dict(
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False),
        zaxis=dict(showgrid=False, visible=False),
        camera=camera,
        dragmode='orbit'
    ),
    scene2=dict(
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False),
        zaxis=dict(showgrid=False, visible=False),
        camera=camera,
        dragmode='orbit'
    ),
    height=600,
    width=1200,
    showlegend=False,
    margin=dict(l=0, r=0, b=0, t=30, pad=0)
)

# camera sync JavaScript
sync_js = """
var plot = document.getElementsByClassName('plotly-graph-div')[0];
var isUpdating = false;
function syncCamera(eventdata) {
    if (isUpdating) return;
    var cameraUpdate = {};
    var needsUpdate = false;
    if (eventdata['scene.camera']) {
        cameraUpdate['scene.camera'] = eventdata['scene.camera'];
        cameraUpdate['scene2.camera'] = eventdata['scene.camera'];
        needsUpdate = true;
    } else if (eventdata['scene2.camera']) {
        cameraUpdate['scene.camera'] = eventdata['scene2.camera'];
        cameraUpdate['scene2.camera'] = eventdata['scene2.camera'];
        needsUpdate = true;
    }
    if (needsUpdate) {
        isUpdating = true;
        Plotly.relayout(plot, cameraUpdate);
        setTimeout(function() { isUpdating = false; }, 0);
    }
}
plot.on('plotly_relayouting', syncCamera);
plot.on('plotly_relayout', syncCamera);
"""

fig.show(renderer='browser', post_script=[sync_js])
