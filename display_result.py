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

# name_prefix = '6-1-1-1-LA PACING CL 300 FROM CS 13 14'
# name_prefix = '6-1-1-LA PACING CL 270 FROM CS3 4'
name_prefix = '6-1-LA PACING CS 11 12 300CL'

predicted_data = np.load(result_folder / f'predictions_{name_prefix}.npy')

geometry_file_name = script_dir.parent / '0_data' / f'{name_prefix}_processed.npz'
data = np.load(geometry_file_name, allow_pickle=True)
geometry_data = {k: data[k] for k in data.files}
nodes = geometry_data['voxel']  # shape (n_node, 3)

#%%
# predicted data for rhythm 0
sample_id = 0
rhythm_id = 0
data_predicted = predicted_data[sample_id][rhythm_id,:]

data_min_pred = np.nanmin(data_predicted)
data_max_pred = np.nanmax(data_predicted)
data_threshold_pred = data_min_pred-0.1
converted_color_pred = common.convert_data_to_color.execute(data_predicted, data_min_pred, data_max_pred, data_threshold_pred)

# Prepare predicted data for rhythm 1
rhythm_id_1 = 1
data_predicted_1 = predicted_data[sample_id][rhythm_id_1,:]

data_min_pred_1 = np.nanmin(data_predicted_1)
data_max_pred_1 = np.nanmax(data_predicted_1)
data_threshold_pred_1 = data_min_pred_1-0.1
converted_color_pred_1 = common.convert_data_to_color.execute(data_predicted_1, data_min_pred_1, data_max_pred_1, data_threshold_pred_1)

#%%
# process the mix rhythm map
##########
clinical_electrogram = geometry_data['clinical_electrogram_unipolar_woi']

# file_path = script_dir.parent / '0_data' / 'simulation_results_6890_20931.npy'
# simulation_data = np.load(file_path, allow_pickle=True).item()
# clinical_electrogram = simulation_data['electrogram_unipolar'].T

# compute local activation time
lat_electrode = np.zeros(clinical_electrogram.shape[0])
for e_id in range(clinical_electrogram.shape[0]):
    egm = clinical_electrogram[e_id,:]

    # find peaks in the -dv/dt
    derivative_uni = -np.diff(egm, prepend=egm[0])

    signal_abs = np.abs(derivative_uni)
    med = np.median(signal_abs) # median
    mad = np.median(np.abs(signal_abs - med)) + 1e-12 # mad: median absolute deviation
    peak_height_threshold = med + 4.0 * mad
    peaks_egm_uni, _ = find_peaks(derivative_uni, height=peak_height_threshold, distance=80)

    if len(peaks_egm_uni) != 0:
        lat_electrode[e_id] = peaks_egm_uni[0]
    elif len(peaks_egm_uni) == 0:
        lat_electrode[e_id] = np.nan

    debug_plot = 0
    if debug_plot == 1:
        plt.figure()
        plt.plot(derivative_uni, label='-dv/dt')
        plt.plot(egm, label='egm')
        plt.axhline(peak_height_threshold, color='green', linestyle='--', label='threshold')
        plt.scatter(peaks_egm_uni, egm[peaks_egm_uni], color='red', label='peaks')
        plt.legend()
        plt.title(f'Electrode {e_id}, LAT: {lat_electrode[e_id]}')

        plt.show()

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
electrode_node_id = geometry_data['electrode_node_id']
# data_electrode = lat_electrode[electrode_node_id]

# Filter out NaN values
valid_mask = ~np.isnan(data_electrode)
data_electrode_valid = data_electrode[valid_mask]
electrode_node_id_valid = electrode_node_id[valid_mask]

data_min_electrode = np.nanmin(data_electrode_valid)
data_max_electrode = np.nanmax(data_electrode_valid)
data_threshold_electrode = data_min_electrode-0.01
converted_color_electrode = common.convert_data_to_color.execute(data_electrode_valid, data_min_electrode, data_max_electrode, data_threshold_electrode)


electrode_node = geometry_data['electrode_positions'][valid_mask,:] # node[electrode_node_id_valid,:]

'''
# prepare node data
data_node = lat_node_full

data_min_node = np.nanmin(data_node)
data_max_node = np.nanmax(data_node)
data_threshold_node = data_min_node-0.01
converted_color_node = common.convert_data_to_color.execute(data_node, data_min_node, data_max_node, data_threshold_node)
'''
#%%
# Create combined figure with 3 subplots
fig = make_subplots(
    rows=1, cols=3,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
    subplot_titles=('Clinical Map', 'Predicted Rhythm 0', 'Predicted Rhythm 1'),
    horizontal_spacing=0.005
)

# Add geometry nodes
scatter_nodes = go.Scatter3d(
    x=nodes[:, 0],
    y=nodes[:, 1],
    z=nodes[:, 2],
    mode='markers',
    marker=dict(
        size=1,
        color='lightgray',
        opacity=0.5,
        symbol='square'
    ),
    name='Electrode'
)
fig.add_trace(scatter_nodes, row=1, col=1)

# Add electrode data scatter
scatter_electrode = go.Scatter3d(
    x=electrode_node[:, 0],
    y=electrode_node[:, 1],
    z=electrode_node[:, 2],
    mode='markers',
    marker=dict(
        size=3,
        color=converted_color_electrode,
        opacity=1,
        symbol='square'
    ),
    name='Electrode'
)
fig.add_trace(scatter_electrode, row=1, col=1)

# Add predicted data scatter
scatter_pred = go.Scatter3d(
    x=nodes[:, 0],
    y=nodes[:, 1],
    z=nodes[:, 2],
    mode='markers',
    marker=dict(
        size=3,
        color=converted_color_pred,
        opacity=1,
        symbol='square'
    ),
    name='Predicted'
)
fig.add_trace(scatter_pred, row=1, col=2)

# Add predicted data for rhythm_id=1
scatter_pred_1 = go.Scatter3d(
    x=nodes[:, 0],
    y=nodes[:, 1],
    z=nodes[:, 2],
    mode='markers',
    marker=dict(
        size=3,
        color=converted_color_pred_1,
        opacity=1,
        symbol='square'
    ),
    name='Predicted 1'
)
fig.add_trace(scatter_pred_1, row=1, col=3)

# Set common camera view for synchronized rotation
camera = dict(
    eye=dict(x=1.5, y=1.5, z=1.5)
)

# Update layout with synchronized camera and scene settings
fig.update_layout(
    scene=dict(
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False),
        zaxis=dict(showgrid=False, visible=False),
        camera=camera
    ),
    scene2=dict(
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False),
        zaxis=dict(showgrid=False, visible=False),
        camera=camera
    ),
    scene3=dict(
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False),
        zaxis=dict(showgrid=False, visible=False),
        camera=camera
    ),
    height=600,
    width=1800,
    showlegend=False,
    margin=dict(l=0, r=0, b=0, t=30, pad=0)
)

# Save to HTML with custom JavaScript for camera synchronization
html_string = fig.to_html(include_plotlyjs='cdn')

# Add JavaScript to synchronize camera movements across all three scenes
sync_js = """
<script>
var plot = document.getElementsByClassName('plotly-graph-div')[0];
var isUpdating = false;

function syncCamera(eventdata) {
    if (isUpdating) return;
    
    var cameraUpdate = {};
    var needsUpdate = false;
    
    // Check which scene was updated
    if (eventdata['scene.camera']) {
        cameraUpdate['scene.camera'] = eventdata['scene.camera'];
        cameraUpdate['scene2.camera'] = eventdata['scene.camera'];
        cameraUpdate['scene3.camera'] = eventdata['scene.camera'];
        needsUpdate = true;
    } else if (eventdata['scene2.camera']) {
        cameraUpdate['scene.camera'] = eventdata['scene2.camera'];
        cameraUpdate['scene2.camera'] = eventdata['scene2.camera'];
        cameraUpdate['scene3.camera'] = eventdata['scene2.camera'];
        needsUpdate = true;
    } else if (eventdata['scene3.camera']) {
        cameraUpdate['scene.camera'] = eventdata['scene3.camera'];
        cameraUpdate['scene2.camera'] = eventdata['scene3.camera'];
        cameraUpdate['scene3.camera'] = eventdata['scene3.camera'];
        needsUpdate = true;
    }
    
    if (needsUpdate) {
        isUpdating = true;
        Plotly.relayout(plot, cameraUpdate);
        setTimeout(function() {
            isUpdating = false;
        }, 0);
    }
}

// Listen to continuous updates while dragging
plot.on('plotly_relayouting', syncCamera);
// Also listen to final update when mouse is released
plot.on('plotly_relayout', syncCamera);
</script>
"""

# Insert the JavaScript before closing body tag
html_string = html_string.replace('</body>', sync_js + '</body>')

# Write to file and open in browser
import tempfile
import webbrowser
with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
    f.write(html_string)
    temp_path = f.name

webbrowser.open('file://' + temp_path)
print(f"Plot opened in browser with synchronized camera views: {temp_path}")
