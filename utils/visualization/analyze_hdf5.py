import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_trajectories(h5_file_path, snippet_ids=[1500, 2500]):
    """
    Plots trajectories from an HDF5 file and saves the plot as an image.
    
    Parameters:
    - h5_file_path: str, path to the HDF5 file containing trajectory data.
    - output_image_path: str, path where the output image will be saved.
    """
    with h5py.File(h5_file_path, 'r') as f:
        seq_ids = f['joint_data/seq_id']
        positions = f['joint_data/positions']
        action_seq_ids = f['actions/start_seq_id']
        action_positions = f['actions/joint_position']
        action_predictions = f['actions/predicted_actions']

        non_zero_indices = [i for i, x in enumerate(action_seq_ids) if x != 0]
        # print(f'length of action_seq_ids: {len(action_seq_ids)}')
        # print(f'length of non-zero indices: {len(non_zero_indices)}')
        action_seq_ids = action_seq_ids[non_zero_indices]
        action_positions = action_positions[non_zero_indices]
        action_predictions = action_predictions[non_zero_indices]

        if snippet_ids is not None:
            startup_sequence_id = snippet_ids[0]
            cutoff_sequence_id = snippet_ids[1]

            if startup_sequence_id < 0:
                print("Startup sequence ID is negative, skipping plotting.")
                return
            if cutoff_sequence_id < 0:
                print("Cutoff sequence ID is negative, skipping plotting.")
                return
            if cutoff_sequence_id > seq_ids[-1]:
                print(f"Cutoff sequence ID {cutoff_sequence_id} is greater than the last sequence ID {seq_ids[-1]}, adjusting to the last sequence ID.")
                cutoff_sequence_id = seq_ids[-1]
                    
            startup_sequence_id_found = False
            for i, x in enumerate(seq_ids):
                if x > startup_sequence_id and startup_sequence_id_found is False:
                    startup_index = i
                    startup_sequence_id_found = True
                if x > cutoff_sequence_id:
                    seq_ids = seq_ids[startup_index:i]
                    positions = positions[startup_index:i]
                    break

            startup_sequence_id_found = False
            for i, x in enumerate(action_seq_ids):
                if x > startup_sequence_id and startup_sequence_id_found is False:
                    startup_index = i
                    startup_sequence_id_found = True
                if x > cutoff_sequence_id:
                    action_seq_ids = action_seq_ids[startup_index:i]
                    action_positions = action_positions[startup_index:i]
                    action_predictions = action_predictions[startup_index:i]
                    break

        
        fig, axs = plt.subplots(f['metadata'].attrs['total_dof']-1, 1, figsize=(20, 2 * (f['metadata'].attrs['total_dof']-1)), sharex=True)
        j = 0
        for i in range(f['metadata'].attrs['total_dof']):
            label_added = False
            if i == 3:
                continue  # Skip the 4th joint (index 3)
            for i_prediction in range(action_seq_ids.shape[0]):
                # Each prediction starts at action_seq_ids[i_prediction] and only has a prediction once every 4 timesteps (record divisor) and this 75 times (chunk size)
                action_positions_steps = range(action_seq_ids[i_prediction], action_seq_ids[i_prediction] + f['metadata'].attrs['record_divisor']*action_predictions.shape[1], f['metadata'].attrs['record_divisor'])
                if not label_added:
                    axs[j].plot(action_positions_steps, action_predictions[i_prediction, :, i], label=f'Predicted action teachbot', linestyle=':', color='grey')
                    label_added = True
                else:
                    axs[j].plot(action_positions_steps, action_predictions[i_prediction, :, i], linestyle=':', color='grey')
            axs[j].plot(seq_ids[:], positions[:, i], label=f'Robot joint {i}')
            axs[j].plot(action_seq_ids[:], action_positions[:, i], label=f'Robot joint {i} via actions', linestyle='--')    
            axs[j].set_title(f'Trajectory joint {i}')
            if i == f['metadata'].attrs['total_dof'] - 1:
                axs[j].set_xlabel('sequence ID')
            axs[j].set_ylabel(f'joint angle ($^\circ$)')
            axs[j].legend(loc='lower right')
            j += 1
        plt.show()

if __name__ == "__main__":
    logs_dir = '../tos_app_data/POLICY_EXECUTION_FILES'
    hdf5_files = [f for f in os.listdir(logs_dir) if f.endswith('.hdf5')]
    hdf5_file_path = os.path.join(logs_dir, hdf5_files[0])
    print(f"Opening HDF5 file: {hdf5_files[0]}")

    plot_trajectories(hdf5_file_path)