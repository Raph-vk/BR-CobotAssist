import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def _interpolate_actions(start_seq_id, predicted_actions, current_joint_position, total_dof, record_divisor):
        """
        Interpolate actions to provide smooth trajectories at full control frequency. 
        Copied from PolicyInterface class in modules.policy.act.policy_act. but removes the logger and the class context.
        
        The policy predicts actions every record_divisor control cycles (e.g., every 4 cycles at 62.5 Hz),
        but we need to provide smooth control commands at the full control frequency (250 Hz).
        
        Args:
            start_seq_id: Starting sequence ID
            predicted_actions: List of predicted actions from policy (at reduced frequency)
            current_joint_position: Current joint position to start interpolation from
            total_dof: Total degrees of freedom 
            record_divisor: Number of control cycles to interpolate over 
        Returns:
            List of interpolated actions at full control frequency
        """        
        if predicted_actions is None or len(predicted_actions) == 0:
            return []
            
        interpolated_actions = []
        
        # Start with current position as the first reference point
        prev_action = np.array(predicted_actions[0]) 

        # Interpolate between consecutive predicted actions
        for i in range(1, len(predicted_actions)):
            next_action = predicted_actions[i]  # Access numpy array directly
            
            # Ensure next_action has correct dimensions (total_dof elements: joints + gripper)
            if len(next_action) != total_dof:
                if len(next_action) < total_dof:
                    next_action = np.pad(next_action, (0, total_dof - len(next_action)), 'constant')
                else:
                    next_action = next_action[:total_dof]

            # Generate interpolated steps between prev_action and next_action
            for step in range(0, record_divisor):
                # Linear interpolation factor (0.0 to 1.0)
                alpha = step / record_divisor
                
                # Interpolate between previous and next action
                interpolated_action = (1 - alpha) * prev_action + alpha * next_action
                # if step == record_divisor:
                    # print(f'TEST: {interpolated_action == next_action}, \nInterpolated action: \n{interpolated_action}, next_action: \n{next_action}')
                # Create action dictionary with interpolated values
                interpolated_seq_id = start_seq_id + i * record_divisor + step
                interpolated_actions.append({
                    'seq_id': interpolated_seq_id,
                    'action': interpolated_action.tolist()
                })

            # Update prev_action for next iteration
            prev_action = next_action

        # Append last action
        interpolated_actions.append({
            'seq_id': start_seq_id + len(predicted_actions) * record_divisor,
            'action': next_action.tolist()
        })
        
        print(f'start seq_id: {start_seq_id}, first seq_id in interpolated actions: {interpolated_actions[0]["seq_id"]}')
        print(f'last couple seq_id in interpolated actions: {interpolated_actions[-3]["seq_id"]},{interpolated_actions[-2]["seq_id"]},{interpolated_actions[-1]["seq_id"]}')
        print(f'length of interpolated actions: {len(interpolated_actions)}')
        return interpolated_actions


def plot_trajectories(h5_file_path, snippet_ids=[3000, 3030]):
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
        
        interpolated_actions = []
        for i in range(action_seq_ids.shape[0]):
            interpolated_actions.append(_interpolate_actions(action_seq_ids[i], action_predictions[i], action_positions[i], total_dof=f['metadata'].attrs['total_dof'], record_divisor=f['metadata'].attrs['record_divisor']))
        print(f'action_seq_ids.shape[0]: {action_seq_ids.shape[0]}')
        print(f'action_predictions shape: {action_predictions.shape}')
        print(f'shape of interpolated_actions: {type(interpolated_actions)}, {len(interpolated_actions)}')
        print(f'shape of interpolated_actions[0]: {type(interpolated_actions[0])}, {len(interpolated_actions[0])}')
        print(f'shape of interpolated_actions[0][0]: {type(interpolated_actions[0][0])}, {len(interpolated_actions[0][0])}, {(interpolated_actions[0][0])}')
        print(f'interpolated_actions[0][:][seq_id]: {interpolated_actions[0][0]["seq_id"]}')

        ### Plotting the prediction and interpolated actions
        fig, axs_interpol = plt.subplots(f['metadata'].attrs['total_dof']-1, 1, figsize=(20, 2 * (f['metadata'].attrs['total_dof']-1)), sharex=True)
        j = 0
        for i in range(f['metadata'].attrs['total_dof']):
            label_added_pred = False
            label_added_interpol = False
            if i == 3:
                continue  # Skip the 4th joint (index 3)
            for i_prediction in range(action_seq_ids.shape[0]):
                print(f'new prediction {i_prediction} for joint {i}')
                # Each prediction starts at action_seq_ids[i_prediction] and only has a prediction once every 4 timesteps (record divisor) and this 75 times (chunk size)
                action_positions_steps = range(action_seq_ids[i_prediction]+f['metadata'].attrs['record_divisor'], action_seq_ids[i_prediction] + f['metadata'].attrs['record_divisor']*(1+action_predictions.shape[1]), f['metadata'].attrs['record_divisor'])
                for k in range(len(interpolated_actions[i_prediction])):
                    if not label_added_interpol:
                        axs_interpol[j].scatter(interpolated_actions[i_prediction][k]['seq_id'], interpolated_actions[i_prediction][k]['action'][i], label=f'Interpolated action teachbot', color='red', s=5)
                        label_added_interpol = True
                    else:
                        axs_interpol[j].scatter(interpolated_actions[i_prediction][k]['seq_id'], interpolated_actions[i_prediction][k]['action'][i], color='red', s=5)
                    if not label_added_pred:
                        axs_interpol[j].scatter(action_positions_steps, action_predictions[i_prediction, :, i], label=f'Predicted action teachbot', color='blue', s=5)
                        label_added_pred = True
                    else:
                        axs_interpol[j].scatter(action_positions_steps, action_predictions[i_prediction, :, i], color='blue', s=5)
            axs_interpol[j].set_title(f'Trajectory joint {i}')
            if i == f['metadata'].attrs['total_dof'] - 1:
                axs_interpol[j].set_xlabel('sequence ID')
                axs_interpol[j].set_ylabel(f'closed/open [0,1]')
            else:
                axs_interpol[j].set_ylabel(f'joint angle ($^\circ$)')
            axs_interpol[j].legend(loc='lower right')
            j += 1
        plt.show()

        ### Plotting the original trajectories with the predictions
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