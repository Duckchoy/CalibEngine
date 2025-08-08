import torch
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import numpy as np
import random
import os
import subprocess
from collections import deque
import configparser
import csv
import shutil
import re

print("[Checkpoint #0] Launging DDRG. Importing necessary libraries and modules...")
np.set_printoptions(precision=4, suppress=True)

# === Parse INI File for OPC Parameter Bounds ===
def get_optvars_from_ini(ini_path):
    """
    Parse a custom INI-style file to extract parameter bounds.
    Each line should look like:
    optvar = var1 1 0.060000 0.000000 0.120000 0.03000 0.000000

    Only variables starting with flag=1 will be used. Extract:
    - name
    - initial, lower, upper, step

    Args:
        ini_path (str): Path to the parameter INI file.

    Returns:
        List[Tuple[str, float, float, float, float]]: (var_name, initial, lower, upper, step)
    """
    param_bounds = []
    with open(ini_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or not line.startswith("optvar"):
                continue
            try:
                tokens = line.split()
                if len(tokens) < 8:
                    print(f"Skipping malformed line: {line}")
                    continue
                _, _, varname, flag, initial, lower, upper, step, *_ = tokens
                if int(flag) != 1:
                    continue
                initial = float(initial)
                lower = float(lower)
                upper = float(upper)
                step = float(step)
                if lower >= upper:
                    raise ValueError(f"Invalid bounds for {varname}: lower >= upper")
                if step <= 0:
                    raise ValueError(f"Invalid step size for {varname}: must be > 0")
                param_bounds.append((varname, initial, lower, upper, step))
            except ValueError as ve:
                print(f"Skipping line due to error: {line} {ve}")
                continue

    if not param_bounds:
        raise ValueError("No valid parameters found in the INI file.")

    total_combinations = 1
    for _, _, lower, upper, step in param_bounds:
        count = int((upper - lower) / step) + 1
        total_combinations *= count
    print(f"[Checkpoint #1] {len(param_bounds)} parameters loaded. Total number of models = {total_combinations}")

    return param_bounds

# === Actor Network ===
class Actor(nn.Module):
    """
    Neural network representing the actor in DDPG.
    Maps state vectors to continuous action vectors.

    Args:
        state_dim (int): Dimension of the input state.
        action_dim (int): Dimension of the output action.
        actor_hidden_dim (int, optional): Number of hidden units per layer.
    """
    def __init__(self, state_dim, action_dim, actor_hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        """
        Performs a forward pass through the model using the provided input state.

        Args:
            state (torch.Tensor): The input tensor representing the current state.

        Returns:
            torch.Tensor: The output tensor produced by the model after the forward pass.
        """
        return self.model(state)

# === Critic Network ===
class Critic(nn.Module):
    """
    Neural network representing the critic in DDPG.
    Evaluates the value of state-action pairs.

    Args:
        state_dim (int): Dimension of the input state.
        action_dim (int): Dimension of the input action.
        critic_hidden_dim (int, optional): Number of hidden units per layer.
    """
    def __init__(self, state_dim, action_dim, critic_hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(critic_hidden_dim, critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(critic_hidden_dim, 1)
        )

    def forward(self, state, action):
        """
        Performs a forward pass through the model by concatenating the state and action tensors.

        Args:
            state (torch.Tensor): The state input tensor.
            action (torch.Tensor): The action input tensor.

        Returns:
            torch.Tensor: The output of the model after processing the concatenated input.
        """
        return self.model(torch.cat([state, action], dim=-1))

# === Replay Buffer ===
class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.

    Args:
        max_size (int, optional): Maximum number of transitions to store.
    """
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        print(f"[Checkpoint #8] ReplayBuffer initialized with max size={max_size}")

    def push(self, state, action, reward, next_state):
        """
        Store a transition in the buffer.

        Args:
            state (array-like): The observed state.
            action (array-like): The action taken.
            reward (float): The reward received.
            next_state (array-like): The next observed state.
        """
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of samples to return.

        Returns:
            tuple: Batch of (state, action, reward, next_state) tensors.
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*batch)
        return (torch.FloatTensor(state), torch.FloatTensor(action),
                torch.FloatTensor(reward), torch.FloatTensor(next_state))

    def size(self):
        """
        Get the current size of the buffer.

        Returns:
            int: Number of transitions stored.
        """
        return len(self.buffer)

# === OPC Environment with External Model Execution ===
class OPCEnvironment:
    """
    OPCEnvironment provides an interface to an external lithography model for Optical Proximity Correction (OPC) parameter optimization.

    This environment automates the process of generating model source files with updated parameters, creating corresponding configuration files, running external simulation binaries, and extracting performance metrics (such as RMSE) for optimization workflows. It also supports finite-difference gradient estimation for use in gradient-based optimization algorithms.

    Attributes:
        template_path (str): Path to the model template file.
        ini_path (str): Path to the original INI configuration file.
        calib_settings_file (str): Path to the calibration settings file.
        epsilon (float): Epsilon value for finite-difference gradient estimation.
        output_dir (str): Directory to store generated models and run data.
        log_file (str): File to log model runs and parameters.
        model_id (int): Counter for the current model/run.
        use_gradients (bool): Whether to compute gradients via finite differences.

    Methods:
        create_run_dir(parameter_vector):
            Generates a new model source file and configuration files for a given parameter vector, and prepares the run directory.

        get_rmse(parameter_vector):
            Runs the external simulator with the generated model and configuration, extracts the RMSE value, and (optionally) computes gradients with respect to the parameters.

        get_reward(parameter_vector):
            Returns the negative RMSE as a reward signal for optimization algorithms.

        FileNotFoundError: If required files (template, INI, calibration settings, or simulator binary) are missing.
        ValueError: If output files are empty or do not contain valid RMSE values.

    Typical usage example:
        env = OPCEnvironment(model_template_path, var_names, ini_file_path, calib_cpp, grad_perturbation)
        rmse, gradients = env.get_rmse(parameter_vector)
        reward = env.get_reward(parameter_vector)
    """
    def __init__(self, model_template_path, var_names, ini_file_path, calib_cpp, grad_perturbation, model_run_dir="RunData", log_file="model_log.txt", use_gradients=False):
        if not os.path.isfile(model_template_path):
            raise FileNotFoundError(f"Model template file '{model_template_path}' does not exist.")

        self.template_path = model_template_path
        self.var_names = var_names
        self.ini_path = ini_file_path
        self.calib_settings_file = calib_cpp
        self.epsilon = grad_perturbation
        self.output_dir = model_run_dir
        self.log_file = log_file
        self.model_id = 0
        self.use_gradients = use_gradients

        if os.path.exists(self.output_dir):
            print(f"Warning: Output directory '{self.output_dir}' exists and will be overwritten.")
            shutil.rmtree(self.output_dir)
        os.makedirs(os.path.abspath(self.output_dir), exist_ok=False)
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("Model Log (parameters and associated filenames):")
        print(f"[Checkpoint #3] OPCEnvironment initialized with model template: {self.template_path}, output dir: {self.output_dir}")


    def create_run_dir(self, parameter_vector):
        """
        Creates a new run directory for a model configuration based on the provided parameter vector.
        This involves generating a new model source file with updated parameters, creating a corresponding
        INI configuration file, and copying/updating the calibration settings file to reference the new model.

            parameter_vector (array-like): Sequence of parameter values to substitute into the model template.
                Each value corresponds to a variable name in `self.var_names`.

            tuple:
                model_subdir (str): Path to the newly created model subdirectory.
                model_filename (str): Filename of the generated model source file.
                ini_copy_path (str): Path to the newly created INI configuration file.

        Raises:
            FileNotFoundError: If the calibration settings file does not exist.

        Side Effects:
            - Creates a new subdirectory under `self.output_dir`.
            - Writes a new model source file with updated parameter values and namespace.
            - Writes a new INI file referencing the new model file.
            - Copies and updates the calibration settings file to reference the new model.
        """
        self.model_id = len(os.listdir(self.output_dir))
        model_subdir = os.path.join(self.output_dir, f"modeldata_{self.model_id}")
        os.makedirs(model_subdir, exist_ok=False)

        template_basename = os.path.splitext(os.path.basename(self.template_path))[0]
        model_filename = f"{template_basename}_{self.model_id}.cpp"
        model_path = os.path.join(model_subdir, model_filename)

        def replace_old_value_in_line(line, new_value):
            prefix = 'FMOOptVar('
            if prefix not in line:
                return line  # Return unchanged if not matching

            start = line.index(prefix) + len(prefix)
            end = line.index(')', start)
            args = line[start:end]

            arg_list = [arg.strip() for arg in args.split(',')]
            arg_list[1] = str(new_value)
            new_args = ', '.join(arg_list)
            new_line = line[:start] + new_args + line[end:]
            return new_line

        with open(self.template_path, 'r', encoding='utf-8') as template_file:
            model_data = template_file.read()

        lines = model_data.splitlines()
        new_lines = []
        for line in lines:
            for name, val in zip(self.var_names, parameter_vector):
                if f'FMOOptVar("{name}",' in line:
                    line = replace_old_value_in_line(line, val)
            new_lines.append(line)

        model_data = "\n".join(new_lines)

        namespace_pattern = re.compile(r'namespace\s+\w+\s*\{')
        model_basename = os.path.splitext(os.path.basename(model_path))[0]
        model_data = namespace_pattern.sub(f'namespace {model_basename} {{', model_data, count=1)

        with open(model_path, 'w', encoding='utf-8') as new_model_file:
            new_model_file.write(model_data)

        # Step 2: Create a new INI configuration file pointing to the new model
        with open(self.ini_path, 'r', encoding='utf-8') as ini_original:
            ini_lines = ini_original.readlines()

        ini_basename = os.path.splitext(os.path.basename(self.ini_path))[0]
        ini_copy_path = os.path.join(model_subdir, f"{ini_basename}_{self.model_id}.ini")
        model_file_only = os.path.basename(model_path)
        with open(ini_copy_path, 'w', encoding='utf-8') as ini_copy:
            for line in ini_lines:
                if line.strip().startswith("selected_model_file"):
                    ini_copy.write(f"selected_model_file = {model_file_only}\n")
                elif line.strip().startswith("open_model_file"):
                    ini_copy.write(f"open_model_file = {model_file_only}\n")
                else:
                    ini_copy.write(line)

        # Step 3: Copy and update calib_settings file
        calib_settings_src = self.calib_settings_file
        calib_settings_dst = os.path.join(model_subdir, self.calib_settings_file)
        if not os.path.isfile(calib_settings_src):
            raise FileNotFoundError(f"Calibration settings file '{calib_settings_src}' does not exist.")
        with open(calib_settings_src, 'r', encoding='utf-8') as f:
            calib_lines = f.readlines()
        with open(calib_settings_dst, 'w', encoding='utf-8') as f:
            for line in calib_lines:
                if 'pCalibrator->SetModel(' in line:
                    leading_ws = line[:len(line) - len(line.lstrip())]  # preserve leading whitespace
                    f.write(f'{leading_ws}pCalibrator->SetModel("{model_filename}");\n')
                else:
                    f.write(line)

        return model_subdir, model_filename, os.path.basename(ini_copy_path)

    def get_rmse(self, parameter_vector):
        """
        Calculates the Root Mean Square Error (RMSE) and its gradients with respect to a set of parameters by generating model files

        This method performs the following steps:
            1. Generates a new model file by substituting parameter values into a template.
            2. Creates a corresponding INI configuration file pointing to the new model.
            3. Copies and updates the calib_settings file for the new model.
            4. Logs the parameter vector used for this run.
            5. Executes an external binary simulator with the generated configuration.
            6. Reads the RMSE value produced by the simulator.
            7. For each parameter, perturbs it by a small epsilon, regenerates the model and config, reruns the simulator, and estimates the gradient using finite differences.

            parameter_vector (array-like): Array of parameter values to be optimized.

            tuple:
                rmse_value (float): The computed RMSE for the given parameter vector.
                gradients (np.ndarray): Array of gradients of the RMSE with respect to each parameter.
        """
        model_subdir, model_filename, ini_id = self.create_run_dir(parameter_vector)
        print(f">> Model subdirectory created: {model_subdir}")
        # model_id = int(os.path.basename(model_subdir).split('_')[-1])

        with open(self.log_file, 'a', encoding='utf-8') as log:
            log.write(f"{model_filename}: {parameter_vector}\n")

        print(">> Getting model RMSE...")
        exec_bin = os.path.join(os.environ.get("exec", ""), "bin", "calib_manager2")  # 'exec' proprietary
        if not os.path.isfile(exec_bin):
            raise FileNotFoundError(f"Could not find binary at {exec_bin}. Check that $exec is set correctly.")
        run_log_path = os.path.join(model_subdir, f"RUN_{self.model_id}.log")
        with open(run_log_path, "w", encoding="utf-8") as run_log:
            process = subprocess.run([exec_bin, ini_id], cwd=model_subdir, stdout=run_log, stderr=subprocess.STDOUT, check=True)
        print(f">> Run exited with status {process.returncode}")
        print("------------------")

        # Find the IterationData_0 directory inside the model_subdir
        iteration_dir = os.path.join(model_subdir, "IterationData_0")
        if not os.path.isdir(iteration_dir):
            raise FileNotFoundError(f"Could not find IterationData_0 in {model_subdir}")
        calibration_log = os.path.join(iteration_dir, "calibration.log")
        if not os.path.isfile(calibration_log):
            raise FileNotFoundError(f"Could not find calibration.log in {iteration_dir}")

        with open(calibration_log, 'r', encoding='utf-8') as result_file:
            lines = [line for line in result_file if line.strip()]
            if not lines:
                raise ValueError(f"calibration.log is empty in {iteration_dir}")
            last_line = lines[-1]
            # Try splitting by tab or comma, and extract the last float value
            if '\t' in last_line:
                cols = last_line.strip().split('\t')
            else:
                cols = last_line.strip().split(',')
            # Find the last column that can be converted to float
            for col in reversed(cols):
                try:
                    model_rmse = float(col)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Could not find a valid float RMSE value in line: {last_line}")
        print(f">> Model ID: {self.model_id}, RMSE: {model_rmse}")

        # TODO: Step 4: Calculate gradients using finite differences
        gradients = np.zeros_like(parameter_vector)
        if self.use_gradients:
            print(">> Calculating gradients...")
            with open(self.ini_path, 'r', encoding='utf-8') as ini_original:
                ini_lines = ini_original.readlines()

            for i in range(len(parameter_vector)):
                perturbed = np.array(parameter_vector)
                perturbed[i] += self.epsilon

                temp_model_filename = f"grad_temp_model_{self.model_id}_{i}.cpp"
                temp_model_path = os.path.join(model_subdir, temp_model_filename)

                with open(self.template_path, 'r', encoding='utf-8') as tf:
                    grad_model_data = tf.read()
                    for name, val in zip(self.var_names, perturbed):
                        grad_model_data = grad_model_data.replace(f'FMOOptVar("{name}",', f'FMOOptVar("{name}", {val:.6f},')

                with open(temp_model_path, 'w', encoding='utf-8') as f:
                    f.write(grad_model_data)

                temp_ini_path = os.path.join(model_subdir, f"grad_config_{self.model_id}_{i}.ini")
                with open(temp_ini_path, 'w', encoding='utf-8') as f:
                    for line in ini_lines:
                        if line.strip().startswith("selected_model_file"):
                            f.write(f"selected_model_file = {temp_model_path}\n")
                        elif line.strip().startswith("open_model_file"):
                            f.write(f"open_model_file = {temp_model_path}\n")
                        else:
                            f.write(line)

                temp_calib_settings_dst = os.path.join(model_subdir, f"calib_settings_grad_{i}")
                with open(self.calib_settings_file, 'r', encoding='utf-8') as f:
                    grad_calib_lines = f.readlines()
                with open(temp_calib_settings_dst, 'w', encoding='utf-8') as f:
                    for line in grad_calib_lines:
                        if 'pCalibrator->SetModel(' in line:
                            f.write(f'pCalibrator->SetModel("{temp_model_filename}");\n')
                        else:
                            f.write(line)

                subprocess.run(["external_binary", temp_ini_path], check=True)
                with open("rmse_output.txt", 'r', encoding='utf-8') as f:
                    perturbed_rmse = float(f.readline().strip())

                # TODO: the perturbed models can be run parallelly (either on clients or threads, depends on how many optvars are to be perturbed?)
                # Once both are complete you can do the gradient calculation. Obviously, careful with slow clients and other such issues.
                gradients[i] = (perturbed_rmse - model_rmse) / self.epsilon

        return model_rmse, gradients

    def get_reward(self, parameter_vector):
        """
        Take a step in the environment using the given parameters.

        Args:
            parameter_vector (array-like): OPC parameter values.

        Returns:
            float: Reward (negative RMSE).
        """
        reward, _ = self.get_rmse(parameter_vector)
        return -reward

# === DDPG ===
class DDPG:
    """
    Deep Deterministic Policy Gradient (DDPG) agent for continuous action spaces.

    This class implements the DDPG algorithm, which combines actor-critic methods with experience replay and target networks for stable learning in environments with continuous action spaces.

        state_dim (int): Dimension of the input state space.
        bounds (list of tuples): Each tuple specifies (lower, upper, step) for discretizing and scaling each action dimension.
        lr_actor (float): Learning rate for the actor network optimizer.
        lr_critic (float): Learning rate for the critic network optimizer.
        max_buffer (int): Maximum size of the replay buffer.
        gamma (float): Discount factor for future rewards.
        tau (float): Soft update coefficient for target networks.
        actor_hidden_dim (int): Number of hidden units in the actor network.
        critic_hidden_dim (int): Number of hidden units in the critic network.

    Attributes:
        actor (nn.Module): The actor network responsible for policy approximation.
        actor_target (nn.Module): Target actor network for stable updates.
        critic (nn.Module): The critic network for value estimation.
        critic_target (nn.Module): Target critic network for stable updates.
        actor_optimizer (torch.optim.Optimizer): Optimizer for the actor network.
        critic_optimizer (torch.optim.Optimizer): Optimizer for the critic network.
        gamma (float): Discount factor.
        tau (float): Soft update parameter.
        replay_buffer (ReplayBuffer): Experience replay buffer.
        bounds (list): Action bounds for scaling and discretization.

    Methods:
        scale_action(action): Scales and discretizes actions from [-1, 1] to the specified bounds.
        select_action(state, noise): Returns a scaled action for a given state, with optional exploration noise.
        train(batch_size=64): Performs a training step for both actor and critic networks using samples from the replay buffer.
    """
    def __init__(self, state_dim, action_dim, bounds, lr_actor, lr_critic, max_buffer, gamma, tau, actor_hidden_dim, critic_hidden_dim):
        self.actor = Actor(state_dim, action_dim, actor_hidden_dim)
        print(f"[Checkpoint #4] Actor network initiated with state_dim={state_dim}, action_dim={action_dim}, hidden_dim={actor_hidden_dim}")
        self.actor_target = Actor(state_dim, action_dim, actor_hidden_dim)
        print("[Checkpoint #5] Actor target network initiated with similar dimensions.")
        self.critic = Critic(state_dim, action_dim, critic_hidden_dim)
        print(f"[Checkpoint #6] Critic network initiated with state_dim={state_dim}, action_dim={action_dim}, hidden_dim={critic_hidden_dim}")
        self.critic_target = Critic(state_dim, action_dim, critic_hidden_dim)
        print("[Checkpoint #7] Critic target network initiated with similar dimensions.")
        # Target networks stabilize training by providing slowly-updated reference values, which helps prevent divergence during learning.

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer(max_buffer)

        self.bounds = bounds

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        print("[Checkpoint #9] Deep Deterministic Policy Gradient agent initialized..")

    def scale_action(self, action):
        """
        Scale and discretize the action from [-1, 1] to the parameter bounds.

        Args:
            action (array-like): Raw action output from the actor.

        Returns:
            np.ndarray: Scaled and discretized action.
        """
        scaled = []
        for i, val in enumerate(action):
            low, high, step_size = self.bounds[i]
            val = (val + 1) / 2  # Map from [-1, 1] to [0, 1]
            val = low + val * (high - low)
            val = round(val / step_size) * step_size
            val = np.clip(val, low, high)
            scaled.append(val)
        print(f">> Selected scaled action: {scaled}")
        return np.array(scaled)

    def select_action(self, state, noise):
        """
        Select an action for a given state, with optional exploration noise.

        Args:
            state (array-like): Current state.
            noise (float, optional): Standard deviation of Gaussian noise.

        Returns:
            np.ndarray: Scaled action.
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        raw_action = self.actor(state).detach().numpy()[0]
        noisy_action = raw_action + noise * np.random.randn(*raw_action.shape)
        print(f">> Selected noisy action: {noisy_action}")
        return self.scale_action(noisy_action)

    def train(self, batch_size=64):
        """
        Train the actor and critic networks using a batch from the replay buffer.

        Args:
            batch_size (int, optional): Number of samples per batch.
        """
        if self.replay_buffer.size() < batch_size:
            return

        state, action, reward, next_state = self.replay_buffer.sample(batch_size)

        with torch.no_grad():
            target_q = reward + self.gamma * self.critic_target(next_state, self.actor_target(next_state))

        critic_loss = nn.MSELoss()(self.critic(state, action), target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.get_reward()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.get_reward()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        print(f"[Checkpoint #11] Critic loss: {critic_loss.item()}, Actor loss: {actor_loss.item()}")

# === Training ===
def train_drl_agent(
    action_vector_len,
    actor_hidden_dim,
    actor_lr,
    bounds,
    calib_cpp,
    critic_hidden_dim,
    critic_lr,
    episodes,
    exploration_noise,
    gamma,
    grad_perturbation,
    ini_file_path,
    initial_action,
    max_buffer_size,
    max_steps,
    model_template_path,
    state_vector_len,
    target_vars,
    tau,
    use_gradients=True
):
    """
    Train a DDPG agent in the OPC environment using the provided model template and parameter bounds.

    This function sets up the OPC environment and a DDPG agent, then runs multiple training episodes.
    In each episode, the agent interacts with the environment, collects experience, and updates its policy and value networks.
    Training metrics are logged to a CSV file, and model checkpoints are saved after each episode.

    Args:
        action_vector_len (int): Length of the action vector.
        actor_hidden_dim (int): Number of hidden units in the actor network.
        actor_lr (float): Learning rate for the actor network.
        bounds (list of tuple): List of parameter bounds as (lower, upper, step) for each parameter.
        calib_cpp (str): Path to the calibration settings file.
        critic_hidden_dim (int): Number of hidden units in the critic network.
        critic_lr (float): Learning rate for the critic network.
        episodes (int): Number of training episodes.
        exploration_noise (float): Initial exploration noise for action selection.
        gamma (float): Discount factor for future rewards.
        grad_perturbation (float): Epsilon for finite-difference gradient estimation.
        ini_file_path (str): Path to the OPC INI file.
        initial_action (list): Initial action values (from INI).
        max_buffer_size (int): Maximum size of the replay buffer.
        max_steps (int): Maximum steps per episode.
        model_template_path (str): Path to the OPC model template file.
        state_vector_len (int): Length of the state vector.
        target_vars (list): List of variable names to optimize.
        tau (float): Soft update coefficient for target networks.
        use_gradients (bool): Whether to use gradients in the state.
    Returns:
        DDPG: The trained DDPG agent.
    """
    env = OPCEnvironment(model_template_path, target_vars, ini_file_path, calib_cpp, grad_perturbation, use_gradients=use_gradients)
    agent = DDPG(state_vector_len, action_vector_len, bounds,
                 actor_lr, critic_lr, max_buffer_size, gamma, tau, actor_hidden_dim, critic_hidden_dim)

    print(f"[Checkpoint #10] Starting training for {episodes} episodes, {max_steps} steps per episode.")
    print("=========================================================")
    log_file = "training_metrics.csv"
    with open(log_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "TotalReward"])

    for episode in range(episodes):
        if initial_action is not None:
            prev_action = np.array(initial_action)
        else:
            prev_action = np.zeros(action_vector_len)
        prev_reward = 0.0
        prev_gradients = np.zeros(action_vector_len)
        if use_gradients:
            state = np.concatenate([prev_action, [prev_reward], prev_gradients])
        else:
            state = np.concatenate([prev_action, [prev_reward]])
        total_reward = 0
        print(f">> Starting episode {episode + 1}/{episodes} with initial state: {state}")
        for step in range(max_steps):
            action = agent.select_action(state, noise=exploration_noise)
            rmse, gradients = env.get_rmse(action)
            reward = -rmse
            total_reward += reward
            if use_gradients:
                next_state = np.concatenate([action, [reward], gradients])
            else:
                next_state = np.concatenate([action, [reward]])
            agent.replay_buffer.push(state, action, reward, next_state)
            agent.train()
            state = next_state
            print(f"*** Episode: {episode + 1}, Step: {step + 1}, Action: {action}, Reward: {reward}, State: {state}")

        print(f"Episode {episode + 1}/{episodes} over, Total Reward: {total_reward:.4f}")
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([episode + 1, total_reward])
        exploration_noise *= 0.99

        torch.save(agent.actor.state_dict(), "actor_checkpoint.pt")
    torch.save(agent.critic.state_dict(), "critic_checkpoint.pt")
    print("=========================================================")
    print("[Checkpoint #13] Training completed. Checkpoints saved: actor_checkpoint.pt, critic_checkpoint.pt")
    return agent

# === Inference ===
def run_inference(agent, input_state_vec):
    """
    Run inference using a trained agent to obtain optimized parameters.

    Args:
        agent (DDPG): Trained DDPG agent.
        input_state_vec (array-like): Input state vector.

    Returns:
        np.ndarray: Optimized OPC parameters.
    """
    state = torch.FloatTensor(input_state_vec).unsqueeze(0)
    with torch.no_grad():
        optimized_params = agent.scale_action(agent.actor(state).squeeze(0).numpy())
    print(f"[Checkpoint #14] Running inference with input state: {input_state_vec}, Inference result (optimized parameters): {optimized_params}")
    return optimized_params

# === Main ===
if __name__ == "__main__":
    # === File Paths ===
    INI_FILE = "CONTROL.ini"
    USE_GRADIENTS = False  # TODO: Gradient use is not implemented/optimized yet, parallelization is needed.

    ini_vals = get_optvars_from_ini(INI_FILE)
    optvars = [entry[0] for entry in ini_vals]
    initial_values = [entry[1] for entry in ini_vals]
    ini_bounds = [(entry[2], entry[3], entry[4]) for entry in ini_vals]

    # === Environment and Agent Configuration ===
    # For dynamic state: state = [previous_action (n), previous_reward (1)]
    # For dynamic state: [action_dim] + [reward] + [gradients]
    ACTION_DIM = len(ini_bounds)
    # Adjust state dimension based on use_gradients
    if USE_GRADIENTS:
        STATE_DIM = ACTION_DIM + 1 + ACTION_DIM
    else:
        STATE_DIM = ACTION_DIM + 1

    trained_agent = train_drl_agent(
        action_vector_len=ACTION_DIM,
        actor_hidden_dim=256,
        actor_lr=1e-3,
        bounds=ini_bounds,
        calib_cpp="CALIB.cpp",
        critic_hidden_dim=256,
        critic_lr=1e-3,
        episodes=2, # 100
        exploration_noise=0.1,
        gamma=0.99,
        grad_perturbation=1e-4,
        ini_file_path=INI_FILE,
        initial_action=initial_values,
        max_buffer_size=100000,
        max_steps= 2, # 50
        model_template_path="ModelTemplate_Rev0.cpp",
        state_vector_len=STATE_DIM,
        target_vars=optvars,
        tau=0.005,
        use_gradients=USE_GRADIENTS
    )

    if USE_GRADIENTS:
        initial_state = np.concatenate([initial_values, [0.0], np.zeros(ACTION_DIM)])
    else:
        initial_state = np.concatenate([initial_values, [0.0]])
    optimal_params = run_inference(trained_agent, initial_state)
    print("Optimal OPC Parameters from Inference:", optimal_params)
