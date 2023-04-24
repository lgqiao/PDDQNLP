import datetime
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import matplotlib.pyplot as plt
import click
import gym
from UAV_MEC_env_1 import UAV_MEC
import numpy as np
# from gym.wrappers import Monitor
# from torch.utils.tensorboard import SummaryWriter

from agents.nstep_pdqn import PDQNNStepAgent
from utils import ClickPythonLiteralOption


# from wrappers import SoccerScaledParameterisedActionWrapper

# SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def evaluate(env, agent, episodes=1000):
    returns = []
    timesteps = []
    goals = []

    for i_eps in range(episodes):
        state = env.reset()
        terminal = False
        n_steps = 0
        total_reward = 0.
        info = {'status': "NOT_SET"}
        while not terminal:
            n_steps += 1
            state = np.array(state, dtype=np.float32, copy=False)
            act, act_param, all_action_parameters = agent.act(state)
            action = env.find_action(act)
            state, reward, terminal = env.step(action, act_param)
            total_reward += reward
        print(info['status'])
        goal = info['status'] == 'GOAL'
        timesteps.append(n_steps)
        returns.append(total_reward)
        goals.append(goal)
    return np.column_stack((returns, timesteps, goals))


@click.command()
@click.option('--seed', default=0, help='Random seed.', type=int)
@click.option('--episodes', default=10000, help='Number of epsiodes.', type=int)
@click.option('--evaluation-episodes', default=1000, help='Episodes over which to evaluate after training.', type=int)
@click.option('--batch-size', default=64, help='Minibatch size.', type=int)
@click.option('--gamma', default=0.99, help='Discount factor.', type=float)
@click.option('--update-ratio', default=0.1, help='Ratio of updates to samples.', type=float)
@click.option('--inverting-gradients', default=True,
              help='Use inverting gradients scheme instead of squashing function.', type=bool)
@click.option('--initial-memory-threshold', default=1000, help='Number of transitions required to start learning.',
              type=int)
@click.option('--replay-memory-size', default=5000, help='Replay memory size in transitions.', type=int)  # 500000
@click.option('--epsilon-start', default=0.95, help='Initial epsilon value.', type=float)
@click.option('--epsilon-steps', default=1000, help='Number of episodes over which to linearly anneal epsilon.',
              type=int)
@click.option('--epsilon-final', default=0.02, help='Final epsilon value.', type=float)
@click.option('--learning-rate-actor', default=0.00001, help="Actor network learning rate.", type=float)
@click.option('--learning-rate-actor-param', default=0.00001, help="Critic network learning rate.", type=float)
@click.option('--clip-grad', default=1., help="Gradient clipping.", type=float)  # 1 better than 10.
@click.option('--beta', default=0.2, help='Averaging factor for on-policy and off-policy targets.', type=float)  # 0.5
@click.option('--scale-actions', default=True, help="Scale actions.", type=bool)
# @click.option('--layers', default=(256,128,64), help='Duplicate action-parameter inputs.')
#               # cls=ClickPythonLiteralOption)
@click.option('--save-freq', default=0, help='How often to save models (0 = never).', type=int)
@click.option('--save-dir', default="results/soccer", help='Output directory.', type=str)
@click.option('--title', default="PDQN", help="Prefix of output files", type=str)
def run(seed, episodes, batch_size, gamma, inverting_gradients, initial_memory_threshold, replay_memory_size,
        epsilon_steps, learning_rate_actor, learning_rate_actor_param, title, epsilon_start, epsilon_final, clip_grad,
        beta,
        scale_actions, evaluation_episodes, update_ratio, save_freq, save_dir):
    if save_freq > 0 and save_dir:
        save_dir = os.path.join(save_dir, title + "{}".format(str(seed)))
        # print(";;;;;;;;;;;;;")
        os.makedirs(save_dir, exist_ok=True)

    env = UAV_MEC()
    dir = os.path.join(save_dir, title)
    # env = Monitor(env, directory=os.path.join(dir, str(seed)), video_callable=False, write_upon_reset=False, force=True)
    np.random.seed(seed)

    agent_class = PDQNNStepAgent

    agent = agent_class(
        env.n_features, env.n_actions,
        actor_kwargs={
            'activation': "relu", },
        actor_param_kwargs={
            'activation': "relu", },
        batch_size=batch_size,
        learning_rate_actor=learning_rate_actor,  # 0.0001
        learning_rate_actor_param=learning_rate_actor_param,  # 0.001
        epsilon_initial=epsilon_start,
        epsilon_steps=epsilon_steps,
        epsilon_final=epsilon_final,
        gamma=gamma,  # 0.99
        clip_grad=clip_grad,
        beta=beta,
        initial_memory_threshold=initial_memory_threshold,
        replay_memory_size=replay_memory_size,
        inverting_gradients=inverting_gradients,
        seed=seed)
    print(agent)
    network_trainable_parameters = sum(p.numel() for p in agent.actor.parameters() if p.requires_grad)
    network_trainable_parameters += sum(p.numel() for p in agent.actor_param.parameters() if p.requires_grad)
    print("Total Trainable Network Parameters: %d" % network_trainable_parameters)
    max_steps = 15000
    returns = []
    action_save = []
    param_save = []
    moving_avg_rewards = []
    start_time_train = time.time()
    # log_dir = os.path.split(os.path.abspath(__file__))[0] + "/logs/train/" + title + "__" + str(seed) + " " + SEQUENCE
    # writer = SummaryWriter(log_dir)

    for i_eps in range(episodes):
        if save_freq > 0 and save_dir and i_eps % save_freq == 0:
            agent.save_models(os.path.join(save_dir, str(i_eps)))
        info = {'status': "NOT_SET"}
        state = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)

        act, act_param, all_action_parameters = agent.act(state)
        action = env.find_action(act)
        episode_reward = 0.
        action_tmp = []
        param_tmp = []
        transitions = []
        for i_step in range(max_steps):
            # print(act,"........................")
            next_state, reward, terminal = env.step(action, act_param)
            # print(reward, "........................")
            next_state = np.array(next_state, dtype=np.float32, copy=False)
            action_tmp.append(action)
            param_tmp.append(act_param)
            next_act, next_act_param, next_all_action_parameters = agent.act(next_state)
            next_action = env.find_action(next_act)
            transitions.append([state, np.concatenate(([act], all_action_parameters.data)).ravel(), reward,
                                next_state, np.concatenate(([next_act],
                                                            next_all_action_parameters.data)).ravel(), terminal])

            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            action = next_action
            state = next_state
            episode_reward += reward
            # print("transitions", transitions[0][2])
            if terminal:
                break
        agent.end_episode()
        print(i_eps, "episode_reward：", episode_reward)

        # calculate n-step returns
        n_step_returns = compute_n_step_returns(transitions, gamma)
        for t, nsr in zip(transitions, n_step_returns):
            t.append(nsr)
            agent.replay_memory.append(state=t[0], action_with_param=t[1], reward=t[2], next_state=t[3],
                                       done=t[5], n_step_return=nsr)

        n_updates = int(update_ratio * i_step)
        for _ in range(n_updates):
            agent._optimize_td_loss()

        if i_eps % 2 == 0:
            agent.actor_target.load_state_dict(agent.actor.state_dict())
            agent.actor_param_target.load_state_dict(agent.actor_param.state_dict())

        returns.append(episode_reward)
        action_save.append(action_tmp)
        param_save.append(param_tmp)
    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()
    np.save(file="data/Reward.npy", arr=returns)
    # np.save(file="data/11/action_save.npy", arr=action_save)
    # np.save(file="data/11/param_save.npy", arr=param_save)

    # end_time_train = time.time()
    if save_freq > 0 and save_dir:
        agent.save_models(os.path.join(save_dir, str(i_eps)))

    # if evaluation_episodes > 0:
    #     print("Evaluating agent over {} episodes".format(evaluation_episodes))
    #     agent.epsilon_final = 0.
    #     agent.epsilon = 0.
    #     agent.noise = None
    #     agent.actor.eval()
    #     agent.actor_param.eval()
    #     start_time_eval = time.time()
    #     evaluation_results = evaluate(env, agent, evaluation_episodes)  # returns, timesteps, goals
    #     np.save(os.path.join(dir, title + "{}e".format(str(seed))), evaluation_results)

    print(agent)


def compute_n_step_returns(episode_transitions, gamma):
    n = len(episode_transitions)
    n_step_returns = np.zeros((n,))
    n_step_returns[n - 1] = episode_transitions[n - 1][2]  # Q-value is just the final reward
    for i in range(n - 2, 0, -1):
        reward = episode_transitions[i][2]
        target = n_step_returns[i + 1]
        n_step_returns[i] = reward + gamma * target
    return n_step_returns


if __name__ == '__main__':
    run()
