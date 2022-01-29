import datetime
import matplotlib.pyplot as plt


def get_current_time():
    now = datetime.datetime.now()
    return now.strftime('%Y_%m_%d_%H_%M')


def build_reward_plot(base_dir, reward_data, episode, shop_distribution):
    data = {}
    for agent in shop_distribution:
        for shop in agent:
            data[shop] = [d[0][shop] for d in reward_data]
    build_plot(data, f"Rewards till {episode}", f"{base_dir}/reward_{episode}")


def build_count_plot(base_dir, count_data, episode, shop_distribution):
    data = {}
    for agent in shop_distribution:
        for shop in agent:
            data[shop] = count_data[shop]
    build_plot(data, f"Tries till {episode}", f"{base_dir}/tries_{episode}")


def build_loss_plot(base_dir, loss_data, episode, shop_distribution):
    for index, agent in enumerate(shop_distribution):
        title = f"Loss for agent #{index} (episode {episode})"
        agents_loss = [actor_critic for d in loss_data for actor_critic in d[index]]
        for network in ['actor', 'critic']:
            data = {
                network: [loss[network] for loss in agents_loss]
            }
            path = f"{base_dir}/{network}loss_agent_{index}_{episode}"
            build_plot(data, title, path)


def build_plot(data, title, path):
    plt.title(title, fontsize=12)
    for label, d in data.items():
        plt.plot(d, label=label)
    plt.legend()
    plt.savefig(path)
    plt.clf()
