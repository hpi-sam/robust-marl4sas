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
    build_plot(data, f"Rewards till {episode}", f"{base_dir}/reward")
    write_data(len(reward_data), data, 'reward', base_dir)


def build_count_plot(base_dir, count_data, episode, shop_distribution):
    data = {}
    for agent in shop_distribution:
        for shop in agent:
            data[shop] = count_data[shop]
    build_plot(data, f"Tries till {episode}", f"{base_dir}/tries")
    write_data(len(count_data), data, 'tries', base_dir)


def build_loss_plot(base_dir, loss_data, episode, shop_distribution):
    for index, agent in enumerate(shop_distribution):
        title = f"Loss for agent #{index} (episode {episode})"
        agents_loss = [actor_critic for d in loss_data for actor_critic in d[index]]
        for network in ['actor', 'critic']:
            data = {
                network: [loss[network] for loss in agents_loss]
            }
            path = f"{base_dir}/{network}loss_agent_{index}"
            build_plot(data, title, path)


def build_plot(data, title, path):
    plt.title(title, fontsize=12)
    for label, d in data.items():
        plt.plot(d, label=label)
    plt.legend()
    plt.savefig(path)
    plt.clf()


def write_data(count, data, title, base_dir):
    with open(f"{base_dir}/{title}.txt", "a+") as file:
        for i in range(count):
            data_string = ''
            for shop in data:
                data_string += str(data[shop][i]) + ','
            file.write(data_string + '\n')
