import datetime
import numpy
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
    write_data(len(list(count_data.values())[0]), data, 'tries', base_dir)


def build_loss_plot(base_dir, loss_data, episode, shop_distribution):
    # TODO dont create plots based on shob_distribution but also on real number of items in loss_data
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
    colors = ['#009e61', '#5a6065', '#00799e', '#f6ba00', '#b10639', '#dd6108']
    plt.figure(figsize=(4,3))
    plt.title(title, fontsize=16)
    # axes = plt.gca()
    # axes.set_ylim([0, 20])
    for label, d in data.items():
        x = numpy.array([i for i in range(len(d))])
        plt.plot(x, [float(value) for value in d], colors.pop(), label=label)
    plt.legend()
    plt.savefig(path)
    plt.clf()


def write_data(count, data, title, base_dir):
    with open(f"{base_dir}/{title}.txt", "w+") as file:
        for i in range(count):
            data_string = ''.join(str(data[shop][i]) + ',' for shop in data)
            file.write(data_string + '\n')


def build_plot_from_file(titles, path, output_path, base_dir='./mrubis_controller/marl/data/logs'):
    data = {}
    with open(f'{base_dir}/{path}.txt') as file:
        lines = file.readlines()
        for index, title in enumerate(titles, start=1):
            data[title] = [line.split(',')[index] for line in lines]
    build_plot(data, 'Number of tries', f'{base_dir}/{output_path}.png')


if __name__ == "__main__":
    build_plot_from_file(['mrubis #1'], '2022_02_04_13_33/tries', 'robustness_shop')
