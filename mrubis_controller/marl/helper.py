import datetime

import matplotlib.pyplot as plt
import numpy


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
    for index, agent in enumerate(shop_distribution):
        avgs = []
        for i in range(episode):
            filter_counts = list((filter(lambda x: x != -1, [count_data[shop][i] for shop in agent])))
            avg = -1 if len(filter_counts) == 0 else sum(filter_counts) / len(filter_counts)
            avgs.append(avg)
        data['Agent ' + str(index)] = avgs
    build_plot(data, f"Tries till {episode}", f"{base_dir}/tries")
    write_data(len(list(count_data.values())[0]), data, 'tries', base_dir)


def build_loss_plot(base_dir, loss_data, episode):
    prepared_data = {}
    for loss in loss_data:
        for key, value in loss.items():
            if key not in prepared_data:
                prepared_data[key] = []
            if value:
                prepared_data[key].append(value[0])

    for agent in prepared_data.keys():
        title = f"Loss for agent #{agent} (episode {episode})"
        for network in ['actor', 'critic']:
            data = {
                network: [loss[network] for loss in prepared_data[agent]]
            }
            path = f"{base_dir}/{network}loss_agent_{agent}"
            build_plot(data, title, path)

def build_reward_plot(base_dir, reward_data, episode, shop_distribution):
    data = {}
    for agent in shop_distribution:
        for shop in agent:
            data[shop] = [d[0][shop] for d in reward_data]
    build_plot(data, f"Rewards till {episode}", f"{base_dir}/reward")
    write_data(len(reward_data), data, 'reward', base_dir)


def build_regret_plot(base_dir, regret_data):

    title = f"Regret"
    path = f"{base_dir}/regret"
    # write_data(len(regret_data), regret_data, 'regret', base_dir)
    print(regret_data)
    with open(f"{path}.txt", "w+") as file:
        for i in regret_data:
            file.write(str(i) + '\n')

    colors = ['#B1063A', '#134293', '#058B79', '#DD9108', '#009e61',
              '#5a6065', '#00799e', '#f6ba00', '#b10639', '#dd6108']
    plt.figure(figsize=(8, 6))
    plt.title(title, fontsize=16)

    x = numpy.array([i for i in range(len(regret_data))])

    plt.plot(x, regret_data, colors.pop(), label="test")
    plt.legend()
    plt.savefig(path)
    plt.clf()
    plt.close()


def build_plot(data, title, path):
    colors = ['#B1063A', '#134293', '#058B79', '#DD9108', '#009e61',
              '#5a6065', '#00799e', '#f6ba00', '#b10639', '#dd6108']
    plt.figure(figsize=(8, 6))
    plt.title(title, fontsize=16)
    # axes = plt.gca()
    # axes.set_ylim([0, 20])
    for label, d in data.items():
        x = numpy.array([i for i in range(len(d))])
        d[0] = d[0] if d[0] > 0 else 0
        for index, value in enumerate(d):
            if value < 0:
                d[index] = d[index - 1]
        plt.plot(x, [float(value) for value in d], colors.pop(), label=label)
    plt.legend()
    plt.savefig(path)
    plt.clf()
    plt.close()


def write_data(count, data, title, base_dir):
    with open(f"{base_dir}/{title}.txt", "w+") as file:
        for i in range(count):
            data_string = ''.join((str(data[shop][i]) if data[shop][i] > 0 else '') + ',' for shop in data)
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
