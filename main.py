import neat
import gym
import visualize
import os
import numpy as np
from time import sleep


env_name = "LunarLander-v2"
max_steps = 10000

def render(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = gym.make(env_name)
    observation = env.reset()
    for t in range(max_steps):
        env.render()
        output = net.activate(observation)
        observation, reward, done, info = env.step(np.argmax(output))
        sleep(0.01)
        if done:
            break
    env.close()

def eval(genomes, config):
    _, best_genome = genomes[0]
    best_score = -10000
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        env = gym.make(env_name)
        observation = env.reset()
        score = 0.0
        for t in range(max_steps):
            output = net.activate(observation)
            observation, reward, done, info = env.step(np.argmax(output))
            score += reward
            if done:
                break
        genome.fitness = score
        if score > best_score:
            best_genome = genome
            best_score = score
        env.close()

    render(best_genome, config)

def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    winner = p.run(eval, 50)

    print('\nBest genome:\n{!s}'.format(winner))

    visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    render(winner, config)


if __name__  == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)