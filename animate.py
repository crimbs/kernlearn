import json
from kernlearn.plot_utils import *


# Sheep
def plot_sheep():
    sheep_data = json.load(open("data/sheep/json/data1.json", "r"))
    sheep_path = "figures/sheep"
    x = sheep_data["x"]
    u = sheep_data["u"]
    # uv = sheep_data["uv"]
    v = sheep_data["v"]
    t = sheep_data["t"]
    trajectory_plot(sheep_path, x, v)
    # animate_trajectory(sheep_path, u, uv, t)


def plot_fish():
    fish_data = json.load(open("data/fish/json/processed_data.json", "r"))
    fish_path = "figures/fish"
    x = fish_data["x"]
    v = fish_data["v"]
    t = fish_data["t"]
    trajectory_plot(fish_path, x, v)
    animate_trajectory(fish_path, x, v, t)
    plot_3d(fish_path, x, t)
    anim_plot_3d_rotating(fish_path, x, t)
    anim_plot_3d_time(fish_path, x, t)
