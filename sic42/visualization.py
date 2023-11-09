import io
import os
import subprocess
import shutil
import contextlib
from base64 import b64encode


import pygame
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import Video, HTML
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict

from sic42.utils import delnodes



COLOR_MAPPER = {0: "black", 1: "blue", 2: "red", 3: "green", 4: "white", 5: "yellow"}


def int_to_bgr(
    frame: np.ndarray
) -> np.ndarray:
    """
    convert single-integer color code to RGB-like encoding for video frames

    frame: numpy array depicting game board
    """
    B = ((frame == 4) | (frame == 1)) * 255
    G = ((frame == 4) | (frame == 3) | (frame == 5)) * 255
    R = ((frame == 4) | (frame == 2) | (frame == 5)) * 255
    return np.stack((B, G, R), axis=2).astype('uint8')


def torgb(x):
    B = (x == 4 or x == 1) * 255
    G = (x == 4 or x == 3 or x == 5) * 255
    R = (x == 4 or x == 2 or x == 5) * 255
    return (R, G, B)


def get_max_intensity(frame, fac):
    return np.array([[0 if len(f) == 0 else fac * max([max([l for l, d in v]) for v in f.values()]) for f in r] for r in frame])


def to_grayscale(frame):
    return np.stack((frame, frame, frame), axis=2).astype('uint8')


def scale(
    arr: np.ndarray,
    n: int
) -> np.ndarray:
    """
    upscales frame

    arr: numpy array
    n: factor by which to upscale arr
    """
    return np.kron(arr, np.ones((n, n), dtype='uint8'))


def save_frames(frames, path):
    os.makedirs(path)
    ndigs = int(np.ceil(np.log10(len(frames))))+1
    for i, f in enumerate(frames):
        numstr = str(i)
        suffix = (ndigs - len(numstr)) * '0' + numstr
        Image.fromarray(f).save(os.path.join(path, f'frame_{suffix}.jpg'))


def add_pheromone_map(
    pheromone_frames: List,
    scaling_factor,
    fps,
    outfname,
    max_initial_intensity
):
    fac = 255 / max_initial_intensity
    frames = [to_grayscale(scale(get_max_intensity(f, fac), scaling_factor)) for f in pheromone_frames]
    tmppath = f'{outfname}_temp'
    outpath = f'{outfname}.mp4'
    delnodes([tmppath, 'leftside.mp4', 'rightside.mp4'])
    save_frames(frames, tmppath)
    with contextlib.redirect_stdout(None):
        os.system(f"ffmpeg -framerate {fps} -pattern_type glob -i '{tmppath}/*.jpg' rightside.mp4 -loglevel quiet")
    os.rename(outpath, 'leftside.mp4')
    with contextlib.redirect_stdout(None):
        os.system(
            f'ffmpeg -y -loglevel panic -i leftside.mp4 -i rightside.mp4 -filter_complex "[0]pad=iw+5:color=white[left];[left][1]hstack=inputs=2" {outpath}'
        )
    delnodes([tmppath, 'leftside.mp4', 'rightside.mp4'])
    


def generate_video(
    frames: List[np.ndarray],
    pheromone_frames: List=None,
    scaling_factor: int=1,
    fps: int=10,
    pheromone_map: bool=False,
    outfname: str='output',
    max_initial_intensity: int=None
) -> None:
    """
    convert list of frames to video file

    frames: list of numpy arrays representing game board states
    fps: number of frames per seconds
    outfname: name of video file (without file suffix)
    """
    frames = [scale(frame, scaling_factor) for frame in frames]
    frames = [int_to_bgr(f) for f in frames]
    tmppath = f'{outfname}_frames'
    if os.path.exists(tmppath):
        shutil.rmtree(tmppath)
    outfname_full = f'{outfname}.mp4'
    delnodes([outfname_full])
    save_frames(frames, tmppath)
    with contextlib.redirect_stdout(None):
        os.system(f"ffmpeg -framerate {fps} -pattern_type glob -i '{tmppath}/*.jpg' {outfname_full} -loglevel quiet")
    delnodes([tmppath])
    if pheromone_map:
        add_pheromone_map(pheromone_frames, scaling_factor, fps, outfname, max_initial_intensity)



def save_plots(simulation, path, plot_categories_mapper):
    metadata = simulation.metadata
    swarms = list(metadata[0].keys())
    props = list(metadata[0][swarms[0]].keys())
    plot_categories_path_mapper = dict()
    for plot_category in plot_categories_mapper.keys():
        plot_category_path = os.path.join(path, plot_category)
        plot_categories_path_mapper[plot_category] = plot_category_path
        os.makedirs(plot_category_path)
    for prop in props:
        for plot_category, indicator_function in plot_categories_mapper.items():
            if indicator_function(prop):
                plot_path = plot_categories_path_mapper[plot_category]
        for swarm in swarms:
            color = COLOR_MAPPER[simulation.environment.swarm_configs[swarm]['symbol']]
            timeseries = [x[swarm][prop] for x in metadata]
            plt.plot(timeseries, label=swarm, color=color)
        plt.title(prop)
        plt.legend(title='Swarm')
        plt.xlabel('Time')
        plt.ylabel(prop)
        plt.savefig(os.path.join(plot_path, f'{prop}.png'))
        plt.close()


class PygameVisualizer:
    def __init__(self, pygame_params, width, height, swarm_symbol_mapper, item_symbol_mapper, scale=1, record_video=False, show_graphs=False):
        self.pygame_params = pygame_params
        pygame.init()
        self.paused = False
        self.scale = scale
        if show_graphs:
            self.simulation_width = width * scale
            self.simulation_height = height * scale
            self.plot_width = 400  # or any desired width for the plot
            self.window = pygame.display.set_mode((self.simulation_width + self.plot_width, self.simulation_height))
        else:
            self.window = pygame.display.set_mode((width * scale, height * scale))
        
        icons_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icons')

        # Load and scale images for swarms
        self.swarm_symbol_mapper = swarm_symbol_mapper
        self.item_symbol_mapper = item_symbol_mapper
        self.swarm_colors = {swarm_name: torgb(symbol) for swarm_name, symbol in swarm_symbol_mapper.items()}
        swarm_image_scaler = (
            self.pygame_params['agent_scaling_factor'],
            self.pygame_params['agent_scaling_factor']
        )
        self.swarm_images = {
            swarm_name: self.load_and_scale_image(
                os.path.join(icons_path, 'agent.png'),
                self.swarm_colors[swarm_name],
                swarm_image_scaler
            ) for swarm_name in swarm_symbol_mapper.keys()
        }

        # Set colors and images for items
        self.item_colors = {item_name: torgb(symbol) for item_name, symbol in item_symbol_mapper.items()}        
        item_image_scaler = (
            self.pygame_params['item_scaling_factor'],
            self.pygame_params['item_scaling_factor']
        )
        self.item_images = dict()
        for item_name in item_symbol_mapper.keys():
            fn = os.path.join(icons_path, f'{item_name}.png')
            if os.path.exists(fn):
                self.item_images[item_name] = self.load_and_scale_image(fn, self.item_colors[item_name], item_image_scaler)
        
        self.record_video = record_video
        if self.record_video:
            self.frame_number = 0
            self.video_dir = 'video_frames'
            if not os.path.exists(self.video_dir):
                os.makedirs(self.video_dir)
                
        # Initialize plot update parameters
        self.plot_update_interval = self.pygame_params['plot_update_interval']
        self.frame_counter = 0
        self.cached_plot_surface = None


    def color_surface(self, surface, rgb):
        red, green, blue = rgb
        arr = pygame.surfarray.pixels3d(surface)
        arr[:,:,0] = red
        arr[:,:,1] = green
        arr[:,:,2] = blue
    
    def load_and_scale_image(self, image_path, color, size):
        original_image = pygame.image.load(image_path)
        self.color_surface(original_image, color)
        scaled_image = pygame.transform.scale(original_image, size)
        image_rect = scaled_image.get_rect()
        offset = (image_rect.width // 2, image_rect.height // 2)
        return {'image': scaled_image, 'offset': offset}

    def draw_agents(self, agents):
        for agent in agents:
            y, x = (coord * self.scale for coord in agent.loc)
            swarm_name = agent.swarm_name
            if swarm_name in self.swarm_images:
                image_data = self.swarm_images[swarm_name]
                image = image_data['image']
                offset_x, offset_y = image_data['offset']
                self.window.blit(image, (x - offset_x, y - offset_y))
            else:
                energy_level = agent.energy_level
                agent_color = self.swarm_colors.get(swarm_name, (255, 255, 255))
                radius = max(min(int(energy_level / 10), 20), 5) 
                pygame.draw.circle(self.window, agent_color, (x, y), radius)


    def draw_items(self, items):
        default_color = (200, 200, 200)  
        item_size = (10, 10)  
        for item in items:
            y, x = (coord * self.scale for coord in item.loc)
            item_type = item.name
            if item_type in self.item_images:
                image_data = self.item_images[item_type]
                image = image_data['image']
                offset_x, offset_y = image_data['offset']
                self.window.blit(image, (x - offset_x, y - offset_y))
            else:
                item_color = self.item_colors.get(item_type, default_color)
                pygame.draw.rect(self.window, item_color, (x, y, *item_size))
                

    def draw_pheromones(self, pheromones):
        # If you want a semi-transparent overlay, retain the following lines
        surface = pygame.Surface((self.window.get_width(), self.window.get_height()), pygame.SRCALPHA)
        surface.set_alpha(self.pygame_params['alpha'])  # Set the alpha level for the surface. 128 is semi-transparent.

        # Call the new method
        self.draw_pheromone_heatmap(surface, pheromones)
        
        self.window.blit(surface, (0, 0))


    def draw_pheromone_heatmap(self, surface, pheromones):
        for i, row in enumerate(pheromones):
            for j, cell in enumerate(row):
                if not cell:
                    continue

                # Assuming the cell structure is { 'pheromone_type': [(intensity, _), ...], ... }
                for pheromone_type, pheromone_data in cell.items():
                    if not pheromone_data:
                        continue

                    for pheromone_info in pheromone_data:
                        intensity, _ = pheromone_info
                        if intensity == 0:
                            continue

                        # Map intensity to a color (e.g., blue at intensity 0 to red at intensity 255)
                        color = self.intensity_to_color(intensity)

                        # Draw the rectangle
                        rect = pygame.Rect(j * self.scale, i * self.scale, self.scale, self.scale)
                        pygame.draw.rect(surface, color, rect)

    def intensity_to_color(self, intensity):
        # Linear interpolation between two colors based on intensity
        low_color = pygame.Color(0, 0, 255)  # Blue at intensity 0
        high_color = pygame.Color(255, 0, 0)  # Red at intensity 255
        factor = intensity / 255.0
        r = int(low_color.r + factor * (high_color.r - low_color.r))
        g = int(low_color.g + factor * (high_color.g - low_color.g))
        b = int(low_color.b + factor * (high_color.b - low_color.b))
        return pygame.Color(r, g, b)

    def get_clicked_agent(self, click_pos, agents):
        for agent in agents:
            y, x = (coord * self.scale for coord in agent.loc)
            distance = ((click_pos[0] - x) ** 2 + (click_pos[1] - y) ** 2) ** 0.5
            if distance < self.pygame_params['tooltip_max_distance']:
                return agent
        return None

    def get_agent_under_mouse(self, agents):
        mouse_pos = pygame.mouse.get_pos()
        return self.get_clicked_agent(mouse_pos, agents)

    def show_popup(self, agent):
        y, x = (coord * self.scale for coord in agent.loc)
        popup_width = self.pygame_params['popup_width']
        popup_height = self.pygame_params['popup_height']

        # Adjust x if the popup would be cut off on the right
        if x + popup_width > self.window.get_width():
            x -= popup_width

        # Adjust y if the popup would be cut off on the bottom
        if y + popup_height > self.window.get_height():
            y -= popup_height

        popup_rect = pygame.draw.rect(self.window, (255, 255, 255), (x, y, popup_width, popup_height))
        font = pygame.font.SysFont(None, self.pygame_params['popup_font_size'])
        energy_text = font.render(f"Energy: {agent.energy_level}", True, (0, 0, 0))
        self.window.blit(energy_text, (x + 10, y + 10))  

    def generate_plot_surface(self, stats): 
        DPI = self.pygame_params['DPI']
        #number_of_subplots = len(stats)
        number_of_subplots = len(next(iter(stats[0].values())))
        
        subplot_height_in_pixels = self.simulation_height / number_of_subplots
        subplot_height_in_inches = subplot_height_in_pixels / DPI
        figure_width_in_inches = self.plot_width / DPI
        figure_height_in_inches = subplot_height_in_inches * number_of_subplots

        fig, axes = plt.subplots(nrows=number_of_subplots, figsize=(figure_width_in_inches, figure_height_in_inches), sharex=True)
        
        # If there's only one subplot, make sure axes is a list for consistent indexing
        if number_of_subplots == 1:
            axes = [axes]
        
        swarms = list(stats[0].keys())
        props = list(stats[0][swarms[0]].keys())
        for ax, prop in zip(axes, props):
            for swarm in swarms:
                color = COLOR_MAPPER[self.swarm_symbol_mapper[swarm]]
                timeseries = [x[swarm][prop] for x in stats]
                ax.plot(timeseries, label=swarm, color=color)
            ax.set_title(prop)
            ax.legend(title='Swarm')
            ax.set_xlabel('Time')
            ax.set_ylabel(prop)
        
        plt.tight_layout()  # Adjust layout to prevent overlap
        
        # Convert matplotlib figure to a pygame surface
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=DPI)
        plt.close(fig)
        buf.seek(0)
        plot_surface = pygame.image.load(buf)
        
        return plot_surface

    def update(self, agents, items, pheromones=None, draw_pheromones_flag=True, stats=None):
        hovered_agent = None

        # Handle events first
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()
            # Detect spacebar press to toggle pause
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused

        # Check for an agent under the mouse
        hovered_agent = self.get_agent_under_mouse(agents)

        # Clear the screen
        self.window.fill((0, 0, 0))

        # Draw all elements
        if draw_pheromones_flag and pheromones is not None:
            self.draw_pheromones(pheromones)
        self.draw_agents(agents)
        self.draw_items(items)

        # If paused, draw the "Paused" text and handle the popup if an agent is hovered
        if self.paused:
            if hovered_agent:
                self.show_popup(hovered_agent)

            font = pygame.font.SysFont(None, 55)
            pause_text = font.render('Paused', True, (255, 255, 255))
            self.window.blit(pause_text, (self.window.get_width() // 2 - pause_text.get_width() // 2, 
                                        self.window.get_height() // 2 - pause_text.get_height() // 2))

        # Generate plot surface if stats_dict is provided
        if stats:
            if self.frame_counter % self.plot_update_interval == 0:
                self.cached_plot_surface = self.generate_plot_surface(stats)
            # Increment the frame counter
            self.frame_counter += 1
            self.frame_counter %= self.plot_update_interval
        
        # Use cached plot surface if available
        if self.cached_plot_surface:
            # Position the plot at the bottom-right corner of the window
            pos_x = self.simulation_width
            pos_y = 0
            self.window.blit(self.cached_plot_surface, (pos_x, pos_y))

        pygame.display.flip()  # Update the screen
       
        # Save the current frame as an image if record_video is True
        if self.record_video:
            frame_path = os.path.join(self.video_dir, f'frame_{self.frame_number:05d}.png')
            pygame.image.save(self.window, frame_path)
            self.frame_number += 1


def generate_video_from_frames(
    frames_path: str,
    output_filename: str,
    fps: int
):
    delnodes([output_filename])
    command = [
        'ffmpeg',
        '-framerate', str(fps),  # or another framerate if desired
        '-i', os.path.join(frames_path, 'frame_%05d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-loglevel', 'quiet',
        output_filename
    ]
    
    try:
        with contextlib.redirect_stdout(None):
            subprocess.run(command, check=True)  # check=True will raise CalledProcessError if ffmpeg fails
    except subprocess.CalledProcessError:
        print("Error generating the video.")
        return

    # If video generation was successful, delete the frames and the folder
    for filename in os.listdir(frames_path):
        os.remove(os.path.join(frames_path, filename))
    shutil.rmtree(frames_path)

