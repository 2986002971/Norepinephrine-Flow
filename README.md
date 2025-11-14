# Norepinephrine_Flow

This project appears to be related to reinforcement learning, likely involving `mujoco` environments and agents.

## Setup and Installation

To set up the development environment and install Python dependencies, use `uv`:

```bash
uv sync
```

## Running on a Headless Server

If you are running this project on a headless server (a server without a graphical display), you will need to install `Xvfb` (X Virtual Framebuffer). This is required because the project uses `mujoco` for environment rendering, which needs a display context even if you are not directly viewing the output.

First, install `Xvfb` on your system. For Debian/Ubuntu-based systems, you can use:

```bash
sudo apt-get update
sudo apt-get install -y xvfb libgl1-mesa-dri libgl1-mesa-glx libosmesa6 libegl1-mesa mesa-utils
```

Once `Xvfb` is installed, the project should automatically detect the headless environment and use a virtual display for rendering. You do not need to modify your run commands.
