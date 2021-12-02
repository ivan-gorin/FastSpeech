from datetime import datetime
import importlib
from datetime import datetime
import numpy as np


class WanDBWriter:
    def __init__(self, config):
        self.writer = None
        self.selected_module = ""

        import wandb
        wandb.login()

        if config['trainer'].get('wandb_project') is None:
            raise ValueError("please specify project name for wandb")

        wandb.init(
            project=config['trainer'].get('wandb_project'),
            config=config
        )
        self.wandb = wandb

        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar("steps_per_sec", 1 / duration.total_seconds())
            self.timer = datetime.now()

    def scalar_name(self, scalar_name):
        return f"{scalar_name}_{self.mode}"

    def add_scalar(self, scalar_name, scalar):
        self.wandb.log({
            self.scalar_name(scalar_name): scalar,
        }, step=self.step)

    def add_scalars(self, tag, scalars):
        self.wandb.log({
            **{f"{scalar_name}_{tag}_{self.mode}": scalar for scalar_name, scalar in scalars.items()}
        }, step=self.step)

    def add_image(self, scalar_name, image):
        self.wandb.log({
            self.scalar_name(scalar_name): self.wandb.Image(image)
        }, step=self.step)

    def add_audio(self, scalar_name, audio, sample_rate=None):
        audio = audio.detach().cpu().numpy().T
        self.wandb.log({
            self.scalar_name(scalar_name): self.wandb.Audio(audio, sample_rate=sample_rate)
        }, step=self.step)

    def add_text(self, scalar_name, text):
        self.wandb.log({
            self.scalar_name(scalar_name): self.wandb.Html(text)
        }, step=self.step)

    def add_histogram(self, scalar_name, hist, bins=None):
        hist = hist.detach().cpu().numpy()
        np_hist = np.histogram(hist, bins=bins)
        if np_hist[0].shape[0] > 512:
            np_hist = np.histogram(hist, bins=512)

        hist = self.wandb.Histogram(
            np_histogram=np_hist
        )

        self.wandb.log({
            self.scalar_name(scalar_name): hist
        }, step=self.step)

    def add_images(self, scalar_name, images):
        raise NotImplementedError()

    def add_pr_curve(self, scalar_name, scalar):
        raise NotImplementedError()

    def add_embedding(self, scalar_name, scalar):
        raise NotImplementedError()





class TensorboardWriter:
    def __init__(self, log_dir, enabled):
        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = (
                    "Warning: visualization (Tensorboard) is configured to use, but currently not installed on "
                    "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to "
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                )
                raise Exception(message)

        self.step = 0
        self.mode = ""

        self.tb_writer_ftns = {
            "add_scalar",
            "add_scalars",
            "add_image",
            "add_images",
            "add_audio",
            "add_text",
            "add_histogram",
            "add_pr_curve",
            "add_embedding",
        }
        self.tag_mode_exceptions = {"add_histogram", "add_embedding"}
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar("steps_per_sec", 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = "{}/{}".format(tag, self.mode)
                    add_data(tag, data, self.step, *args, **kwargs)

            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError(
                    "type object '{}' has no attribute '{}'".format(
                        self.selected_module, name
                    )
                )
            return attr
