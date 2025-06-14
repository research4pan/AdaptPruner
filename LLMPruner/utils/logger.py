import os
import sys
import time
import codecs
import logging


class LoggerWithDepth():
    def __init__(self, env_name, config, root_dir = 'runtime_log', overwrite = True, setup_sublogger = True):
        if os.path.exists(os.path.join(root_dir, env_name)) and not overwrite:
            raise Exception("Logging Directory {} Has Already Exists. Change to another name or set OVERWRITE to True".format(os.path.join(root_dir, env_name)))
        
        self.env_name = env_name
        self.root_dir = root_dir
        self.log_dir = os.path.join(root_dir, env_name)
        self.overwrite = overwrite

        self.format = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                        "%Y-%m-%d %H:%M:%S")

        os.makedirs(self.log_dir, exist_ok=True)
        
        # Save Hyperparameters
        self.write_description_to_folder(os.path.join(self.log_dir, 'description.txt'), config)
        self.checkpoint_path = os.path.join(self.log_dir, 'compressed_model_checkpoint.bin')

        if setup_sublogger:
            sub_name = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
            self.setup_sublogger(sub_name, config)

    def setup_sublogger(self, sub_name, sub_config):
        self.sub_dir = os.path.join(self.log_dir, sub_name)
        os.makedirs(self.sub_dir, exist_ok=True)

        self.write_description_to_folder(os.path.join(self.sub_dir, 'description.txt'), sub_config)
        with open(os.path.join(self.sub_dir, 'train.sh'), 'w') as f:
            f.write('python ' + ' '.join(sys.argv))

        # Setup File/Stream Writer
        log_format=logging.Formatter("%(asctime)s - %(levelname)s :       %(message)s", "%Y-%m-%d %H:%M:%S")
        
        self.writer = logging.getLogger()
        if not self.writer.hasHandlers():
            fileHandler = logging.FileHandler(os.path.join(self.sub_dir, "training.log"))
            fileHandler.setFormatter(log_format)
            self.writer.addHandler(fileHandler)

            consoleHandler = logging.StreamHandler()
            consoleHandler.setFormatter(log_format)
            self.writer.addHandler(consoleHandler)
        
        self.writer.setLevel(logging.INFO)

        # Checkpoint
        self.checkpoint_path = os.path.join(self.sub_dir, 'output_model.bin')      
        self.lastest_checkpoint_path = os.path.join(self.sub_dir, 'latest_model.bin')   

    def log(self, info):
        self.writer.info(info)

    def write_description_to_folder(self, file_name, config):
        with codecs.open(file_name, 'w') as desc_f:
            desc_f.write("- Training Parameters: \n")
            for key, value in config.items():
                desc_f.write("  - {}: {}\n".format(key, value))
