import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.python.client import device_lib


def parse_desc(desc) -> str:
	"""Return parsed value like '1080 Ti', '1080', '1070', '1070'"""
	return desc.split("GTX ")[1].split(",")[0]


def set_gpu_device(gpu_device):
	"""Set GPU device by specifying its name."""
	assert gpu_device in ['1080 Ti', '1080', '1070', '1070']
	all_devices = device_lib.list_local_devices()
	selected_devices = [_ for _ in all_devices if _.device_type == "GPU"]
	selected_devices = [
		_ for _ in selected_devices if parse_desc(_.physical_device_desc) == gpu_device
	]
	device_name = selected_devices[0].name.replace("/device", "/physical_device")
	physical_devices = tf.config.list_physical_devices('GPU')
	physical_selected = [_ for _ in physical_devices if _.name == device_name]
	assert physical_selected; "No physical devices found!"
	print("Using selected GPU", physical_selected)
	tf.config.set_visible_devices(physical_selected, 'GPU')


def get_optimizer(name: str):
	if name == "yogi":
		return tfa.optimizers.Yogi()
	return name
