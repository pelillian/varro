import importlib.util
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
spec = importlib.util.spec_from_file_location("varro.cython.fast_cram", os.path.join(dir_path, "fast_cram.cpython-37m-x86_64-linux-gnu.so"))
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
