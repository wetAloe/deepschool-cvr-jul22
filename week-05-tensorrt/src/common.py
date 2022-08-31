import os
from typing import Optional, Any, Tuple
import pycuda.driver as cuda
import tensorrt as trt
from tensorrt.tensorrt import IBuilderConfig, IOptimizationProfile, INetworkDefinition

TRT_LOGGER = trt.Logger()

# HOST - наш компутер(процессор, оперативка и пр)
# DEVICE - наша видеокарта


class HostDeviceMem:
    """
    Simple helper data class that's a little nicer to use than a 2-tuple.
    Связка ссылок на память
    """

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine, batch_size):
    """
    Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
    Выделяем память на хосте и на девайсе
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        shape = engine.get_binding_shape(binding)
        if shape[0] == -1:
            size = abs(trt.volume(engine.get_binding_shape(binding)) * batch_size)
        else:
            size = trt.volume(engine.get_binding_shape(binding))

        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    """
    This function is generalized for multiple inputs/outputs.
    inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # перекидываем данные с хоста на девайс
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # выполняем вычисления
    context.execute_async(
        batch_size=batch_size, bindings=bindings, stream_handle=stream.handle
    )
    # перекидываем данные с девайса на хост
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # синхронизируемся
    stream.synchronize()
    return [out.host for out in outputs]


def build_engine(
    onnx_file_path: str,
    trt_file_path: Optional[str] = None,
    fp16: bool = False,
    int8: bool = False,
    int8_calibrator: Optional[Any] = None,
    max_batch_size: int = 1,
    max_workspace_size: int = 1 << 30,  # 1GB гибибайт)
    min_shape: Optional[Tuple[int]] = None,
    opt_shape: Optional[Tuple[int]] = None,
    max_shape: Optional[Tuple[int]] = None,
):
    """
    Класс построения engine-а на основе ONNX файла.

    Наглядный пример того, как работает конвертация в TensorRT!
    """
    # создаём билдер - основной класс
    builder = trt.Builder(TRT_LOGGER)
    # создаём конфиг конвертации
    config: IBuilderConfig = builder.create_builder_config()
    # создаём `определение` сети(описание графа) из которого билдим engine
    network: INetworkDefinition = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    # создаём onnx-парсер, связанный с сетью
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # загружаем и парсим onnx файл
    if not os.path.exists(onnx_file_path):
        print(f"ONNX file {onnx_file_path} not found.")
        exit(0)
    print(f"Loading ONNX file from path {onnx_file_path}...")
    with open(onnx_file_path, "rb") as model:
        print("Beginning ONNX file parsing")
        parser.parse(model.read())

    # работаем с динамическим размером батча
    if min_shape is not None:
        # создаём оптимизационный профиль и закидываем его в конфиг
        profile: IOptimizationProfile = builder.create_optimization_profile()
        profile.set_shape(network.get_input(0).name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    # определяем конфиг
    config.max_workspace_size = max_workspace_size

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = int8_calibrator

    builder.max_batch_size = max_batch_size

    # билдим engine
    engine = builder.build_engine(network, config)

    # сохраняем engine на диск
    print("Completed creating Engine")
    if trt_file_path:
        with open(trt_file_path, "wb") as f:
            f.write(engine.serialize())
        print("Engine saved.")
    return engine


def get_engine(engine_file_path: str):
    """
    Return TensorRT engine by file name.
    """
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


# IInt8MinMaxCalibrator
# IInt8EntropyCalibrator2
class EntropyCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, data, cache_file, batch_size=1):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8MinMaxCalibrator.__init__(self)

        self.cache_file = cache_file

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.data = data
        self.batch_size = batch_size
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print(
                "Calibrating batch {:}, containing {:} images".format(
                    current_batch, self.batch_size
                )
            )

        batch = self.data[
            self.current_index : self.current_index + self.batch_size
        ].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
