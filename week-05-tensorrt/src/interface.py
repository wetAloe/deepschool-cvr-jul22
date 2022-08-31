import numpy as np
from tensorrt.tensorrt import ICudaEngine, IExecutionContext
from .common import get_engine, allocate_buffers, do_inference


class TRTModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Загружаем engine
        self.engine: ICudaEngine = get_engine(model_name)
        assert self.engine is not None

        # инициализируем контекст выполнения
        self.context: IExecutionContext = self.engine.create_execution_context()
        # прибито гвоздями: 0 - input, 1 - output
        # (1, 3, 224, 224)
        self.input_shape = self.engine.get_binding_shape(0)[1:]
        self.output_shape = self.engine.get_binding_shape(1)[1:]

    def __call__(self, data: np.ndarray) -> np.ndarray:
        batch_size = data.shape[0]
        # Указываем динамический размер входной `свзязки`
        self.context.set_binding_shape(0, (batch_size, *self.input_shape))

        # выделяем память, определяем входные-выходные `связки`
        inputs, outputs, bindings, stream = allocate_buffers(self.engine, batch_size)
        # закидиваем данные на хост в нужное место
        inputs[0].host = data

        # выполняем вычисления
        output = do_inference(
            self.context,
            bindings=bindings,
            inputs=inputs,
            outputs=outputs,
            stream=stream,
            batch_size=batch_size,
        )[0]  # только один выход

        # готовим данные к выходу
        output = output.reshape(-1, *self.output_shape)
        return output
