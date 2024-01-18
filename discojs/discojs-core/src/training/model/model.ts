import { tf, dataset, Task } from '../..'
import { Trainer } from '../trainer/trainer'

/**
 * Convenient interface to a TF.js model allowing for custom fit functions, while keeping
 * the model object compatible with TF.js.
 */
// TODO: Would be neat to extend tf.LayersModel, but it would require a super() call with a
// config argument that is not available given only a tf.LayersModel...
export abstract class Model<ModelConfig = unknown> {
    constructor(
        public readonly task: Task<ModelConfig>,
        public readonly tfjs: tf.LayersModel
    ) {}

    abstract fit(trainer: Trainer, data: dataset.DataSplit): Promise<void>
}
