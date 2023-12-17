import { tf } from '../'

/**
 * Convenient type for the common dataset type used in TF.js.
 */
export type Dataset<T extends tf.TensorContainer = tf.TensorContainer> = tf.data.Dataset<T>
