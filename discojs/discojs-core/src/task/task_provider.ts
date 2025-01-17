import { tf, Task } from '..'

export interface TaskProvider {
    getTask: () => Task
    getModel: () => Promise<tf.LayersModel>
}

export function isTaskProvider(obj: any): obj is TaskProvider {
    if (
        'getModel' in obj &&
        typeof obj.getModel === 'function' &&
        'getTask' in obj &&
        typeof obj.getTask === 'function'
    ) {
        return true
    } else {
        return false
    }
}
