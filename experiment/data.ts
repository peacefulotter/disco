import { data, Task } from '@epfml/discojs-node'
import { getDataset } from './core/dataset-node'

export async function loadData(task: Task): Promise<data.DataSplit> {
    // const dir = path.join('datasets', task.taskID)
    // const files = { train: 'wiki.train.raw.pp', valid: 'wiki.valid.raw.pp' }

    const getData = async (split: string) => {
        const dataset = await getDataset(split)
        return await data.TextData.init(dataset, task)
    }

    return {
        train: await getData('train'),
        validation: await getData('valid'),
    }
}
