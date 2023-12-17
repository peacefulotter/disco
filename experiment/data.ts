import { readdir } from 'fs/promises'
import path from 'path'
import { data, Task, node } from '@epfml/discojs-node/src'
import { TextConfig, TextSource } from './src/dataset/data_loader'
import { TOKENIZED_FILE_EXTENSION } from './core/preprocess'

async function getDatasetSource(root: string, splits: (keyof TextSource)[]): Promise<TextSource> {
    console.log('Preprocessed dataset located at:', root)
    const files = await readdir(root)
    console.log('Found', files.length, 'files in dataset under', root)
    return Object.fromEntries(
        splits.map((split) => {
            const s = split == 'train' ? 'test' : split // TODO: remove this, just for testing
            const files_path = files
                .filter((f) => f.endsWith(TOKENIZED_FILE_EXTENSION) && f.includes(s))
                .map((f) => path.join(root, f))
            return [split, files_path]
        })
    ) as TextSource
}

export async function loadData(task: Task): Promise<data.DataSplit> {
    // const dir = path.join('datasets', task.taskID)
    // const files = { train: 'wiki.train.raw.pp', valid: 'wiki.valid.raw.pp' }

    // const getData = async (split: string) => {
    //     const dataset = await getDataset(split)
    //     return await data.TextData.init(dataset, task)
    // }

    const root = path.join(import.meta.dir, 'datasets', 'wikitext-103')
    const source = await getDatasetSource(root, ['train', 'validation'])
    const config: Partial<TextConfig> = {}

    return await new node.data.NodeTextLoader(task).loadAll(source, config)
    // return {
    //     train: await getData('train'),
    //     validation: await getData('valid'),
    // }
}
