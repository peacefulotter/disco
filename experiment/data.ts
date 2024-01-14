import { readdir } from 'fs/promises'
import path from 'path'
import { dataset, Task, node } from '@epfml/discojs-node/src'
import { TOKENIZED_FILE_EXTENSION } from './preprocess'

async function getDatasetSource(
    root: string,
    splits: (keyof dataset.loader.TextSource)[]
): Promise<dataset.loader.TextSource> {
    console.log('Preprocessed dataset located at:', root)
    const files = await readdir(root)
    return Object.fromEntries(
        splits.map((split) => {
            const splitFiles = files.filter(
                (f) => f.endsWith(TOKENIZED_FILE_EXTENSION) && f.includes(split)
            )

            console.log(
                'Found',
                splitFiles.length,
                'files in dataset for the',
                split,
                'split.'
            )

            const splitFilesPath = splitFiles.map((f) => path.join(root, f))
            return [split, splitFilesPath]
        })
    ) as dataset.loader.TextSource
}

export async function loadData(task: Task): Promise<dataset.DataSplit> {
    // TODO: Make this even more generic so that it works for any dataset
    // 1) replace wikitext-103 with task.id => need to document that task.id and dataset folder name should be the same then
    // 2) move getDatasetSource to core so that the web version can use it as well
    const root = path.join(import.meta.dir, 'datasets', 'wikitext-103')
    const source = await getDatasetSource(root, ['train', 'validation'])
    const config: Partial<dataset.loader.TextConfig> = {}

    return await new node.dataset.loader.NodeTextLoader(task).loadAll(
        source,
        config
    )
}
