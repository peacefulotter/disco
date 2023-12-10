import fs from 'fs'
import path from 'path'
import Rand from 'rand-seed'

import { node, data, Task } from '@epfml/discojs-node'
import { getDataset } from './core/dataset-node'

const rand = new Rand('1234')

function shuffle<T, U>(array: T[], arrayTwo: U[]): void {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(rand.next() * (i + 1))
        const temp = array[i]
        array[i] = array[j]
        array[j] = temp

        const tempTwo = arrayTwo[i]
        arrayTwo[i] = arrayTwo[j]
        arrayTwo[j] = tempTwo
    }
}

function filesFromFolder(dir: string, folder: string): string[] {
    const f = fs.readdirSync(dir + folder)
    return f.map((file) => dir + folder + '/' + file)
}

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
