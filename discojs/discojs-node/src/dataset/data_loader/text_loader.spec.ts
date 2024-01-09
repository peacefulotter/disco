import fs from 'fs'
import path from 'path'
import { expect } from 'chai'
import { encode } from 'gpt-tokenizer/model/text-davinci-003'

import { tf, node, Task } from '../..'
import { TextConfig } from '@epfml/discojs-core/src/dataset/data_loader'
import {
    TokenizedDataset,
    TokenizedIterResult,
} from '@epfml/discojs-core/src/dataset/data_loader/text_loader'

const datasetsFolder = path.join(
    /* @ts-ignore */
    import.meta.dir,
    '../../../../../experiment',
    'datasets',
    'wikitext-103'
)

const inputFiles = {
    train: [path.join(datasetsFolder, 'test.tokens')],
}

const wikitextTask: Task = {
    taskID: 'wikitext-103',
    displayInformation: {},
    trainingInformation: {},
} as Task

const config: TextConfig & { batchSize: number } = {
    blockSize: 3,
    batchSize: 2,
    vocabSize: 50257,
}

const getTokenizedSample = async () => {
    const loaded = await new node.dataset.loader.NodeTextLoader(wikitextTask).loadAll(
        inputFiles,
        config
    )
    const ds = loaded.train.dataset as TokenizedDataset
    const iter = await ds.batch(config.batchSize).iterator()
    const { value, done } = (await iter.next()) as TokenizedIterResult
    return { value, done }
}

/**
 * Reads the RAW dataset (not preprocessed) and tokenizes the equivalent of the first batch.
 */
const getRawTokenizedSample = async () => {
    const size = config.batchSize * (config.blockSize + 1)
    const wikiRaw = fs.createReadStream(path.join(datasetsFolder, 'test'), {
        encoding: 'utf8',
        start: 0,
        end: size * 3, // * 3 to account for spaces between words and tabs
    })
    const iter = wikiRaw.iterator()
    const { value: chunk } = await iter.next()
    const tokens = encode(chunk).slice(0, size)
    return tokens
}

describe('text loader', () => {
    it('loads a batched sample', async () => {
        const { value, done } = await getTokenizedSample()
        const { xs, ys } = value
        const x = xs.flatten()
        const y = ys.argMax(2).flatten() // get indices of max values along last axis

        expect(xs.shape).to.eql([config.batchSize, config.blockSize])
        expect(ys.shape).to.eql([config.batchSize, config.blockSize, config.vocabSize])
        expect(x.equal([220, 198, 796, 347, 2852, 353]))
        expect(y.equal(tf.tensor([198, 796, 5199, 2852, 353, 796])))
        expect(done).to.eql(false)
    })

    it('dataset is tokenized properly', async () => {
        const tokens = await getRawTokenizedSample()
        const { value } = await getTokenizedSample()
        /**
         * Flatten the batch by taking the first token in x and the rest in y, since y is x shifted by 1 + 1 token
         * e.g. [a, b, c, d, e, f] -> x = [a, b, c, d, e] and y = [b, c, d, e, f]
         * thus x[0] + y = [a, b, c, d, e, f]
         **/
        const { xs, ys } = value
        const x = await xs.array()
        const y = await (ys.argMax(2) as tf.Tensor2D).array() // get indices of max values along last axis
        const arr = []
        for (let i = 0; i < config.batchSize; i++) {
            arr.push(x[i][0], ...y[i])
        }
        expect(arr).to.eql(tokens)
    })
})
