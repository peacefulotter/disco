import { tf, Task, TaskProvider } from '../index.node'

import { model } from '@epfml/gpt-tfjs'
const { GPTLMHeadModel } = model

export const configModels = {
    gpt2: {
        nLayer: 12,
        nHead: 12,
        nEmbd: 768,
        vocabSize: 50257,
        blockSize: 1024,
    },
    'gpt2-medium': {
        nLayer: 24,
        nHead: 16,
        nEmbd: 1024,
        vocabSize: 50257,
        blockSize: 1024,
    },
    'gpt2-large': {
        nLayer: 36,
        nHead: 20,
        nEmbd: 1280,
        vocabSize: 50257,
        blockSize: 1024,
    },
    'gpt2-xl': {
        nLayer: 48,
        nHead: 25,
        nEmbd: 1600,
        vocabSize: 50257,
        blockSize: 1024,
    },
    'gpt-mini': { nLayer: 6, nHead: 6, nEmbd: 192 },
    'gpt-micro': { nLayer: 4, nHead: 4, nEmbd: 128 },
    'gpt-nano': { nLayer: 3, nHead: 3, nEmbd: 48 },
} as const

const modelType: keyof typeof configModels = 'gpt-nano'
const modelConfig = configModels[modelType]

const config = {
    ...modelConfig,
    debug: false,
    verbose: false,
    modelType,
    batchSize: 8,
    blockSize: 128,
    lr: 0.001,
    shuffle: NaN,
    weightDecay: false,
    optimizer: 'adamw',
    embdDrop: 0.2,
    residDrop: 0.2,
    bias: true,
    vocabSize: 50257,
} as const

export const wikitext: TaskProvider = {
    getTask(): Task {
        return {
            taskID: 'wikitext-103',
            displayInformation: {
                taskTitle: 'wikitext-103-raw',
                summary: {
                    preview:
                        'In this challenge, we ask you to do next word prediction on a dataset of Wikipedia articles.',
                    overview:
                        'Wikitext-103-raw is a dataset comprising unprocessed text excerpts from Wikipedia articles, designed for tasks related to natural language processing and language modeling.',
                },
                limitations:
                    'The dataset may contain noise, inconsistencies, and unstructured content due to its raw nature, potentially posing challenges for certain NLP tasks.',
                tradeoffs:
                    'The raw format may lack structured annotations and may require additional preprocessing for specific applications.',
                dataFormatInformation:
                    'The dataset is organized as a large text file, with each line representing a segment of raw text from Wikipedia articles.',
                dataExampleText:
                    'An example excerpt from the dataset could be: "The history of artificial intelligence dates back to ancient times, with philosophical discussions on the nature of thought and reasoning."',
            },
            trainingInformation: {
                modelID: 'wikitext-103-raw-model',
                epochs: 10,
                roundDuration: 10,
                validationSplit: 0.2,
                batchSize: 10,
                modelCompileData: {
                    optimizer: 'sgd',
                    loss: 'categoricalCrossentropy',
                    metrics: ['categoricalCrossentropy'], // 'perplexity' doesnt exist
                },
                dataType: 'text',
                preprocessingFunctions: [
                    // preprocessing is done prior to training
                    // data.TextPreprocessing.Tokenize,
                    // data.TextPreprocessing.Padding,
                ],
                scheme: 'Federated',
                noiseScale: undefined,
                decentralizedSecure: true,
                minimumReadyPeers: 3,
                maxShareValue: 100,
            },
        }
    },

    async getModel(): Promise<tf.LayersModel> {
        const gpt = GPTLMHeadModel(config)
        console.log(typeof gpt.model)
        return gpt.model
    },
}
